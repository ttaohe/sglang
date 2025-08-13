import logging
import threading as td
from dataclasses import dataclass, field
from typing import Any, Optional, Type

from torch.cuda.streams import ExternalStream

from sglang.srt.server_args import PortArgs, ServerArgs
from .pdmux_context import (
    create_greenctx_stream_by_percent_py,
    initialize_stream_groups,
    get_stream_groups,
)

logger = logging.getLogger(__name__)


@dataclass
class SemiPDSchedulerSharedState:
    max_total_num_tokens: Optional[int] = None
    decode_model_runner: Optional[Any] = None
    decode_ready_event: td.Event = field(default_factory=td.Event)


class SchedulerSemiPDLauncher:
    @staticmethod
    def share_model_and_buffers(source_runner, target_runner):
        target_runner.model = source_runner.model

        src_pool = getattr(source_runner, "token_to_kv_pool", None)
        tgt_pool = getattr(target_runner, "token_to_kv_pool", None)
        if src_pool is not None and tgt_pool is not None:
            if hasattr(src_pool, "k_buffer") and hasattr(src_pool, "v_buffer"):
                setattr(tgt_pool, "k_buffer", getattr(src_pool, "k_buffer"))
                setattr(tgt_pool, "v_buffer", getattr(src_pool, "v_buffer"))
            elif hasattr(src_pool, "kv_buffer"):
                setattr(tgt_pool, "kv_buffer", getattr(src_pool, "kv_buffer"))

        if (
            hasattr(source_runner, "req_to_token_pool")
            and hasattr(target_runner, "req_to_token_pool")
            and hasattr(source_runner.req_to_token_pool, "req_to_token")
        ):
            target_runner.req_to_token_pool.req_to_token = (
                source_runner.req_to_token_pool.req_to_token
            )

    @staticmethod
    def init_streams(sm_prefill: float, sm_decode: float, tp_rank: int, *, engine_mode: str, gpu_id: int) -> tuple[ExternalStream, ExternalStream]:
        if engine_mode == "semipd":
            initialize_stream_groups(gpu_id)
            groups = get_stream_groups()
            # Return a reasonable default pair for bootstrapping
            # Prefill uses the exclusive prefill stream (group 0),
            # Decode will handle multi-capture internally over all decode streams.
            pstream = groups[0][0]
            dstream = groups[1][1]
            return pstream, dstream
        # Fallback to single pair
        pstream, dstream = create_greenctx_stream_by_percent_py(sm_prefill, sm_decode, tp_rank)
        return pstream, dstream

    @staticmethod
    def run_decode_thread(
        scheduler_cls: Type[Any],
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        dstream: ExternalStream,
        shared_state: SemiPDSchedulerSharedState,
    ):
        dscheduler = scheduler_cls(
            server_args, port_args, gpu_id, tp_rank, dp_rank, bypass_load_weight=False
        )
        logger.info("Decode scheduler init finished.")

        if dscheduler.enable_overlap:
            shared_state.decode_model_runner = dscheduler.tp_worker.worker.model_runner
        else:
            shared_state.decode_model_runner = dscheduler.tp_worker.model_runner
        shared_state.max_total_num_tokens = dscheduler.max_total_num_tokens

        dscheduler.init_attention_backend()
        # Delegate per-role forward stream binding to mixin
        if hasattr(dscheduler, "init_forward_streams") and getattr(server_args, "engine_mode", "normal") == "semipd":
            dscheduler.init_forward_streams()
        else:
            dscheduler.init_cuda_graphs(dstream)
            dscheduler.tp_worker.worker.init_forward_stream(dstream)

        shared_state.decode_ready_event.set()

        logger.info("Decode scheduler initialized. Starting event loop...")
        if dscheduler.enable_overlap:
            dscheduler.event_loop_overlap()
        else:
            dscheduler.event_loop_normal()

    @staticmethod
    def run_prefill_thread(
        scheduler_cls: Type[Any],
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        pstream: ExternalStream,
        shared_state: SemiPDSchedulerSharedState,
        pipe_writer,
    ):
        shared_state.decode_ready_event.wait()

        server_args.max_total_tokens = shared_state.max_total_num_tokens
        logger.info(
            f"Prefill scheduler using max_total_tokens: {shared_state.max_total_num_tokens}"
        )

        pscheduler = scheduler_cls(
            server_args, port_args, gpu_id, tp_rank, dp_rank, bypass_load_weight=True
        )
        logger.info("Prefill scheduler init finished.")

        if pscheduler.enable_overlap:
            target_runner = pscheduler.tp_worker.worker.model_runner
        else:
            target_runner = pscheduler.tp_worker.model_runner

        SchedulerSemiPDLauncher.share_model_and_buffers(
            source_runner=shared_state.decode_model_runner, target_runner=target_runner
        )

        pscheduler.init_attention_backend()
        # Delegate per-role forward stream binding to mixin
        if hasattr(pscheduler, "init_forward_streams") and getattr(server_args, "engine_mode", "normal") == "semipd":
            pscheduler.init_forward_streams()
        else:
            pscheduler.tp_worker.init_forward_stream(pstream)

        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": shared_state.max_total_num_tokens,
                "max_req_input_len": pscheduler.max_req_input_len,
            }
        )

        logger.info("Prefill scheduler initialized. Starting event loop...")
        pscheduler.event_loop_normal()


