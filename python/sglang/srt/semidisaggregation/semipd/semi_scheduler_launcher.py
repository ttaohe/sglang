import logging
import threading as td
from dataclasses import dataclass, field
from queue import Queue
from typing import Any, Optional, Type

from sglang.srt.semidisaggregation.semipd.utils import InstanceRole
from torch.cuda.streams import ExternalStream

from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.semidisaggregation.semipd.pdmux_context import (
    create_greenctx_stream_by_percent_py,
    initialize_stream_groups,
    get_stream_groups,
)
import threading
logger = logging.getLogger(__name__)

instance_role = threading.local()

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
        moe_ep_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        shared_state: SemiPDSchedulerSharedState,
    ):

        # In semipd, set role and bind TP routing BEFORE constructing scheduler/model_runner
        if getattr(server_args, "engine_mode", "normal") == "semipd":
            instance_role.role = InstanceRole.DECODE

        port_args.nccl_port = port_args.d_nccl_port
        port_args.scheduler_input_ipc_name = port_args.d_scheduler_input_ipc_name
        logging.info(f"decode port_args: {port_args}")
        logging.info(f"scheduler_input_ipc_name: {port_args.scheduler_input_ipc_name}")
        dscheduler = scheduler_cls(
            server_args, port_args, gpu_id, tp_rank, moe_ep_rank, pp_rank, dp_rank, None
        )
        logger.info("Decode scheduler init finished.")
        # Attach shared state for inter-thread queues
        setattr(dscheduler, "shared_state", shared_state)
        # Rebind mailboxes now that shared_state is attached
        if hasattr(dscheduler, "rebind_ipc_channels_decode"):
            try:
                dscheduler.rebind_ipc_channels_decode()
            except Exception:
                logger.exception("Failed to rebind decode IPC channels")

        if dscheduler.enable_overlap:
            shared_state.decode_model_runner = dscheduler.tp_worker.worker.model_runner
        else:
            shared_state.decode_model_runner = dscheduler.tp_worker.model_runner
        # register decode runner for later prefill adoption during its init
        try:
            from .runner import register_decode_runner
            register_decode_runner(shared_state.decode_model_runner)
            logging.info("register_decode_runner success")
        except Exception:
            pass
        shared_state.max_total_num_tokens = dscheduler.max_total_num_tokens

        # Delegate per-role forward stream binding to mixin
        dscheduler.init_forward_streams(instance_role.role)
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
        moe_ep_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        shared_state: SemiPDSchedulerSharedState,
        pipe_writer,
    ):
        shared_state.decode_ready_event.wait()

        server_args.max_total_tokens = shared_state.max_total_num_tokens
        logger.info(
            f"Prefill scheduler using max_total_tokens: {shared_state.max_total_num_tokens}"
        )

        instance_role.role = InstanceRole.PREFILL

        # 告诉下游不要重复加载权重
        server_args.skip_model_weight_loading = True
        port_args.nccl_port = port_args.p_nccl_port
        port_args.scheduler_input_ipc_name = port_args.p_scheduler_input_ipc_name
        # 关闭 prefill 的 overlap，确保选择非 overlap 的 TpModelWorker
        server_args.disable_overlap_schedule = True
        pscheduler = scheduler_cls(server_args, port_args, gpu_id, tp_rank, moe_ep_rank, pp_rank, dp_rank, None)
        logger.info("Prefill scheduler init finished.")
        # Attach shared state for inter-thread queues
        setattr(pscheduler, "shared_state", shared_state)
        # Rebind mailboxes now that shared_state is attached
        if hasattr(pscheduler, "rebind_ipc_channels_prefill"):
            try:
                pscheduler.rebind_ipc_channels_prefill()
            except Exception:
                logger.exception("Failed to rebind prefill IPC channels")

        # Delegate per-role forward stream binding to mixin
        pscheduler.init_forward_streams(instance_role.role)

        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": shared_state.max_total_num_tokens,
                "max_req_input_len": pscheduler.max_req_input_len,
            }
        )

        logger.info("Prefill scheduler initialized. Starting event loop...")
        pscheduler.event_loop_normal()


