import faulthandler
import logging
import os
import signal
import time
from http import HTTPStatus

from types import SimpleNamespace
from typing import Optional, Union, List, Any, NamedTuple, Tuple

import zmq
import numpy as np
import psutil
import setproctitle
import torch
import torch.distributed as dist


from dataclasses import dataclass, field

import threading as td
import multiprocessing as mp
from torch.cuda.streams import ExternalStream
from sglang.srt.constrained.base_grammar_backend import (
    INVALID_GRAMMAR_OBJ,
)
from sglang.srt.semidisaggregation.semipd.utils import InstanceRole
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.managers.schedule_batch import FINISH_ABORT, MultimodalInputs, Req, global_server_args_dict
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import validate_input_length, DPBalanceMeta
from sglang.srt.server_args import PortArgs, SemiPDPortArgs, ServerArgs
from sglang.srt.utils import (
    configure_logger,
    get_bool_env_var,
    set_gpu_proc_affinity,
    suppress_other_loggers,
    broadcast_pyobj, 
    get_zmq_socket,
    is_cpu
)
from sglang.utils import get_exception_traceback

from sglang.srt.managers.io_struct import (
    BatchProcessPrefillResultReq,
    FlushCacheReq,
    GetInternalStateReq,
    GetNextPrefillBatchInput,
    GetNextPrefillBatchOutput,
    TokenizedGenerateReqInput,
    GetInternalStateReqOutput
)
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import EmbeddingBatchResult, GenerationBatchResult

from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.managers.scheduler_metrics_mixin import (
    RECORD_STEP_TIME,
)
from sglang.srt.semidisaggregation.semipd.semi_scheduler_launcher import (
    SchedulerSemiPDLauncher,
    SemiPDSchedulerSharedState,
)
from sglang.srt.semidisaggregation.semipd.semi_scheduler_mixin import SchedulerSemiPDMixin

# Test retract decode for debugging purposes
TEST_RETRACT = get_bool_env_var("SGLANG_TEST_RETRACT")

_is_cpu = is_cpu()

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

class SemiPDScheduler(Scheduler, SchedulerSemiPDMixin):
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        dp_balance_meta: Optional[DPBalanceMeta] = None,
    ):
        super().__init__(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            moe_ep_rank,
            pp_rank,
            dp_rank,
            dp_balance_meta,
        )
        # initialize semipd stream groups on construction for derived classes
        self.init_semipd_stream_groups()

    def add_to_waiting_queue(self, req: Req):
        if req.is_retracted:
            self.waiting_queue.insert(0, req)
        else:
            self.waiting_queue.append(req)
        logging.debug(f"add to waiting queue: {req.rid}")

    def handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
    ):
        """
        SemiPD changes:
          - disable grammar
          - handle retracted requests
        """
        logger.debug(f"New request {recv_req.rid}, #tokens: {len(recv_req.input_ids)}")
            # Create a new request
        if (
            recv_req.session_params is None
            or recv_req.session_params.id is None
            or recv_req.session_params.id not in self.sessions
        ):
            if recv_req.input_embeds is not None:
                # Generate fake input_ids based on the length of input_embeds
                seq_length = len(recv_req.input_embeds)
                fake_input_ids = [1] * seq_length
                recv_req.input_ids = fake_input_ids

            if recv_req.bootstrap_port is None:
                # Use default bootstrap port
                recv_req.bootstrap_port = self.server_args.disaggregation_bootstrap_port

            req = Req(
                recv_req.rid,
                recv_req.input_text,
                recv_req.input_ids,
                recv_req.sampling_params,
                return_logprob=recv_req.return_logprob,
                top_logprobs_num=recv_req.top_logprobs_num,
                token_ids_logprob=recv_req.token_ids_logprob,
                stream=recv_req.stream,
                lora_id=recv_req.lora_id,
                input_embeds=recv_req.input_embeds,
                custom_logit_processor=recv_req.custom_logit_processor,
                return_hidden_states=recv_req.return_hidden_states,
                eos_token_ids=self.model_config.hf_eos_token_id,
                bootstrap_host=recv_req.bootstrap_host,
                bootstrap_port=recv_req.bootstrap_port,
                bootstrap_room=recv_req.bootstrap_room,
                data_parallel_rank=recv_req.data_parallel_rank,
                vocab_size=self.model_config.vocab_size,
            )
            req.tokenizer = self.tokenizer

            if (
                recv_req.session_params is not None
                and recv_req.session_params.id is not None
            ):
                req.finished_reason = FINISH_ABORT(
                    f"Invalid request: session id {recv_req.session_params.id} does not exist"
                )
                # SemiPD
                self.add_to_waiting_queue(req)
                return
        else:
            # Create a new request from a previous session
            session = self.sessions[recv_req.session_params.id]
            req = session.create_req(recv_req, self.tokenizer)
            if isinstance(req.finished_reason, FINISH_ABORT):
                # SemiPD
                self.add_to_waiting_queue(req)
                return

        # Handle multimodal inputs
        if recv_req.mm_inputs is not None:
            image_inputs = MultimodalInputs.from_dict(recv_req.mm_inputs)
            # Expand a single image token into multiple dummy tokens for receiving image embeddings
            req.origin_input_ids = self.pad_input_ids_func(
                req.origin_input_ids, image_inputs
            )
            req.extend_image_inputs(image_inputs)

            if len(req.origin_input_ids) >= self.max_req_input_len:
                req.set_finish_with_abort(
                    error_msg=(
                        "Multimodal prompt is too long after expanding multimodal tokens. "
                        f"After expanding {len(req.origin_input_ids_unpadded)=} => {len(req.origin_input_ids)} >= {self.max_req_input_len}."
                    )
                )
                # SemiPD
                self.add_to_waiting_queue(req)
                return

        # Validate prompts length
        error_msg = validate_input_length(
            req,
            self.max_req_input_len,
            self.server_args.allow_auto_truncate,
        )
        if error_msg:
            req.origin_input_ids = [0]
            req.sampling_params.max_new_tokens = 0
            # SemiPD
            self.add_to_waiting_queue(req)
            return

        # Copy more attributes
        if recv_req.logprob_start_len == -1:
            # By default, only return the logprobs for output tokens
            req.logprob_start_len = len(req.origin_input_ids) - 1
        else:
            req.logprob_start_len = recv_req.logprob_start_len

        req.sampling_params.max_new_tokens = min(
            (
                req.sampling_params.max_new_tokens
                if req.sampling_params.max_new_tokens is not None
                else 1 << 30
            ),
            self.max_req_len - len(req.origin_input_ids) - 1,
        )

        # Init grammar cache for this request
        add_to_grammar_queue = False
        if (
            req.sampling_params.json_schema is not None
            or req.sampling_params.regex is not None
            or req.sampling_params.ebnf is not None
            or req.sampling_params.structural_tag is not None
        ):
            assert self.grammar_backend is not None
            if req.sampling_params.json_schema is not None:
                key = ("json", req.sampling_params.json_schema)
            elif req.sampling_params.regex is not None:
                key = ("regex", req.sampling_params.regex)
            elif req.sampling_params.ebnf is not None:
                key = ("ebnf", req.sampling_params.ebnf)
            elif req.sampling_params.structural_tag:
                key = ("structural_tag", req.sampling_params.structural_tag)

            value, cache_hit = self.grammar_backend.get_cached_or_future_value(key)
            req.grammar = value

            if not cache_hit:
                req.grammar_key = key
                add_to_grammar_queue = True
            else:
                if value is INVALID_GRAMMAR_OBJ:  # We hit a cached invalid grammar.
                    error_msg = f"Invalid grammar request with cache hit: {key=}"
                    req.set_finish_with_abort(error_msg)

        if add_to_grammar_queue:
            # SemiPD
            raise NotImplementedError("Grammar is not supported in SemiPD mode")
        else:
            # SemiPD
            self.add_to_waiting_queue(req)


class SemiPDPrefillScheduler(SemiPDScheduler):

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        dp_balance_meta: Optional[DPBalanceMeta] = None,
    ):
        super().__init__(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            moe_ep_rank,
            pp_rank,
            dp_rank,
            dp_balance_meta,
        )

        self.enable_overlap = False
        self.chunked_rid = None
        
        if self.attn_tp_rank == 0:
            context = zmq.Context(2)
            self.send_to_d_instance = get_zmq_socket(
                context, zmq.PUSH, port_args.d_scheduler_input_ipc_name, False
            )
            self.bridge_socket = get_zmq_socket(
                context, zmq.PULL, port_args.bridge_ipc_name, True
            )
        else:
            self.send_to_d_instance = SimpleNamespace(send_pyobj=lambda x: None)
            self.bridge_socket = SimpleNamespace(recv_pyobj=lambda: None)
        
    
    
    def get_internal_state(self, recv_req: GetInternalStateReq):
        ret = dict(global_server_args_dict)
        ret["last_gen_throughput"] = self.last_gen_throughput
        ret["memory_usage"] = {
            "weight": round(
                self.tp_worker.worker.model_runner.weight_load_mem_usage, 2
            ),
            "kvcache": round(
                self.token_to_kv_pool_allocator.get_kvcache().mem_usage, 2
            ),
            "token_capacity": int(self.max_total_num_tokens),
        }

        if not _is_cpu:
            # TODO: Prefill scheduler don't have cuda graph attr
            if hasattr(self.tp_worker.worker.model_runner, "cuda_graph_mem_usage"):
                ret["memory_usage"]["cuda_graph"] = round(
                    self.tp_worker.worker.model_runner.cuda_graph_mem_usage, 2
                )
            else:
                ret["memory_usage"]["cuda_graph"] = 0

        if not self.spec_algorithm.is_none() and self.cum_spec_accept_count > 0:
            ret["avg_spec_accept_length"] = (
                self.cum_spec_accept_length / self.cum_spec_accept_count
            )
        if RECORD_STEP_TIME:
            ret["step_time_dict"] = self.step_time_dict

        ret["load"] = self.get_load()

        return GetInternalStateReqOutput(internal_state=ret)


    def to_extend_batch(self, resp: GetNextPrefillBatchOutput):
        can_run_list = [r for r in self.waiting_queue if r.rid in resp.rids]
        # Sort by the order of resp.rids
        can_run_list.sort(key=lambda r: resp.rids.index(r.rid))

        if self.chunked_rid != resp.chunked_rid:
            # Last chunked req has finished prefilling, remove it from waiting queue
            new_waiting_queue = []
            for r in self.waiting_queue:
                if r.rid == self.chunked_rid:
                    continue
                if r.rid in resp.rids and r.rid != resp.chunked_rid:
                    continue
                new_waiting_queue.append(r)
            self.waiting_queue = new_waiting_queue
            self.chunked_rid = resp.chunked_rid
        else:
            self.waiting_queue = [
                r
                for r in self.waiting_queue
                if r.rid not in resp.rids or r.rid == resp.chunked_rid
            ]

        for i, r in enumerate(can_run_list):
            assert r.rid == resp.rids[i]
            r.extend_input_len = resp.extend_input_lens[i]
            req_pool_idx = resp.req_pool_indices[i]
            pre_len = resp.prefix_lens[i]
            r.prefix_indices = self.req_to_token_pool.req_to_token[
                req_pool_idx, :pre_len
            ]
            r.fill_ids = r.origin_input_ids[: pre_len + r.extend_input_len]

        batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            self.server_args.enable_custom_logit_processor,
        )
        batch.prepare_for_extend(pre_allocated_req_pool_indices=resp.req_pool_indices) # 这里extend prefill batch使用的是decode给prefill 组的batch，所以decode应该也为prefill分配好了显存indices，不需要prefill自己再去创建
        return batch

    def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
        resp = None
        if self.waiting_queue and self.attn_tp_rank == 0:
            n_prefill_tokens = 0
            candidates = []
            for r in self.waiting_queue:
                if n_prefill_tokens > self.server_args.chunked_prefill_size:
                    break
                n_prefill_tokens += len(r.origin_input_ids)
                candidates.append(r.rid)

            req = GetNextPrefillBatchInput(rids=candidates)
            logger.debug(f"Send request to D worker: {req}")
            self.send_to_d_instance.send_pyobj(req)
            resp = self.bridge_socket.recv_pyobj()
            logger.debug(f"Recv response from D worker: {resp}")
            assert isinstance(
                resp, GetNextPrefillBatchOutput
            ), f"Expected GetNextPrefillBatchOutput, but got {type(resp)}"
        if self.attn_tp_size > 1:
            attn_tp_rank_0 = self.dp_rank * self.attn_tp_size
            resp = broadcast_pyobj(
                [resp],
                self.attn_tp_rank,
                self.attn_tp_cpu_group,
                src=attn_tp_rank_0,
            )[0]

        ret = None
        if resp and len(resp.rids) > 0:
            ret = self.to_extend_batch(resp)

        # Handle DP attention
        if self.server_args.enable_dp_attention:
            ret, _ = self.prepare_dp_attn_batch(ret)

        return ret

    def process_batch_result_prefill(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
        launch_done: Optional[td.Event] = None,
    ):

        next_token_logits = None
        if result.logits_output is not None:
            logging.debug("deal result prefill start")
            next_token_logits = result.logits_output.next_token_logits.cpu().numpy()
            logging.debug("deal result prefill finished")

        req = BatchProcessPrefillResultReq(
            next_token_ids=result.next_token_ids.tolist(),
            next_token_logits=next_token_logits,
            pp_hidden_states_proxy_tensors=result.pp_hidden_states_proxy_tensors,
            can_run_cuda_graph=result.can_run_cuda_graph,
        )

        self.send_to_d_instance.send_pyobj(req)

    def flush_cache_wrapped(self, recv_req: FlushCacheReq):
        logger.info("Ignore flush cache request")
    



class SemiPDDecodeScheduler(SemiPDScheduler):
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        dp_balance_meta: Optional[DPBalanceMeta] = None,
    ):
        super().__init__(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            moe_ep_rank,
            pp_rank,
            dp_rank,
            dp_balance_meta,
        )

        self._request_dispatcher._mapping.extend(
            [
                (GetNextPrefillBatchInput, self.get_next_prefill_batch),
                (BatchProcessPrefillResultReq, self.process_prefill_result),
            ]
        )

        # For requests that has been sent to the prefill scheduler but not yet finished.
        self.scheduled_prefill_batches: List[ScheduleBatch] = []

        if self.attn_tp_rank == 0:
            context = zmq.Context(2)

            assert isinstance(port_args, SemiPDPortArgs)
            self.bridge_socket = get_zmq_socket(
                context, zmq.PUSH, port_args.bridge_ipc_name, False
            )
            self.send_to_p_instance = get_zmq_socket(
                context, zmq.PUSH, port_args.p_scheduler_input_ipc_name, False
            )
        else:
            self.bridge_socket = SimpleNamespace(send_pyobj=lambda x: None)
            self.send_to_p_instance = SimpleNamespace(send_pyobj=lambda x: None)

    def update_running_batch(self, batch: ScheduleBatch) -> Optional[ScheduleBatch]:
        """
        Semi-PD changes:
          - add the retracted requests to the prefill scheduler
        """
        initial_bs = batch.batch_size()

        batch.filter_batch()
        if batch.is_empty():
            batch.batch_is_full = False
            return batch

        # Check if decode out of memory
        if not batch.check_decode_mem(self.decode_mem_cache_buf_multiplier) or (
            TEST_RETRACT and batch.batch_size() > 10
        ):
            old_ratio = self.new_token_ratio

            retracted_reqs, new_token_ratio = batch.retract_decode(self.server_args)
            self.new_token_ratio = new_token_ratio

            logger.info(
                "Decode out of memory happened. "
                f"#retracted_reqs: {len(retracted_reqs)}, "
                f"#new_token_ratio: {old_ratio:.4f} -> {self.new_token_ratio:.4f}"
            )

            # Semi-PD
            for req in retracted_reqs:
                req: Req
                message = TokenizedGenerateReqInput(
                    rid=req.rid,
                    input_text=req.origin_input_text + req.decoded_text,
                    input_ids=req.origin_input_ids + req.output_ids,
                    image_inputs=req.image_inputs,
                    sampling_params=req.sampling_params,
                    return_logprob=req.return_logprob,
                    logprob_start_len=req.extend_logprob_start_len,
                    top_logprobs_num=req.top_logprobs_num,
                    token_ids_logprob=req.token_ids_logprob,
                    stream=req.stream,
                    lora_path=req.lora_path,
                    input_embeds=req.input_embeds,
                    custom_logit_processor=req.custom_logit_processor,
                    return_hidden_states=req.return_hidden_states,
                    is_retracted=True,
                )

                self.waiting_queue.insert(0, req)
                self.send_to_p_instance.send_pyobj(message)
        else:
            self.new_token_ratio = max(
                self.new_token_ratio - self.new_token_ratio_decay,
                self.min_new_token_ratio,
            )

        if batch.batch_size() < initial_bs:
            batch.batch_is_full = False

        # Update batch tensors
        batch.prepare_for_decode()
        return batch

    def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
        if not self.running_batch.is_empty():
            self.running_batch = self.update_running_batch(self.running_batch)
            ret = self.running_batch if not self.running_batch.is_empty() else None
        else:
            ret = None

        # Handle DP attention
        if self.server_args.enable_dp_attention:
            ret, _ = self.prepare_dp_attn_batch(ret)

        return ret

    def get_new_batch_prefill(self, rids: List[str]) -> Optional[ScheduleBatch]:
        """
        Semi-PD changes:
          - keep scheduled prefill batches in scheduled_prefill_batches
          - disable mixed-style chunked prefill
          - skip requests that not in rids
        """
        # Check if the grammar is ready in the grammar queue
        if self.grammar_queue:
            self.move_ready_grammar_requests()

        # Handle the cases where prefill is not allowed
        if (
            self.running_batch.batch_is_full or len(self.waiting_queue) == 0
        ) and self.chunked_req is None:
            return None

        running_bs = len(self.running_batch.reqs)
        # Ignore the check if self.chunked_req is not None.
        # In the non-PP case, when self.chunked_req is not None, num_allocatable_reqs should always be greater than 0,
        # as the space for the chunked request has just been released.
        # In PP case, a chunked req can start in one microbatch and end in another microbatch, so the max_running_requests per microbatch should not be strict.
        # Instead, we should always allow chunked request to be added, otherwise, there will be a memory leak.
        if self.get_num_allocatable_reqs(running_bs) <= 0 and not self.chunked_req:
            self.running_batch.batch_is_full = True
            return None

        if self.enable_hierarchical_cache:
            self.tree_cache.check_hicache_events()

        # Get priority queue
        self.policy.calc_priority(self.waiting_queue)

        # Prefill policy
        adder = PrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            self.running_batch,
            self.new_token_ratio,
            self.max_prefill_tokens,
            self.chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
        )

        if self.chunked_req is not None:
            self.chunked_req.init_next_round_input()
            self.chunked_req = adder.add_chunked_req(self.chunked_req)

        if self.enable_lora:
            lora_set = set([req.lora_id for req in self.running_batch.reqs])

        # Get requests from the waiting queue to a new prefill batch
        for req in self.waiting_queue:
            # Semi-PD
            if req.rid not in rids:
                logging.debug("continue because not in rids")
                continue

            if self.enable_lora and not self.tp_worker.can_run_lora_batch(
                lora_set
                | set([req.lora_id for req in adder.can_run_list])
                | set([req.lora_id])
            ):
                self.running_batch.batch_is_full = True
                break

            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                self.running_batch.batch_is_full = True
                break

            if self.enable_hicache_storage:
                prefetch_done = self.tree_cache.check_prefetch_progress(req.rid)
                if not prefetch_done:
                    # skip staging requests that are ongoing prefetch
                    continue

            req.init_next_round_input(self.tree_cache)
            res = adder.add_one_req(req, has_chunked_req=(self.chunked_req is not None))


            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    logging.info(f"kv cache no space.")
                    if self.enable_hierarchical_cache:
                        # Set batch_is_full after making sure there are requests that can be served
                        self.running_batch.batch_is_full = len(
                            adder.can_run_list
                        ) > 0 or (
                            self.running_batch is not None
                            and not self.running_batch.is_empty()
                        )
                    else:
                        self.running_batch.batch_is_full = True
                break

        # Update waiting queue
        can_run_list: List[Req] = adder.can_run_list
        if len(can_run_list) == 0:
            return None

        self.waiting_queue = [
            x for x in self.waiting_queue if x not in set(can_run_list)
        ]

        if adder.new_chunked_req is not None:
            assert self.chunked_req is None
            self.chunked_req = adder.new_chunked_req

        if self.chunked_req:
            self.chunked_req.is_chunked += 1

        # Print stats
        if self.attn_tp_rank == 0:
            self.log_prefill_stats(adder, can_run_list, running_bs)

        # Create a new batch
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            self.server_args.enable_custom_logit_processor,
            chunked_req=self.chunked_req,
        )
        if self.enable_hierarchical_cache:
            # todo (zhiqiang): disable cuda graph execution if hicache loading triggered
            new_batch.hicache_consumer_index = (
                self.tree_cache.ready_to_load_host_cache()
            )
        new_batch.prepare_for_extend()
        # Semi-PD
        self.scheduled_prefill_batches.append(new_batch)

        # Mixed-style chunked prefill
        if (
            self.is_mixed_chunk
            and not self.running_batch.is_empty()
            and not (new_batch.return_logprob or self.running_batch.return_logprob)
        ):
            # Semi-PD
            raise NotImplementedError(
                "Mixed chunked prefill is not supported in Semi-PD mode"
            )
        else:
            new_batch.decoding_reqs = None

        return new_batch

    def get_next_prefill_batch(self, recv_req: GetNextPrefillBatchInput):
        if self.chunked_req:
            self.tree_cache.cache_unfinished_req(self.chunked_req)
            self.req_to_token_pool.free(self.chunked_req.req_pool_idx)
        batch = self.get_new_batch_prefill(recv_req.rids)

        if batch is None:
            self.bridge_socket.send_pyobj(
                GetNextPrefillBatchOutput(
                    rids=[],
                    chunked_rid=None,
                    req_pool_indices=[],
                    prefix_lens=[],
                    extend_input_lens=[],
                )
            )
        else:
            # Serialize the essential information of the batch
            self.bridge_socket.send_pyobj(
                GetNextPrefillBatchOutput(
                    rids=[r.rid for r in batch.reqs],
                    chunked_rid=(self.chunked_req.rid if self.chunked_req else None),
                    req_pool_indices=[r.req_pool_idx for r in batch.reqs],
                    prefix_lens=[len(r.prefix_indices) for r in batch.reqs],
                    extend_input_lens=[r.extend_input_len for r in batch.reqs],
                )
            )
            rids = [r.rid for r in batch.reqs]
            logging.debug(f"sending reqs: {rids} to prefill scheduler")

    def process_prefill_result(self, recv_req: BatchProcessPrefillResultReq):
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        batch = self.scheduled_prefill_batches.pop(0)
        assert len(batch.reqs) == len(recv_req.next_token_ids)

        logits_processor_output = None
        if recv_req.next_token_logits is not None:
            logits_processor_output = LogitsProcessorOutput(
                next_token_logits=torch.from_numpy(recv_req.next_token_logits).to(
                    self.device, dtype=torch.float16, non_blocking=True
                ),
                hidden_states=None,
            )
        # TODO: return logprobs is not supported in Semi-PD mode
        result = GenerationBatchResult(
            next_token_ids=recv_req.next_token_ids,
            logits_output=logits_processor_output,
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
            bid=-1,  # doesn't matter
            pp_hidden_states_proxy_tensors=recv_req.pp_hidden_states_proxy_tensors,
            can_run_cuda_graph=recv_req.can_run_cuda_graph
        )
        if self.attn_tp_size > 1:
            dist.barrier(group=self.attn_tp_cpu_group)
        batch.output_ids = torch.from_numpy(
            np.array(result.next_token_ids, dtype=np.int64)
        ).to(self.device, dtype=torch.int64, non_blocking=True)
        self.process_batch_result_prefill(batch, result)
        logging.debug(f"recv prefill results {recv_req.next_token_ids}")
        batch.filter_batch(chunked_req_to_exclude=self.chunked_req)

        if not batch.is_empty():
            if self.running_batch.is_empty():
                self.running_batch = batch
            else:
                self.running_batch.merge_batch(batch)


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
    sm_prefill: float,
    sm_decode: float,
):
    # --- Process Configuration ---
    if dp_rank is None:
        prefix = f"TP {tp_rank}"
    else:
        prefix = f"DP{dp_rank} TP{tp_rank}"
    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        dp_rank = int(os.environ["SGLANG_DP_RANK"])

    setproctitle.setproctitle(f"sglang::semi_pd_scheduler{prefix.replace(' ', '_')}")
    faulthandler.enable()
    parent_process = psutil.Process().parent()

    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, gpu_id)

    # --- Stream and State Initialization ---
    pstream, dstream = SchedulerSemiPDLauncher.init_streams(
        sm_prefill, sm_decode, tp_rank, engine_mode=getattr(server_args, "engine_mode", "normal"), gpu_id=gpu_id
    )
    
    shared_state = SemiPDSchedulerSharedState()

    try:
        # --- Thread Creation and Startup ---
        d_thread = td.Thread(
            target=SchedulerSemiPDLauncher.run_decode_thread,
            name="decode_scheduler",
            args=(SemiPDDecodeScheduler, server_args, port_args, gpu_id, tp_rank, moe_ep_rank, pp_rank, dp_rank, shared_state),
        )
        p_thread = td.Thread(
            target=SchedulerSemiPDLauncher.run_prefill_thread,
            name="prefill_scheduler",
            args=(SemiPDPrefillScheduler, server_args, port_args, gpu_id, tp_rank, moe_ep_rank, pp_rank, dp_rank, shared_state, pipe_writer),
        )
        
        d_thread.start()
        p_thread.start()
        
        # --- Monitor and Join ---
        d_thread.join()
        p_thread.join()
        
        logger.info("Scheduler threads have completed. Exiting process.")

    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler process hit an exception: {traceback}")
        if parent_process.is_running():
            parent_process.send_signal(signal.SIGQUIT)

class SemiPDStandaloneScheduler:
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: SemiPDPortArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
    ):
        nccl_port = port_args.s_nccl_port
        self.tp_worker = TpModelWorker(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            nccl_port=nccl_port,
            bypass_load_weight=False,
            instance_role=InstanceRole.OTHER,
        )

        self.max_total_num_tokens = self.tp_worker.max_total_num_tokens

    def get_ipc_info(self):
        return self.tp_worker.get_ipc_info()

    def event_loop(self):
        while True:
            time.sleep(1)

class MemoryCachingContext:
    """
    Disable tensor reuse cache.

    This is used for avoiding memory caching in model loading, some of the model parameters
    which get relative small size, will reuse memory from cache pool. This will cause the IPC
    memory panic, so we disable the memory caching for real model loading.
    """

    def __init__(self, enable_caching: bool = True):
        self.enable_caching = enable_caching

    def __enter__(self):
        if not self.enable_caching:
            os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.enable_caching:
            del os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"]

def run_standalone_scheduler_process(
    server_args: ServerArgs,
    port_args: SemiPDPortArgs,
    gpu_id: int,
    tp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
    bypass_load_weight: bool = False,
    p_ipc_info_queue: mp.Queue = None,
    d_ipc_info_queue: mp.Queue = None,
):
    setproctitle.setproctitle("sglang::semi_pd_standalone_scheduler")
    faulthandler.enable()

    # [For Router] if env var "SGLANG_DP_RANK" exist, set dp_rank to the value of the env var
    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        dp_rank = int(os.environ["SGLANG_DP_RANK"])

    role = "Standalone"
    # Configure the logger
    if dp_rank is None:
        configure_logger(server_args, prefix=f" {role} TP{tp_rank}")
    else:
        configure_logger(server_args, prefix=f" {role} DP{dp_rank} TP{tp_rank}")
    suppress_other_loggers()

    # Set cpu affinity to this gpu process
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, gpu_id)

    # Create a scheduler and run the event loop
    try:
        with MemoryCachingContext(enable_caching=False):
            scheduler = SemiPDStandaloneScheduler(
                server_args,
                port_args,
                gpu_id,
                tp_rank,
                dp_rank,
            )
        ipc_info = scheduler.get_ipc_info()
        p_ipc_info_queue.put(ipc_info)
        d_ipc_info_queue.put(ipc_info)

        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": scheduler.max_total_num_tokens,
            }
        )

        scheduler.event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")