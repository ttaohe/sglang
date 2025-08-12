"""
Semipd-specific TP group helpers.
Keep core sglang.srt.distributed.parallel_state intact; we only wrap its APIs.
"""

import threading
import torch
import logging

logger = logging.getLogger(__name__)

_SEMIPD_TP_DECODE = None
_SEMIPD_TP_PREFILL = None
_SEMIPD_TL = threading.local()



def _get_tp_groups_from_core():
    import sglang.srt.distributed.parallel_state as core_ps
    # 在部分时序下，模型并行尚未完全初始化，这里只做“尝试获取”，由首次 get_tp_group 调用触发创建
    if not core_ps.model_parallel_is_initialized():
        return core_ps, None, torch.distributed.get_backend(torch.distributed.group.WORLD), []
    world_group = core_ps.get_world_group()
    backend = torch.distributed.get_backend(world_group.device_group)
    tp_world_size = core_ps.get_tp_group().world_size
    world_size = torch.distributed.get_world_size()
    if tp_world_size == 0 or world_size % tp_world_size != 0:
        group_ranks = []
    else:
        num_tp_groups = world_size // tp_world_size
        group_ranks = [
            list(range(i * tp_world_size, (i + 1) * tp_world_size))
            for i in range(num_tp_groups)
        ]
    return core_ps, world_group, backend, group_ranks


def semipd_tp_groups_initialized() -> None:
    global _SEMIPD_TP_DECODE, _SEMIPD_TP_PREFILL
    if _SEMIPD_TP_DECODE is not None and _SEMIPD_TP_PREFILL is not None:
        return
    role = getattr(_SEMIPD_TL, "role", None)
    core_ps, world_group, backend, group_ranks = _get_tp_groups_from_core()
    if world_group is None:
        # 延迟到首次真正使用 get_tp_group 时再创建
        return
    if role == "prefill":
        if _SEMIPD_TP_PREFILL is None and group_ranks:
            _SEMIPD_TP_PREFILL = core_ps.init_model_parallel_group(
        group_ranks,
                world_group.local_rank,
        backend,
                use_message_queue_broadcaster=True,
                group_name="semipd_prefill_tp",
            )
            if _SEMIPD_TP_PREFILL.ca_comm:
                _SEMIPD_TP_PREFILL.ca_comm.disabled = True
    elif role == "decode":
        if _SEMIPD_TP_DECODE is None and group_ranks:
            _SEMIPD_TP_DECODE = core_ps.init_model_parallel_group(
            group_ranks,
                world_group.local_rank,
            backend,
                use_message_queue_broadcaster=True,
                group_name="semipd_decode_tp",
                )

def set_semipd_thread_role(role: str) -> None:
    assert role in ("decode", "prefill")
    _SEMIPD_TL.role = role


def get_tp_group_decode_semipd():
    semipd_tp_groups_initialized()
    return _SEMIPD_TP_DECODE


def get_tp_group_prefill_semipd():
    semipd_tp_groups_initialized()
    return _SEMIPD_TP_PREFILL

