"""
Attention TP group helpers for Semi-PD.

This module creates two independent Attention TP groups (prefill/decode)
with identical memberships but separate ProcessGroup handles so that
collective sequence numbers do not interfere across concurrent paths.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

from sglang.srt.distributed import parallel_state as core_ps
from sglang.srt.layers.dp_attention import get_attention_tp_size

logger = logging.getLogger(__name__)

_ATTN_TP_GROUP_PREFILL = None
_ATTN_TP_GROUP_DECODE = None


def _build_group_ranks(attn_tp_size: int, tp_size: int, pp_size: int) -> List[List[int]]:
    # Align grouping strategy with dp_attention.initialize_dp_attention
    groups = [list(range(head, head + attn_tp_size)) for head in range(0, pp_size * tp_size, attn_tp_size)]
    world_size = torch.distributed.get_world_size()
    # If multiple replicas exist, reconstruct by global linear division
    if len(groups) * attn_tp_size != world_size:
        num_groups = world_size // attn_tp_size
        groups = [list(range(i * attn_tp_size, (i + 1) * attn_tp_size)) for i in range(num_groups)]
    return groups


def ensure_attention_groups(tp_size: int, dp_size: int, pp_size: int):
    global _ATTN_TP_GROUP_PREFILL, _ATTN_TP_GROUP_DECODE
    if _ATTN_TP_GROUP_PREFILL is not None and _ATTN_TP_GROUP_DECODE is not None:
        return

    attn_tp_size = tp_size // max(1, dp_size)
    # sanity check: match dp_attention's computed size
    try:
        if get_attention_tp_size() != attn_tp_size:
            logger.warning(
                "attention tp size mismatch between semipd and dp_attention: %s vs %s",
                attn_tp_size,
                get_attention_tp_size(),
            )
    except Exception:
        # dp_attention may be uninitialized in some flows; continue
        pass

    world_group = core_ps.get_world_group()
    backend = torch.distributed.get_backend(world_group.device_group)
    group_ranks = _build_group_ranks(attn_tp_size, tp_size, pp_size)

    _ATTN_TP_GROUP_PREFILL = core_ps.init_model_parallel_group(
        group_ranks,
        world_group.local_rank,
        backend,
        use_message_queue_broadcaster=True,
        group_name="attention_tp_prefill",
    )
    _ATTN_TP_GROUP_DECODE = core_ps.init_model_parallel_group(
        group_ranks,
        world_group.local_rank,
        backend,
        use_message_queue_broadcaster=True,
        group_name="attention_tp_decode",
    )


def get_attention_tp_group_by_role(role: Optional[str], tp_size: int, dp_size: int, pp_size: int):
    ensure_attention_groups(tp_size, dp_size, pp_size)
    if role == "prefill":
        return _ATTN_TP_GROUP_PREFILL
    else:
        return _ATTN_TP_GROUP_DECODE


