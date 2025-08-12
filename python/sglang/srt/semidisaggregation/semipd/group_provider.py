from __future__ import annotations

from typing import Optional

from sglang.srt.semidisaggregation.semipd.parallel_state_semipd import (
    semipd_tp_groups_initialized,
    get_tp_group_decode_semipd,
    get_tp_group_prefill_semipd,
    _SEMIPD_TL as _semipd_tl,
)
from sglang.srt.semidisaggregation.semipd.attention_groups import (
    get_attention_tp_group_by_role,
)
from sglang.srt.distributed.parallel_state import get_tp_group as core_get_tp_group


def get_role() -> Optional[str]:
    return getattr(_semipd_tl, "role", None)


def get_tp_group_role_aware():
    role = get_role()
    # ensure groups are initialized before fetching
    try:
        semipd_tp_groups_initialized()
    except Exception:
        pass
    if role == "prefill":
        return get_tp_group_prefill_semipd()
    elif role == "decode":
        return get_tp_group_decode_semipd()
    return core_get_tp_group()


def get_attention_tp_group_role_aware(server_args):
    role = get_role()
    return get_attention_tp_group_by_role(
        role,
        tp_size=server_args.tp_size,
        dp_size=server_args.dp_size,
        pp_size=server_args.pp_size,
    )


