import torch
from torch.cuda.streams import ExternalStream
from typing import NamedTuple, Tuple
import time

import greenctx as gtx
import logging

from .stream_switch_config import (
    STREAM_GROUP_BALANCED,
    STREAM_GROUP_DECODE_HEAVY,
    DEFAULT_STREAM_GROUP,
)

STREAM_GROUPS = []
SM_RATIOS = []
CURRENT_STREAM_IDX = 0
CURRENT_STREAM_GROUP = None

# Track last switch time for minimum interval enforcement
_LAST_SWITCH_TIME = 0

class SMAllocation(NamedTuple):
    sm_a: int
    sm_b: int
    actual_percent_a: float
    actual_percent_b: float
    streams: Tuple[ExternalStream, ExternalStream]

def create_greenctx_stream_by_percent_py(
    target_percent_a: float, target_percent_b: float, device_id: int
):
    result = gtx.create_greenctx_stream_by_percent(
        target_percent_a, target_percent_b, device_id
    )
    stream_a = ExternalStream(stream_ptr=result.streamA_ptr, device=device_id)
    stream_b = ExternalStream(stream_ptr=result.streamB_ptr, device=device_id)
    
    return (stream_a, stream_b)


def initialize_stream_groups(gpu_id: int):
    global STREAM_GROUPS, SM_RATIOS, CURRENT_STREAM_IDX, CURRENT_STREAM_GROUP
    # for pd_multiplexing, Init stream_groups
    SM_RATIOS = [
        # (prefill_sm_ratio, decode_sm_ratio)
        (0.9, 0.1),
        (0.8, 0.2),
        (0.1, 0.9),
    ]
    STREAM_GROUPS = [
        # (prefill_stream, decode_stream)
        create_greenctx_stream_by_percent_py(SM_RATIOS[0][0], SM_RATIOS[0][1], gpu_id),
        create_greenctx_stream_by_percent_py(SM_RATIOS[1][0], SM_RATIOS[1][1], gpu_id),
        create_greenctx_stream_by_percent_py(SM_RATIOS[2][0], SM_RATIOS[2][1], gpu_id),
    ]

    CURRENT_STREAM_IDX = 0
    CURRENT_STREAM_GROUP = STREAM_GROUPS[CURRENT_STREAM_IDX]

    for i,stream_pair in enumerate(STREAM_GROUPS):
        logging.info(f"Stream pair{i}: \nprefill stream:{stream_pair[0]} decode stream:{stream_pair[1]}")


def set_current_stream_idx(idx: int):
    global CURRENT_STREAM_IDX, CURRENT_STREAM_GROUP
    if idx < 0 or idx >= len(STREAM_GROUPS):
        raise ValueError(f"Invalid stream index: {idx}")
    CURRENT_STREAM_IDX = idx
    CURRENT_STREAM_GROUP = STREAM_GROUPS[CURRENT_STREAM_IDX]


def get_stream_groups() -> list[tuple[ExternalStream, ExternalStream]]:
    """Get the stream groups."""
    return STREAM_GROUPS


def get_sm_ratios() -> list[tuple[float, float]]:
    """Get the SM ratios."""
    return SM_RATIOS


def get_current_stream_idx() -> int:
    """Get the current stream index."""
    return CURRENT_STREAM_IDX


def can_switch_stream() -> bool:
    """Check if enough time has passed since last switch."""
    from .stream_switch_config import STREAM_SWITCH_MIN_INTERVAL
    current_time = time.time()
    return (current_time - _LAST_SWITCH_TIME) >= STREAM_SWITCH_MIN_INTERVAL


def switch_to_decode_heavy():
    """Switch to decode-heavy configuration."""
    global _LAST_SWITCH_TIME
    if can_switch_stream():
        set_current_stream_idx(STREAM_GROUP_DECODE_HEAVY)
        _LAST_SWITCH_TIME = time.time()
        logging.info(f"Switched to decode-heavy stream group (index {STREAM_GROUP_DECODE_HEAVY})")
        return True
    return False


def switch_to_balanced():
    """Switch to balanced configuration."""
    global _LAST_SWITCH_TIME
    if can_switch_stream():
        set_current_stream_idx(STREAM_GROUP_BALANCED)
        _LAST_SWITCH_TIME = time.time()
        logging.info(f"Switched to balanced stream group (index {STREAM_GROUP_BALANCED})")
        return True
    return False


def get_prefill_stream():
    """Get current prefill stream."""
    return STREAM_GROUPS[CURRENT_STREAM_IDX][0]


def get_decode_stream():
    """Get current decode stream."""
    return STREAM_GROUPS[CURRENT_STREAM_IDX][1]


def get_stream_group_name(idx: int) -> str:
    """Get human-readable name for stream group."""
    from .stream_switch_config import (
        STREAM_GROUP_BALANCED,
        STREAM_GROUP_DECODE_HEAVY,
        STREAM_GROUP_PREFILL_HEAVY,
    )
    
    if idx == STREAM_GROUP_PREFILL_HEAVY:
        return "prefill-heavy"
    elif idx == STREAM_GROUP_BALANCED:
        return "balanced"
    elif idx == STREAM_GROUP_DECODE_HEAVY:
        return "decode-heavy"
    else:
        return f"unknown-{idx}"
