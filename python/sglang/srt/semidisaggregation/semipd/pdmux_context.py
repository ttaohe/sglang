import torch
from torch.cuda.streams import ExternalStream
from typing import NamedTuple, Tuple

import greenctx as gtx
import logging

STREAM_GROUPS = []
SM_RATIOS = []
CURRENT_STREAM_IDX = 0
CURRENT_STREAM_GROUP = None

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
        (0.5, 0.5),
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

    logging.info(STREAM_GROUPS)


def set_current_stream_idx(idx: int):
    global CURRENT_STREAM_IDX, CURRENT_STREAM_GROUP
    if idx < 0 or idx >= len(STREAM_GROUPS):
        raise ValueError(f"Invalid stream index: {idx}")
    CURRENT_STREAM_IDX = idx
    CURRENT_STREAM_GROUP = STREAM_GROUPS[CURRENT_STREAM_IDX]


def get_stream_groups() -> list[tuple[ExternalStream, ExternalStream]]:
    """Get the stream groups."""
    return STREAM_GROUPS


def get_sm_ratios() -> list[tuple[int, int]]:
    """Get the SM counts."""
    return SM_RATIOS


def get_current_stream_idx() -> int:
    """Get the current stream index."""
    return CURRENT_STREAM_IDX
