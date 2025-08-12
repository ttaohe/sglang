"""
Mixin class providing multiplexing scheduling logic
"""

import logging
import time

import torch
import torch.distributed as dist
from torch.cuda.streams import ExternalStream

from sglang.srt.semidisaggregatio.semipd.pdmux_context import (
    get_current_stream_idx,
    set_current_stream_idx,
)

logger = logging.getLogger(__name__)


class SchedulerMultiplexMixin:

    # TODO(jason-fxz): This is a temporary demo
    def adjust_stream_groups(self) -> tuple[int, tuple[ExternalStream, ExternalStream]]:
        if not self.running_batch.is_empty() and self.split_prefill_batch:
            decode_bs = self.running_batch.batch_size()
            stream_idx =  max(1, min(self.real_sm_group_num - 2, decode_bs * (self.real_sm_group_num - 2) // 36))
            set_current_stream_idx(stream_idx)
        elif not self.running_batch.is_empty():
            set_current_stream_idx(self.real_sm_group_num - 1)
        else:
            set_current_stream_idx(0)

        stream_idx = get_current_stream_idx()

        self.tp_worker.model_runner.update_decode_attn_backend(stream_idx)
        return stream_idx, self.stream_groups[stream_idx]

    
