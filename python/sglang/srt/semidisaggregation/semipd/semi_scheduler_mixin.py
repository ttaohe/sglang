import logging
from typing import List, Tuple

import torch
from torch.cuda.streams import ExternalStream

from .pdmux_context import (
    initialize_stream_groups,
    get_stream_groups,
    get_current_stream_idx,
)
from .utils import InstanceRole

logger = logging.getLogger(__name__)


class SchedulerSemiPDMixin:
    def init_semipd_stream_groups(self) -> None:
        if getattr(self.server_args, "engine_mode", "normal") != "semipd":
            return
        # Initialize and cache groups on the instance for later use
        if get_stream_groups() is None:
            initialize_stream_groups(self.gpu_id)
        self.stream_groups = get_stream_groups()

    def init_forward_streams(self, role: InstanceRole) -> None:
        if getattr(self.server_args, "engine_mode", "normal") != "semipd":
            return
        assert hasattr(self, "stream_groups"), "Call init_semipd_stream_groups() first"

        stream_idx = get_current_stream_idx()
        stream_group = self.stream_groups[stream_idx]

        logging.info(f"current stream group: {stream_group}")

        if role == InstanceRole.DECODE:
            decode_stream = stream_group[1]
            self.set_forward_stream(decode_stream)
            runner = (
                self.tp_worker.worker.model_runner
                if hasattr(self.tp_worker, "worker")
                else self.tp_worker.model_runner
            )
            runner.init_cuda_graphs()
        elif role == InstanceRole.PREFILL:
            prefill_stream = stream_group[0]
            self.set_forward_stream(prefill_stream)

        else:
            logger.warning(f"Unknown instance_role: {role} for SemiPD; skip per-stream init")
