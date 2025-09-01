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
        initialize_stream_groups(self.gpu_id)
        self.stream_groups: List[Tuple[ExternalStream, ExternalStream]] = get_stream_groups()

    def init_forward_streams(self, role: InstanceRole) -> None:
        if getattr(self.server_args, "engine_mode", "normal") != "semipd":
            return
        assert hasattr(self, "stream_groups"), "Call init_semipd_stream_groups() first"

        stream_idx = get_current_stream_idx()
        stream_group = self.stream_groups[stream_idx]

        logging.info(f"current stream group: {stream_group}")

        if role == InstanceRole.DECODE:
            # Decode runs with overlap on the framework-selected decode stream
            decode_stream = stream_group[1]
            decode_stream = torch.cuda.Stream()
            runner = (
                self.tp_worker.worker.model_runner
                if hasattr(self.tp_worker, "worker")
                else self.tp_worker.model_runner
            )
            # Capture graphs on the selected stream without changing core code
            runner.init_cuda_graphs(decode_stream)
        elif role == InstanceRole.PREFILL:
            pass
            # Prefill runs without overlap on the framework-selected prefill stream
            # No cuda graph capture for prefill to keep behavior consistent
        else:
            logger.warning("Unknown instance_role for SemiPD; skip per-stream init")
