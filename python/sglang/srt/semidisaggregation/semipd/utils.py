import os
from dataclasses import dataclass
from enum import Enum
from typing import List

# import semi_pd_ipc
import torch
import zmq
import logging

PREFILL_ENGINE_SM_PERCENTILE = int(os.getenv("SEMI_PD_PREFILL_SM_PERCENTILE", 80))
DECODE_ENGINE_SM_PERCENTILE = int(os.getenv("SEMI_PD_DECODE_SM_PERCENTILE", 100))

SM_PREFILL_RATIO = float(os.getenv("SM_PREFILL_RATIO", 0.1))
SM_DECODE_RATIO = float(os.getenv("SM_DECODE_RATIO", 0.9))

class InstanceRole(Enum):
    PREFILL = 0
    DECODE = 1
    OTHER = 2

class AggregatedSocket:
    def __init__(self, sockets: List[zmq.Socket]):
        self.sockets = sockets

    def send_pyobj(self, obj):
        for socket in self.sockets:
            socket.send_pyobj(obj)

