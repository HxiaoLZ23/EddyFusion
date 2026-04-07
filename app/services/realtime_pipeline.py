from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from services.inference_service import InferenceInput, InferenceService


@dataclass
class FramePacket:
    frame: np.ndarray
    timestamp: float
    source_type: str
    metadata: dict[str, Any]


class RealtimePipeline:
    """轻量队列 + 限频调度（显示与推理解耦）。"""

    def __init__(self, *, queue_maxlen: int = 8, infer_fps: float = 1.0) -> None:
        self.queue: deque[FramePacket] = deque(maxlen=max(1, queue_maxlen))
        self.infer_fps = max(0.1, float(infer_fps))
        self.last_infer_ts = 0.0

    def update_infer_fps(self, infer_fps: float) -> None:
        self.infer_fps = max(0.1, float(infer_fps))

    def enqueue(self, packet: FramePacket) -> None:
        self.queue.append(packet)

    def queue_size(self) -> int:
        return len(self.queue)

    def maybe_infer(self, service: InferenceService, *, task_id: str | None = None) -> dict[str, Any] | None:
        now = time.time()
        if now - self.last_infer_ts < (1.0 / self.infer_fps):
            return None
        if not self.queue:
            return None
        latest = self.queue[-1]
        self.queue.clear()
        input_data = InferenceInput(
            source_type=latest.source_type,
            task_id=task_id,
            frame=latest.frame,
            timestamp_ms=int(latest.timestamp * 1000),
            metadata=latest.metadata,
        )
        result = service.run(input_data)
        self.last_infer_ts = now
        return result

