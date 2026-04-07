from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

@dataclass
class InferenceInput:
    source_type: str  # upload | camera | rtsp
    task_id: str | None = None
    video_path: str | None = None
    frame: np.ndarray | None = None
    timestamp_ms: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class InferenceService:
    mode: str = "mock"

    def run(self, input_data: InferenceInput) -> dict[str, Any]:
        if self.mode == "real":
            return self._run_real(input_data)
        return self._run_mock(input_data)

    def _run_mock(self, input_data: InferenceInput) -> dict[str, Any]:
        seed_key = input_data.video_path or f"{input_data.source_type}:{input_data.timestamp_ms}"
        if input_data.frame is not None:
            seed_key = f"{seed_key}:{int(float(np.mean(input_data.frame)) * 1000)}"
        random.seed(hash(seed_key) % 100000)
        base = [0.12, 0.21, 0.38, 0.65, 0.44, 0.31]
        scores = [min(0.99, max(0.01, s + random.uniform(-0.06, 0.06))) for s in base]
        if input_data.frame is not None:
            # 摄像头模式用帧亮度抖动构造实时演示分数
            brightness = float(np.mean(input_data.frame)) / 255.0
            scores = [min(0.99, max(0.01, s * 0.5 + brightness * 0.7 + random.uniform(-0.08, 0.08))) for s in scores]
        timeline = []
        for i, s in enumerate(scores):
            event = "风浪异常增强" if s >= 0.5 else "正常波动"
            timeline.append({"time": f"T+{i*10}s", "event": event, "score": float(s)})
        return {
            "task_id": input_data.task_id,
            "mode": "mock",
            "source_type": input_data.source_type,
            "video_path": input_data.video_path,
            "generated_at": int(time.time()),
            "summary": "检测到中后段异常增强趋势，建议人工复核关键时段。",
            "timeline": timeline,
            "peak_score": float(max(scores)),
            "status": "success",
        }

    def _run_real(self, input_data: InferenceInput) -> dict[str, Any]:
        return {
            "task_id": input_data.task_id,
            "mode": "real",
            "source_type": input_data.source_type,
            "video_path": input_data.video_path,
            "status": "not_implemented",
            "message": "真实推理适配层已预留，后续将对接 src.anomaly.detect 与 report。",
            "timeline": [],
        }


def build_inference_service(mode: str = "mock") -> InferenceService:
    return InferenceService(mode=mode)

