from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np

from src.utils.config import resolve_path


@dataclass
class EddyDemoService:
    model_path: str = "outputs/eddy/best.pt"
    conf: float = 0.25
    iou: float = 0.45
    max_frames: int = 120
    frame_stride: int = 10
    _MODEL_CACHE: ClassVar[dict[str, Any]] = {}

    def _load_model(self) -> Any:
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError(f"未安装或无法导入 ultralytics: {e}") from e
        mp = resolve_path(self.model_path)
        if not mp.is_file():
            raise FileNotFoundError(f"未找到涡旋权重: {mp}")
        key = str(mp.resolve())
        if key not in self._MODEL_CACHE:
            self._MODEL_CACHE[key] = YOLO(str(mp))
        return self._MODEL_CACHE[key]

    def infer_video(self, *, video_path: str, task_id: str | None = None) -> dict[str, Any]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        model = self._load_model()
        timeline: list[dict[str, Any]] = []
        hit_previews: list[str] = []
        miss_previews: list[str] = []
        conf_scores: list[float] = []
        sampled = 0
        frame_idx = 0
        out_dir = resolve_path("app/data/eddy_preview")
        out_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        while sampled < self.max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_idx % max(1, int(self.frame_stride)) != 0:
                frame_idx += 1
                continue
            pred_list = model.predict(frame, conf=self.conf, iou=self.iou, verbose=False)
            pred = pred_list[0] if pred_list else None
            num_det = 0
            mean_conf = 0.0
            if pred is not None and getattr(pred, "boxes", None) is not None:
                boxes = pred.boxes
                num_det = int(len(boxes))
                if num_det > 0 and getattr(boxes, "conf", None) is not None:
                    arr = boxes.conf.detach().cpu().numpy().astype(float)
                    mean_conf = float(np.mean(arr))
                    conf_scores.extend(arr.tolist())
            if pred is not None and hasattr(pred, "plot"):
                plotted = pred.plot()
            else:
                plotted = frame
            preview_path = out_dir / f"{task_id or 'eddy'}_{sampled:02d}.jpg"
            cv2.imwrite(str(preview_path), plotted)
            if num_det > 0:
                hit_previews.append(str(preview_path))
            else:
                miss_previews.append(str(preview_path))

            sec = frame_idx / fps if fps > 0 else float(sampled)
            timeline.append(
                {
                    "time": f"T+{sec:.1f}s",
                    "event": "检测到涡旋" if num_det > 0 else "未检测到明显涡旋",
                    "score": float(mean_conf),
                    "count": num_det,
                }
            )
            sampled += 1
            frame_idx += 1
        cap.release()
        peak = max(conf_scores) if conf_scores else 0.0
        previews = hit_previews + miss_previews
        detection_rate = float(len(hit_previews) / max(1, sampled))
        domain_warning = ""
        if detection_rate < 0.1:
            domain_warning = "提示：视频与训练域差异较大，当前结果可信度低，建议改用命题方同口径可视化视频演示。"
        return {
            "task_id": task_id,
            "module": "eddy",
            "mode": "real",
            "status": "success",
            "summary": f"已按每{int(self.frame_stride)}帧检测完成真实推理，优先展示命中涡旋的关键帧。",
            "timeline": timeline,
            "peak_score": float(peak),
            "preview_images": previews,
            "generated_at": int(time.time()),
            "elapsed_sec": float(time.time() - started),
            "meta": {"total_frames": total_frames, "fps": fps, "sampled_frames": sampled, "detection_rate": detection_rate},
            "warnings": [domain_warning] if domain_warning else [],
        }

    def infer_frame(self, *, frame: np.ndarray, task_id: str | None = None) -> dict[str, Any]:
        model = self._load_model()
        pred_list = model.predict(frame, conf=self.conf, iou=self.iou, verbose=False)
        pred = pred_list[0] if pred_list else None
        num_det = 0
        mean_conf = 0.0
        if pred is not None and getattr(pred, "boxes", None) is not None:
            boxes = pred.boxes
            num_det = int(len(boxes))
            if num_det > 0 and getattr(boxes, "conf", None) is not None:
                arr = boxes.conf.detach().cpu().numpy().astype(float)
                mean_conf = float(np.mean(arr))
        annotated = None
        if pred is not None and hasattr(pred, "plot"):
            annotated = pred.plot()
        return {
            "task_id": task_id,
            "module": "eddy",
            "mode": "real",
            "source_type": "camera",
            "status": "success",
            "summary": "实时帧涡旋检测完成。",
            "timeline": [{"time": "now", "event": "检测到涡旋" if num_det > 0 else "未检测到明显涡旋", "score": mean_conf, "count": num_det}],
            "peak_score": float(mean_conf),
            "generated_at": int(time.time()),
            "annotated_frame_bgr": annotated,
        }
