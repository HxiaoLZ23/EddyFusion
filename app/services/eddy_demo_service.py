from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np

from src.eddy.frequency_enhance import enhance_bgr_frequency
from src.eddy.geometry import geometries_from_ultralytics_result
from src.eddy.multichannel_fuse import load_fused_bgr_from_npz
from src.eddy.multiscale_tta import tta_any_detection
from src.utils.config import resolve_path


@dataclass
class EddyDemoService:
    model_path: str = "outputs/eddy/best.pt"
    conf: float = 0.25
    iou: float = 0.45
    max_frames: int = 120
    frame_stride: int = 10
    base_imgsz: int = 640
    #: 频域增强：none | unsharp | laplacian（推理前预处理，用于可视化/消融）
    frequency_mode: str = "none"
    frequency_amount: float = 0.7
    #: 任一分辨率出现检测即记为 TTA 阳性（主结果仍以 base 尺度 predict 出图与几何）
    multiscale_tta: bool = False
    multiscale_scales: tuple[float, ...] = (0.85, 1.0, 1.15)
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

    def _preprocess_bgr(self, frame: np.ndarray) -> np.ndarray:
        if self.frequency_mode in (None, "", "none"):
            return frame
        return enhance_bgr_frequency(
            frame,
            mode=self.frequency_mode,  # type: ignore[arg-type]
            amount=float(self.frequency_amount),
        )

    def infer_multichannel_npz(self, *, npz_path: str, task_id: str | None = None) -> dict[str, Any]:
        """将含 sla / 涡度 / 温度梯度的 NPZ 融合为 BGR 后做单帧检测（模块A 多通道演示）。"""
        p = resolve_path(npz_path)
        if not p.is_file():
            raise FileNotFoundError(f"NPZ 不存在: {p}")
        bgr = load_fused_bgr_from_npz(p)
        return self._infer_bgr(
            bgr,
            task_id=task_id,
            source_type="multichannel_npz",
            extra_meta={"npz_path": str(p), "fuse_order_bgr": ["SLA", "vorticity", "temp_grad"]},
        )

    def _infer_bgr(
        self,
        frame_bgr: np.ndarray,
        *,
        task_id: str | None,
        source_type: str,
        extra_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        model = self._load_model()
        proc = self._preprocess_bgr(frame_bgr)
        tta_hit = False
        if self.multiscale_tta:
            tta_hit = tta_any_detection(
                model,
                proc,
                base_imgsz=int(self.base_imgsz),
                scales=tuple(self.multiscale_scales),
                conf=float(self.conf),
                iou=float(self.iou),
            )
        pred_list = model.predict(
            proc,
            conf=float(self.conf),
            iou=float(self.iou),
            imgsz=int(self.base_imgsz),
            verbose=False,
        )
        pred = pred_list[0] if pred_list else None
        H, W = proc.shape[:2]
        num_det = 0
        mean_conf = 0.0
        if pred is not None and getattr(pred, "boxes", None) is not None:
            boxes = pred.boxes
            num_det = int(len(boxes))
            if num_det > 0 and getattr(boxes, "conf", None) is not None:
                arr = boxes.conf.detach().cpu().numpy().astype(float)
                mean_conf = float(np.mean(arr))
        annotated = pred.plot() if pred is not None and hasattr(pred, "plot") else proc.copy()
        geometries = geometries_from_ultralytics_result(pred, (H, W)) if pred is not None else []
        out_meta = {
            "frequency_mode": self.frequency_mode,
            "multiscale_tta": self.multiscale_tta,
            "tta_any_hit": tta_hit,
            "base_imgsz": int(self.base_imgsz),
        }
        if extra_meta:
            out_meta.update(extra_meta)
        return {
            "task_id": task_id,
            "module": "eddy",
            "mode": "real",
            "source_type": source_type,
            "status": "success",
            "summary": "单帧/融合场涡旋检测完成。",
            "timeline": [
                {
                    "time": "now",
                    "event": "检测到涡旋" if num_det > 0 else "未检测到明显涡旋",
                    "score": mean_conf,
                    "count": num_det,
                    "tta_any_hit": tta_hit,
                }
            ],
            "peak_score": float(mean_conf),
            "geometries": geometries,
            "generated_at": int(time.time()),
            "annotated_frame_bgr": annotated,
            "meta": out_meta,
        }

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
        all_geometries: list[dict[str, Any]] = []
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
            proc = self._preprocess_bgr(frame)
            tta_hit = False
            if self.multiscale_tta:
                tta_hit = tta_any_detection(
                    model,
                    proc,
                    base_imgsz=int(self.base_imgsz),
                    scales=tuple(self.multiscale_scales),
                    conf=float(self.conf),
                    iou=float(self.iou),
                )
            pred_list = model.predict(
                proc,
                conf=float(self.conf),
                iou=float(self.iou),
                imgsz=int(self.base_imgsz),
                verbose=False,
            )
            pred = pred_list[0] if pred_list else None
            H, W = proc.shape[:2]
            num_det = 0
            mean_conf = 0.0
            if pred is not None and getattr(pred, "boxes", None) is not None:
                boxes = pred.boxes
                num_det = int(len(boxes))
                if num_det > 0 and getattr(boxes, "conf", None) is not None:
                    arr = boxes.conf.detach().cpu().numpy().astype(float)
                    mean_conf = float(np.mean(arr))
                    conf_scores.extend(arr.tolist())
            geoms = geometries_from_ultralytics_result(pred, (H, W)) if pred is not None else []
            for g in geoms:
                g["frame_sample"] = sampled
            all_geometries.extend(geoms)

            if pred is not None and hasattr(pred, "plot"):
                plotted = pred.plot()
            else:
                plotted = proc
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
                    "tta_any_hit": bool(tta_hit),
                    "instances": len(geoms),
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
            "summary": f"已按每{int(self.frame_stride)}帧检测完成真实推理；已输出实例几何与可选频域增强/多尺度灵敏度。",
            "timeline": timeline,
            "peak_score": float(peak),
            "preview_images": previews,
            "geometries": all_geometries,
            "generated_at": int(time.time()),
            "elapsed_sec": float(time.time() - started),
            "meta": {
                "total_frames": total_frames,
                "fps": fps,
                "sampled_frames": sampled,
                "detection_rate": detection_rate,
                "frequency_mode": self.frequency_mode,
                "multiscale_tta": self.multiscale_tta,
                "physics_fusion": "video 为 RGB；多通道 SLA/涡度/梯度请用 NPZ 单帧入口",
            },
            "warnings": [domain_warning] if domain_warning else [],
        }

    def infer_frame(self, *, frame: np.ndarray, task_id: str | None = None) -> dict[str, Any]:
        return self._infer_bgr(frame, task_id=task_id, source_type="camera")

    def export_geometries_json(self, geometries: list[dict[str, Any]], path: str | Path) -> Path:
        outp = resolve_path(path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(geometries, ensure_ascii=False, indent=2), encoding="utf-8")
        return outp
