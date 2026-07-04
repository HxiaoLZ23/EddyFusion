from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

import cv2
import numpy as np

from src.eddy.frequency_enhance import enhance_bgr_frequency
from src.eddy.geometry import geometries_from_ultralytics_result
from src.eddy.multichannel_fuse import load_fused_bgr_from_npz
from src.eddy.nc_dual_batch import (
    build_dual_frame_from_triple,
    build_dual_frames_parallel,
    extract_triple_slices_batch,
    probe_netcdf_time_meta,
)
from src.eddy.nc_to_bgr import extract_bgr_frame_from_netcdf, extract_triple_scalar_fields_from_netcdf
from src.eddy.stacked_physics import build_physics_stacked_hw7, build_physics_stacked_hw8, relative_vorticity_and_okubo_weiss_from_uv
from src.eddy.multiscale_tta import tta_any_detection
from src.eddy.mp4_browser_safe import encode_bgr_frames_to_browser_mp4
from src.utils.config import resolve_path


def default_eddy_weight_path_for_stack(stack: Literal["3ch", "7ch"] = "3ch") -> str:
    """与 ``config/eddy.yaml``（3ch Fair-B0）、``config/eddy_enh7.yaml``（7ch）对齐的默认权重路径。"""
    if stack == "7ch":
        for rel in (
            "outputs/eddy_enh7/train/weights/best.pt",
            "outputs/eddy_enh7/best.pt",
            "AutoDL/outputs/eddy_enh7/train/weights/best.pt",
            "AutoDL/outputs/eddy_enh7/best.pt",
        ):
            if resolve_path(rel).is_file():
                return rel
        return "outputs/eddy_enh7/best.pt"
    for rel in (
        "outputs/eddy_v6_b0_fair/best.pt",
        "outputs/eddy_v6_b0_fair/last.pt",
        "AutoDL/outputs/eddy_v6_b0_fair/best.pt",
        "AutoDL/outputs/eddy_v6_b0_fair/last.pt",
        "outputs/eddy_cloud_fair/best.pt",
        "outputs/eddy_cloud_fair/last.pt",
        "outputs/eddy/best.pt",
        "AutoDL/outputs/eddy/train/weights/best.pt",
    ):
        if resolve_path(rel).is_file():
            return rel
    return "outputs/eddy_v6_b0_fair/best.pt"


def default_eddy_weight_path() -> str:
    """兼容旧调用：默认 3 通道基线权重。"""
    return default_eddy_weight_path_for_stack("3ch")


@dataclass
class EddyDemoService:
    model_path: str = "outputs/eddy_v6_b0_fair/best.pt"
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

    @staticmethod
    def _yolo_first_conv_in_channels(model: Any) -> int:
        try:
            import torch.nn as nn

            for m in model.model.modules():
                if isinstance(m, nn.Conv2d):
                    return int(m.in_channels)
        except Exception:
            pass
        return 3

    def _yolo_device(self) -> str | int:
        """Ultralytics ``predict(..., device=...)``；未显式指定时 YOLO 常落在 CPU。"""
        override = os.environ.get("EDDY_YOLO_DEVICE", "").strip()
        if override:
            if override.isdigit():
                return int(override)
            return override
        from src.utils.config import load_yaml, pick_device

        preferred = "cuda"
        for rel in ("config/eddy.yaml", "config/eddy_v6_b0_fair.yaml", "config/eddy_enh.yaml"):
            try:
                preferred = str(load_yaml(rel).get("train", {}).get("device", "cuda"))
                break
            except Exception:
                continue
        picked = pick_device(preferred)
        if picked == "cuda":
            import torch

            if torch.cuda.is_available():
                return 0
            return "cpu"
        return "cpu"

    def _yolo_predict_kwargs(self, **extra: Any) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "conf": float(self.conf),
            "iou": float(self.iou),
            "imgsz": int(self.base_imgsz),
            "verbose": False,
            "device": self._yolo_device(),
        }
        kw.update(extra)
        return kw

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
            dev = self._yolo_device()
            m = YOLO(str(mp))
            if dev != "cpu":
                try:
                    m.to(dev)
                except Exception:
                    pass
            self._MODEL_CACHE[key] = m
        return self._MODEL_CACHE[key]

    def _preprocess_bgr(self, frame: np.ndarray) -> np.ndarray:
        if self.frequency_mode in (None, "", "none"):
            return frame
        return enhance_bgr_frequency(
            frame,
            mode=self.frequency_mode,  # type: ignore[arg-type]
            amount=float(self.frequency_amount),
        )

    def _preprocess_frame_hwc(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim != 3:
            return frame
        if int(frame.shape[2]) == 3:
            return self._preprocess_bgr(frame)
        return frame

    @staticmethod
    def _draw_compact_detections_on_bgr(base_bgr: np.ndarray, pred: Any) -> np.ndarray:
        """单类涡旋：细框 + 仅置信度小字，不显示类名；分割掩码用半透明叠加，避免 Ultralytics 默认大标签挡画面。"""
        out = np.asarray(base_bgr, dtype=np.uint8).copy()
        if pred is None:
            return out
        H, W = int(out.shape[0]), int(out.shape[1])

        masks = getattr(pred, "masks", None)
        if masks is not None and getattr(masks, "data", None) is not None:
            mdata = masks.data
            try:
                md = mdata.detach().float().cpu().numpy()
            except Exception:
                md = np.asarray(mdata)
            for i in range(md.shape[0]):
                m = md[i].astype(np.float32)
                if m.shape != (H, W):
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
                binm = m > 0.5
                if not np.any(binm):
                    continue
                overlay = out.copy()
                overlay[binm] = (80, 200, 255)  # BGR 浅青，勿过亮
                out = cv2.addWeighted(out, 0.78, overlay, 0.22, 0)

        boxes = getattr(pred, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return out
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy() if getattr(boxes, "conf", None) is not None else None
        color = (0, 220, 255)  # BGR 亮青边线
        thick = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.45
        for j in range(len(xyxy)):
            x1, y1, x2, y2 = [int(round(float(t))) for t in xyxy[j]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thick, lineType=cv2.LINE_AA)
            if confs is not None:
                txt = f"{float(confs[j]):.2f}"
                ty = max(14, y1 - 3)
                cv2.putText(out, txt, (x1 + 1, ty), font, fs, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(out, txt, (x1 + 1, ty), font, fs, color, 1, cv2.LINE_AA)
        return out

    def _plot_or_fallback_bgr(
        self,
        pred: Any,
        proc: np.ndarray,
        *,
        bgr_for_plot: np.ndarray | None,
    ) -> np.ndarray:
        base = bgr_for_plot if bgr_for_plot is not None else proc[..., :3]
        if pred is None:
            return base.copy()
        return self._draw_compact_detections_on_bgr(base, pred)

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
            bgr_for_plot=None,
        )

    def infer_netcdf_frame(
        self,
        *,
        nc_path: str,
        time_index: int = 0,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """从 NetCDF 抽取单帧 ADT/流场或 SST/流场 → BGR 或 8ch 物理堆叠，再做涡旋检测。"""
        p = resolve_path(nc_path)
        if not p.is_file():
            raise FileNotFoundError(f"NC 不存在: {p}")
        model = self._load_model()
        ich = self._yolo_first_conv_in_channels(model)
        bgr_vis, nc_meta_bgr = extract_bgr_frame_from_netcdf(p, time_index=int(time_index))
        if ich >= 8:
            adt, u0, v0, nc_meta = extract_triple_scalar_fields_from_netcdf(p, time_index=int(time_index))
            zeta, ow = relative_vorticity_and_okubo_weiss_from_uv(u0, v0)
            hw8 = build_physics_stacked_hw8(adt, u0, v0, zeta, ow)
            proc8 = np.clip(np.asarray(hw8, dtype=np.float64) * 255.0, 0.0, 255.0).astype(np.uint8)
            merged_meta = {
                "nc_path": str(p),
                **{k: v for k, v in nc_meta.items() if k != "nc_path"},
                "inference_input_channels": 8,
                "inference_stack": "physics_hw8_from_nc",
            }
            return self._infer_bgr(
                proc8,
                task_id=task_id,
                source_type="netcdf",
                extra_meta=merged_meta,
                bgr_for_plot=bgr_vis,
            )
        if ich == 7:
            adt, u0, v0, nc_meta = extract_triple_scalar_fields_from_netcdf(p, time_index=int(time_index))
            zeta, ow = relative_vorticity_and_okubo_weiss_from_uv(u0, v0)
            hw7 = build_physics_stacked_hw7(adt, u0, v0, zeta, ow)
            proc7 = np.clip(np.asarray(hw7, dtype=np.float64) * 255.0, 0.0, 255.0).astype(np.uint8)
            merged_meta = {
                "nc_path": str(p),
                **{k: v for k, v in nc_meta.items() if k != "nc_path"},
                "inference_input_channels": 7,
                "inference_stack": "physics_hw7_from_nc",
            }
            return self._infer_bgr(
                proc7,
                task_id=task_id,
                source_type="netcdf",
                extra_meta=merged_meta,
                bgr_for_plot=bgr_vis,
            )
        return self._infer_bgr(
            bgr_vis,
            task_id=task_id,
            source_type="netcdf",
            extra_meta={"nc_path": str(p), **{k: v for k, v in nc_meta_bgr.items() if k != "nc_path"}},
            bgr_for_plot=None,
        )

    def _write_bgr_frames_mp4(
        self,
        frames: list[np.ndarray],
        *,
        fps: float,
        task_id: str | None,
        basename: str = "eddy_nc_clip",
    ) -> dict[str, Any]:
        """将 BGR 帧列表编码为 MP4：优先 ffmpeg H.264（浏览器可播），否则退回 OpenCV mp4v。"""
        if not frames:
            return {"status": "failed", "message": "无视频帧"}
        h0, w0 = int(frames[0].shape[0]), int(frames[0].shape[1])
        norm: list[np.ndarray] = []
        for fr in frames:
            arr = np.asarray(fr, dtype=np.uint8)
            if int(arr.shape[0]) != h0 or int(arr.shape[1]) != w0:
                arr = cv2.resize(arr, (w0, h0))
            norm.append(arr)

        out_dir = resolve_path("app/data/eddy_preview")
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = task_id or uuid.uuid4().hex[:12]
        out_mp4 = out_dir / f"{basename}_{tag}.mp4"

        ok_ff, ff_msg = encode_bgr_frames_to_browser_mp4(
            norm,
            fps=float(fps),
            out_path=out_mp4,
        )
        if ok_ff:
            enc_id = "h264_nvenc" if "nvenc" in ff_msg.lower() else "h264_ffmpeg"
            return {
                "status": "success",
                "mp4_path": str(out_mp4.resolve()),
                "n_frames": len(norm),
                "video_encoding": enc_id,
                "video_encoding_note": ff_msg,
            }

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_mp4), fourcc, float(max(0.5, fps)), (w0, h0))
        if not writer.isOpened():
            return {
                "status": "failed",
                "message": f"ffmpeg 不可用（{ff_msg}），且 OpenCV mp4v 初始化失败；请安装 ffmpeg 或检查 OpenCV。",
            }
        try:
            for fr in norm:
                writer.write(fr)
        finally:
            writer.release()

        if not out_mp4.is_file() or out_mp4.stat().st_size < 32:
            return {"status": "failed", "message": "视频写出失败或文件过小。"}
        return {
            "status": "success",
            "mp4_path": str(out_mp4.resolve()),
            "n_frames": len(norm),
            "video_encoding": "mp4v_opencv",
            "video_encoding_note": f"浏览器可能无法内嵌播放（{ff_msg}）。请安装 ffmpeg 并加入 PATH 后重试，或用 VLC 打开。",
        }

    def infer_netcdf_detection_video(
        self,
        *,
        nc_path: str,
        session_time_index: int = 0,
        fps: float = 6.0,
        max_frames: int = 120,
        single_time_repeats: int = 36,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """NetCDF 全时序（或单时次重复）推理并写出带检测框的 MP4；返回用于会话的某一时刻完整推理结果。"""
        p = resolve_path(nc_path)
        if not p.is_file():
            return {"status": "failed", "message": f"NC 不存在: {p}"}

        _b0, meta0 = extract_bgr_frame_from_netcdf(p, time_index=0)
        del _b0
        tlen_raw = meta0.get("time_len")
        T = int(tlen_raw) if tlen_raw is not None else 1
        T = max(1, T)

        session_ti = max(0, min(int(session_time_index), T - 1))
        cap = max(1, int(max_frames))
        truncated = False

        if T >= 2:
            n_take = min(T, cap)
            indices = list(range(0, n_take))
            if n_take < T:
                truncated = True
        else:
            indices = [0]

        frames: list[np.ndarray] = []
        session_raw: dict[str, Any] | None = None
        for ti in indices:
            one = self.infer_netcdf_frame(nc_path=str(p), time_index=int(ti), task_id=None)
            if one.get("status") != "success":
                return {
                    "status": "failed",
                    "message": f"time_index={ti} 推理失败: {one.get('message', one)}",
                }
            bgr = one.get("annotated_frame_bgr")
            if bgr is None:
                return {"status": "failed", "message": f"time_index={ti} 无可视化帧"}
            frames.append(np.asarray(bgr, dtype=np.uint8))
            if int(ti) == int(session_ti):
                session_raw = one

        if session_raw is None:
            session_raw = self.infer_netcdf_frame(
                nc_path=str(p), time_index=int(session_ti), task_id=None
            )
            if session_raw.get("status") != "success":
                return {
                    "status": "failed",
                    "message": f"会话参考帧 time_index={session_ti} 推理失败: {session_raw.get('message')}",
                }

        if T < 2:
            fr0 = frames[0]
            rep = max(8, int(single_time_repeats))
            frames = [fr0] * rep

        enc = self._write_bgr_frames_mp4(
            frames,
            fps=float(fps),
            task_id=task_id,
            basename="eddy_nc_auto",
        )
        if enc.get("status") != "success":
            return enc

        return {
            "status": "success",
            "mp4_path": enc["mp4_path"],
            "n_frames": enc["n_frames"],
            "time_indices": indices,
            "truncated": truncated,
            "session_frame": session_raw,
            "session_time_index_used": int(session_ti),
            "video_encoding": enc.get("video_encoding"),
            "video_encoding_note": enc.get("video_encoding_note"),
            "meta": {
                "nc_path": str(p),
                "time_len": T,
                "fps": float(fps),
                "max_frames_cap": cap,
                "single_time_repeated": T < 2,
                "video_encoding": enc.get("video_encoding"),
                "video_encoding_note": enc.get("video_encoding_note"),
            },
        }

    def infer_netcdf_clip_mp4(
        self,
        *,
        nc_path: str,
        time_start: int = 0,
        time_stop: int | None = None,
        time_stride: int = 1,
        fps: float = 6.0,
        max_frames: int = 120,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """对 NetCDF 多个 time_index 逐帧推理，合成 MP4（BGR），供 Streamlit `st.video` 播放。"""
        p = resolve_path(nc_path)
        if not p.is_file():
            return {"status": "failed", "message": f"NC 不存在: {p}"}

        _b0, meta0 = extract_bgr_frame_from_netcdf(p, time_index=0)
        del _b0
        tlen = meta0.get("time_len")
        if tlen is None or int(tlen) < 2:
            return {
                "status": "failed",
                "message": "该 NC 无可用时间维或仅单时次，无法生成视频；请使用含 time>1 的格点序列。",
            }

        T = int(tlen)
        t0 = max(0, min(int(time_start), T - 1))
        t1 = T - 1 if time_stop is None else max(0, min(int(time_stop), T - 1))
        if t1 < t0:
            t0, t1 = t1, t0

        stride = max(1, int(time_stride))
        indices = list(range(t0, t1 + 1, stride))
        cap = max(1, int(max_frames))
        truncated = False
        if len(indices) > cap:
            indices = indices[:cap]
            truncated = True

        frames: list[np.ndarray] = []
        for ti in indices:
            one = self.infer_netcdf_frame(nc_path=str(p), time_index=int(ti), task_id=None)
            if one.get("status") != "success":
                return {
                    "status": "failed",
                    "message": f"time_index={ti} 推理失败: {one.get('message', one)}",
                }
            bgr = one.get("annotated_frame_bgr")
            if bgr is None:
                return {"status": "failed", "message": f"time_index={ti} 无可视化帧"}
            frames.append(np.asarray(bgr, dtype=np.uint8))

        enc = self._write_bgr_frames_mp4(
            frames,
            fps=float(fps),
            task_id=task_id,
            basename="eddy_nc_clip",
        )
        if enc.get("status") != "success":
            return enc

        return {
            "status": "success",
            "mp4_path": enc["mp4_path"],
            "n_frames": enc["n_frames"],
            "time_indices": indices,
            "truncated": truncated,
            "video_encoding": enc.get("video_encoding"),
            "video_encoding_note": enc.get("video_encoding_note"),
            "meta": {
                "nc_path": str(p),
                "time_len": T,
                "time_start": t0,
                "time_stop": t1,
                "time_stride": stride,
                "fps": float(fps),
                "video_encoding": enc.get("video_encoding"),
                "video_encoding_note": enc.get("video_encoding_note"),
            },
        }

    @staticmethod
    def _dual_job_dir(job_id: str) -> Path:
        d = resolve_path("app/data/eddy_preview/jobs") / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _extract_nc_dual_frame(
        self,
        nc_path: Path,
        time_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
        """仅读 NC：返回 (底图 BGR, YOLO 输入, 叠加底图可选, meta)。不跑 YOLO。"""
        model = self._load_model()
        ich = self._yolo_first_conv_in_channels(model)
        stack_ch = 8 if ich >= 8 else (7 if ich == 7 else 3)
        rows = extract_triple_slices_batch(nc_path, [int(time_index)])
        if not rows:
            raise ValueError(f"无法从 NC 读取 time_index={time_index}")
        a0, u0, v0, meta = rows[0]
        return build_dual_frame_from_triple(a0, u0, v0, meta, physics_stack_channels=stack_ch)

    def _batch_annotate_yolo_frames(
        self,
        yolo_frames: list[np.ndarray],
        plot_bgrs: list[np.ndarray | None],
    ) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
        if not yolo_frames:
            return [], []
        model = self._load_model()
        procs = [self._preprocess_frame_hwc(f) for f in yolo_frames]
        batch_sz = int(os.environ.get("EDDY_DUAL_YOLO_BATCH", "4"))
        batch_sz = max(1, min(batch_sz, len(procs)))
        annotated: list[np.ndarray] = []
        timeline: list[dict[str, Any]] = []
        for i in range(0, len(procs), batch_sz):
            chunk_p = procs[i : i + batch_sz]
            chunk_plot = plot_bgrs[i : i + batch_sz]
            pred_list = model.predict(chunk_p, **self._yolo_predict_kwargs())
            for proc, pred, bfp in zip(chunk_p, pred_list, chunk_plot):
                num_det = 0
                peak = 0.0
                mean_conf = 0.0
                if pred is not None and getattr(pred, "boxes", None) is not None:
                    boxes = pred.boxes
                    num_det = int(len(boxes))
                    if num_det > 0 and getattr(boxes, "conf", None) is not None:
                        arr = boxes.conf.detach().cpu().numpy().astype(float).reshape(-1)
                        peak = float(np.max(arr))
                        mean_conf = float(np.mean(arr))
                timeline.append(
                    {
                        # peak_score = 当帧所有检测框 conf 的 max（与视频角标一致；非平均）
                        "peak_score": round(peak, 4),
                        "max_conf": round(peak, 4),
                        "mean_conf": round(mean_conf, 4),
                        "status": "hit" if num_det > 0 else "miss",
                        "count": num_det,
                    }
                )
                annotated.append(self._plot_or_fallback_bgr(pred, proc, bgr_for_plot=bfp))
        return annotated, timeline

    def _plan_dual_indices(
        self,
        *,
        nc_path: Path,
        time_start: int,
        time_stop: int | None,
        time_stride: int,
        max_frames: int,
    ) -> tuple[list[int], bool, dict[str, Any]]:
        meta0 = probe_netcdf_time_meta(nc_path)
        tlen = meta0.get("time_len")
        if tlen is None or int(tlen) < 2:
            raise ValueError("该 NC 无可用时间维或仅单时次，无法生成双路视频；请使用含 time>1 的格点序列。")
        T = int(tlen)
        t0 = max(0, min(int(time_start), T - 1))
        t1 = T - 1 if time_stop is None else max(0, min(int(time_stop), T - 1))
        if t1 < t0:
            t0, t1 = t1, t0
        stride_user = max(1, int(time_stride))
        stride = stride_user
        if stride_user == 1:
            if T > 360:
                stride = 3
            elif T > 200:
                stride = 2
        indices = list(range(t0, t1 + 1, stride))
        cap = max(1, int(max_frames))
        truncated = False
        if len(indices) > cap:
            indices = indices[:cap]
            truncated = True
        dual_infer_cap = int(os.environ.get("EDDY_DUAL_MAX_INFER_FRAMES", "120"))
        dual_infer_cap = max(8, min(dual_infer_cap, 120))
        if len(indices) > dual_infer_cap:
            pos = np.linspace(0, len(indices) - 1, num=dual_infer_cap, dtype=float)
            indices = [indices[int(round(x))] for x in pos]
            truncated = True
        plan = {
            "time_len": T,
            "time_start": t0,
            "time_stop": t1,
            "time_stride": stride,
            "time_stride_requested": stride_user,
            "dual_infer_cap": dual_infer_cap,
            "n_infer_frames": len(indices),
        }
        return indices, truncated, plan

    def _cache_dual_frames(self, nc_path: Path, indices: list[int], job_id: str) -> list[str]:
        job_dir = self._dual_job_dir(job_id)
        model = self._load_model()
        ich = self._yolo_first_conv_in_channels(model)
        stack_ch = 8 if ich >= 8 else (7 if ich == 7 else 3)
        slices = extract_triple_slices_batch(nc_path, [int(ti) for ti in indices])
        if len(slices) != len(indices):
            raise RuntimeError(f"批量抽帧数量不一致: 期望 {len(indices)}，实际 {len(slices)}")
        built = build_dual_frames_parallel(slices, physics_stack_channels=stack_ch)
        labels: list[str] = []
        plot_paths: list[str | None] = []
        for k, (base, yolo_in, plot_bgr, meta) in enumerate(built):
            np.save(job_dir / f"base_{k:04d}.npy", base)
            np.save(job_dir / f"yolo_{k:04d}.npy", yolo_in)
            if plot_bgr is not None:
                pp = job_dir / f"plot_{k:04d}.npy"
                np.save(pp, plot_bgr)
                plot_paths.append(str(pp.name))
            else:
                plot_paths.append(None)
            labels.append(str(meta.get("time_label", f"步 {indices[k]}")))
        manifest = {
            "nc_path": str(nc_path.resolve()),
            "indices": indices,
            "time_labels": labels,
            "plot_files": plot_paths,
            "n_frames": len(indices),
            "extract_mode": "batch_nc_open",
            "extract_workers": int(os.environ.get("EDDY_DUAL_EXTRACT_WORKERS", "0")),
        }
        (job_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        return labels

    def _load_dual_job_manifest(self, job_id: str) -> dict[str, Any]:
        job_dir = self._dual_job_dir(job_id)
        mf = job_dir / "manifest.json"
        if not mf.is_file():
            raise FileNotFoundError(f"双路任务不存在或已过期: {job_id}")
        return json.loads(mf.read_text(encoding="utf-8"))

    def complete_dual_mp4_from_job(
        self,
        *,
        job_id: str,
        fps: float = 1.0,
    ) -> dict[str, Any]:
        """阶段二：对缓存帧批量 YOLO，合成带框 MP4（底图 MP4 应在阶段一已生成）。"""
        manifest = self._load_dual_job_manifest(job_id)
        job_dir = self._dual_job_dir(job_id)
        n = int(manifest["n_frames"])
        yolo_frames: list[np.ndarray] = []
        plot_bgrs: list[np.ndarray | None] = []
        for k in range(n):
            yolo_frames.append(np.load(job_dir / f"yolo_{k:04d}.npy"))
            pf = (manifest.get("plot_files") or [None] * n)[k]
            if pf:
                plot_bgrs.append(np.load(job_dir / pf))
            else:
                plot_bgrs.append(None)
        frames_ann, det_stats = self._batch_annotate_yolo_frames(yolo_frames, plot_bgrs)
        labels = list(manifest.get("time_labels") or [])
        detection_timeline: list[dict[str, Any]] = []
        for i, st in enumerate(det_stats):
            row = dict(st)
            row["time"] = labels[i] if i < len(labels) else f"帧 {i}"
            detection_timeline.append(row)
        enc_a = self._write_bgr_frames_mp4(frames_ann, fps=float(fps), task_id=job_id, basename="eddy_nc_ann")
        if enc_a.get("status") != "success":
            return enc_a
        try:
            root = resolve_path(".")
            ann_rel = Path(enc_a["mp4_path"]).resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            ann_rel = enc_a["mp4_path"]
        return {
            "status": "success",
            "phase": "complete",
            "annotated_mp4": ann_rel,
            "preview_annotated": Path(ann_rel).name,
            "n_frames": n,
            "time_labels": manifest.get("time_labels", []),
            "time_indices": manifest.get("indices", []),
            "detection_timeline": detection_timeline,
            "job_id": job_id,
            "video_encoding": enc_a.get("video_encoding"),
        }

    def infer_netcdf_dual_mp4(
        self,
        *,
        nc_path: str,
        time_start: int = 0,
        time_stop: int | None = None,
        time_stride: int = 1,
        fps: float = 1.0,
        max_frames: int = 120,
        task_id: str | None = None,
        deliver: Literal["full", "base", "annotate"] = "full",
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """
        整段 NC 生成双路 MP4：上=无底图/流场，下=带检测框。
        deliver=base：仅读 NC + 编码底图（无 YOLO），可先返回播放；deliver=annotate：对 job 缓存批量 YOLO；
        deliver=full：先底图再批量 YOLO（推理帧数仍受 EDDY_DUAL_MAX_INFER_FRAMES 约束，不减少上限）。
        """
        if deliver == "annotate":
            if not job_id:
                return {"status": "failed", "message": "annotate 阶段需要 job_id"}
            out = self.complete_dual_mp4_from_job(job_id=job_id, fps=float(fps))
            return out

        p = resolve_path(nc_path)
        if not p.is_file():
            return {"status": "failed", "message": f"NC 不存在: {p}"}

        try:
            indices, truncated, plan = self._plan_dual_indices(
                nc_path=p,
                time_start=int(time_start),
                time_stop=time_stop,
                time_stride=int(time_stride),
                max_frames=int(max_frames),
            )
        except ValueError as e:
            return {"status": "failed", "message": str(e)}

        tag = job_id or task_id or uuid.uuid4().hex[:12]
        try:
            time_labels = self._cache_dual_frames(p, indices, tag)
        except Exception as e:
            return {"status": "failed", "message": f"抽帧缓存失败: {e}"}

        frames_base: list[np.ndarray] = []
        job_dir = self._dual_job_dir(tag)
        for k in range(len(indices)):
            frames_base.append(np.load(job_dir / f"base_{k:04d}.npy"))

        enc_b = self._write_bgr_frames_mp4(frames_base, fps=float(fps), task_id=tag, basename="eddy_nc_base")
        if enc_b.get("status") != "success":
            return enc_b

        try:
            root = resolve_path(".")
            base_rel = Path(enc_b["mp4_path"]).resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            base_rel = enc_b["mp4_path"]
        base_name = Path(base_rel).name

        if deliver == "base":
            return {
                "status": "success",
                "phase": "base_ready",
                "job_id": tag,
                "base_mp4": base_rel,
                "preview_base": base_name,
                "fps": float(fps),
                "n_frames": len(frames_base),
                "time_indices": indices,
                "time_labels": time_labels,
                "truncated": truncated,
                "video_encoding": enc_b.get("video_encoding"),
                "video_encoding_note": enc_b.get("video_encoding_note"),
                "meta": {"nc_path": str(p), **plan},
            }

        ann_out = self.complete_dual_mp4_from_job(job_id=tag, fps=float(fps))
        if ann_out.get("status") != "success":
            return ann_out
        try:
            ann_rel = ann_out["annotated_mp4"]
        except KeyError:
            ann_rel = ann_out.get("mp4_path", "")

        return {
            "status": "success",
            "phase": "complete",
            "job_id": tag,
            "base_mp4": base_rel,
            "annotated_mp4": ann_rel,
            "preview_base": base_name,
            "preview_annotated": ann_out.get("preview_annotated", Path(str(ann_rel)).name),
            "fps": float(fps),
            "n_frames": len(frames_base),
            "time_indices": indices,
            "time_labels": time_labels,
            "detection_timeline": ann_out.get("detection_timeline", []),
            "truncated": truncated,
            "video_encoding": enc_b.get("video_encoding"),
            "video_encoding_note": enc_b.get("video_encoding_note"),
            "meta": {"nc_path": str(p), **plan},
        }

    def _infer_bgr(
        self,
        frame_hwc: np.ndarray,
        *,
        task_id: str | None,
        source_type: str,
        extra_meta: dict[str, Any] | None = None,
        bgr_for_plot: np.ndarray | None = None,
    ) -> dict[str, Any]:
        model = self._load_model()
        proc = self._preprocess_frame_hwc(frame_hwc)
        tta_hit = False
        if self.multiscale_tta:
            tta_hit = tta_any_detection(
                model,
                proc,
                base_imgsz=int(self.base_imgsz),
                scales=tuple(self.multiscale_scales),
                conf=float(self.conf),
                iou=float(self.iou),
                device_hint=str(self._yolo_device()),
            )
        pred_list = model.predict(proc, **self._yolo_predict_kwargs())
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
        if pred is not None and hasattr(pred, "plot"):
            raw_base = bgr_for_plot if bgr_for_plot is not None else proc[..., :3]
            base_arr = np.asarray(raw_base)
            if base_arr.dtype != np.uint8:
                base_frame_bgr = np.clip(base_arr, 0, 255).astype(np.uint8)
            else:
                base_frame_bgr = base_arr.copy()
            annotated = self._plot_or_fallback_bgr(pred, proc, bgr_for_plot=bgr_for_plot)
        else:
            raw_base = bgr_for_plot if bgr_for_plot is not None else proc[..., :3]
            base_arr = np.asarray(raw_base)
            if base_arr.dtype != np.uint8:
                base_frame_bgr = np.clip(base_arr, 0, 255).astype(np.uint8)
            else:
                base_frame_bgr = base_arr.copy()
            annotated = base_frame_bgr.copy()
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
            "base_frame_bgr": base_frame_bgr,
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
            proc = self._preprocess_frame_hwc(frame)
            tta_hit = False
            if self.multiscale_tta:
                tta_hit = tta_any_detection(
                    model,
                    proc,
                    base_imgsz=int(self.base_imgsz),
                    scales=tuple(self.multiscale_scales),
                    conf=float(self.conf),
                    iou=float(self.iou),
                    device_hint=str(self._yolo_device()),
                )
            pred_list = model.predict(proc, **self._yolo_predict_kwargs())
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
                plotted = self._plot_or_fallback_bgr(pred, proc, bgr_for_plot=None)
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
        return self._infer_bgr(frame, task_id=task_id, source_type="camera", bgr_for_plot=None)

    def export_geometries_json(self, geometries: list[dict[str, Any]], path: str | Path) -> Path:
        outp = resolve_path(path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(geometries, ensure_ascii=False, indent=2), encoding="utf-8")
        return outp
