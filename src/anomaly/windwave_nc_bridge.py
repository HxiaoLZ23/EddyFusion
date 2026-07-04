"""从含风浪要素的 NetCDF 构建与「配套 NPZ」同形的会话字段，供涡旋页合并或风浪页独立运行。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from src.anomaly.inference import predict_wind_wave_from_series, smooth_baseline
from src.preprocess.anomaly_dataset import extract_wind_wave_series_from_netcdf


def _smooth_baseline(x: np.ndarray) -> np.ndarray:
    """兼容旧引用；新代码请用 `src.anomaly.inference.smooth_baseline`。"""
    return smooth_baseline(x)


def extract_wind_wave_companion_from_netcdf(nc_path: str | Path) -> dict[str, Any] | None:
    """
    提取可 `apply_wind_wave_companion_to_eddy_result` 的字段；缺变量或时序过短则返回 None。
    预测侧默认 WindWaveLSTM 滑窗一步预测；缺权重或序列不足时降级为平滑基线。
    """
    try:
        p = Path(nc_path).expanduser().resolve()
        feat, extract_meta = extract_wind_wave_series_from_netcdf(p)
        tlen = int(feat.shape[0])
        if tlen < 2:
            return None
        pred = predict_wind_wave_from_series(feat)
        assessment_note = None
        if pred.prediction_backend == "smooth_fallback":
            assessment_note = (
                f"风浪预测已降级为平滑基线（{pred.fallback_reason}）；"
                "请同步 outputs/anomaly/best.pt 或上传更长时序 NC。"
            )
        out: dict[str, Any] = {
            "demo_wind_observed": pred.wind_observed,
            "demo_wind_predicted": pred.wind_predicted,
            "demo_wave_observed": pred.wave_observed,
            "demo_wave_predicted": pred.wave_predicted,
            "wind_wave_from_companion_npz": True,
            "wind_wave_from_netcdf": True,
            "wind_wave_nc_extract_meta": extract_meta,
            "prediction_backend": pred.prediction_backend,
            "wind_wave_prediction_meta": pred.meta,
        }
        if assessment_note:
            out["wind_wave_assessment_note"] = assessment_note
        return out
    except Exception:
        return None


def wind_timeline_and_peak_from_companion(companion: dict[str, Any]) -> tuple[list[dict[str, Any]], float]:
    """由 demo_wind/demo_wave 列表生成风浪时间轴与 combo 峰值（与 build_anomaly 用曲线一致）。"""
    wo = np.asarray(companion["demo_wind_observed"], dtype=np.float64)
    wp = np.asarray(companion["demo_wind_predicted"], dtype=np.float64)
    ho = np.asarray(companion["demo_wave_observed"], dtype=np.float64)
    hp = np.asarray(companion["demo_wave_predicted"], dtype=np.float64)
    n = min(wo.size, wp.size, ho.size, hp.size)
    if n < 1:
        return [], 0.0
    wo, wp, ho, hp = wo[:n], wp[:n], ho[:n], hp[:n]
    combo = np.abs(wo - wp) + np.abs(ho - hp)
    timeline = [
        {"time": f"T+{i}", "event": "NC 风浪分量", "score": float(combo[i]), "count": 0} for i in range(int(combo.shape[0]))
    ]
    peak = float(np.max(combo)) if combo.size else 0.0
    return timeline, peak


def build_eddy_result_from_windwave_netcdf(nc_path: str | Path) -> dict[str, Any]:
    """
    仅风浪、无涡旋 YOLO 时的整包 `eddy_last_result`（风浪预警页「独立 NC」入口）。
    """
    p = Path(nc_path).expanduser().resolve()
    companion = extract_wind_wave_companion_from_netcdf(p)
    if companion is None:
        raise ValueError("无法从该 NC 提取风浪时序（需 u10/v10 或有效波高等变量）。")
    timeline, peak = wind_timeline_and_peak_from_companion(companion)
    meta = companion.get("wind_wave_nc_extract_meta") or {}
    pred_backend = companion.get("prediction_backend", "unknown")
    out: dict[str, Any] = {
        "status": "success",
        "module": "eddy",
        "mode": "real",
        "source_type": "netcdf_windwave",
        "summary": (
            f"已由本页上传 NC 构建风浪时序（T={len(timeline)}，预测={pred_backend}），无需先跑涡旋。"
        ),
        "timeline": timeline,
        "peak_score": peak,
        "preview_images": [],
        "geometries": [],
        "generated_at": int(time.time()),
        "meta": {
            "nc_path": str(p),
            "wind_wave_extract": meta,
            "prediction_backend": pred_backend,
            "wind_wave_prediction": companion.get("wind_wave_prediction_meta"),
        },
    }
    out.update(companion)
    return out
