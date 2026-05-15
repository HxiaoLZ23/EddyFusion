"""从含风浪要素的 NetCDF 构建与「配套 NPZ」同形的会话字段，供涡旋页合并或风浪页独立运行。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from src.preprocess.anomaly_dataset import extract_wind_wave_series_from_netcdf


def _smooth_baseline(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size < 3:
        return x.copy()
    k = np.array([0.25, 0.5, 0.25], dtype=np.float64)
    y = np.convolve(x, k, mode="same")
    y[0] = x[0]
    y[-1] = x[-1]
    return y


NC_WIND_ASSESSMENT_NOTE = (
    "NetCDF 格点：由 u10/v10 模长与有效波高（或可用变量推导）得到时序；"
    "pred 为平滑基线，用于演示 obs−pred 残差与 DTW；非命题方评测口径。"
)


def extract_wind_wave_companion_from_netcdf(nc_path: str | Path) -> dict[str, Any] | None:
    """
    提取可 `apply_wind_wave_companion_to_eddy_result` 的字段；缺变量或时序过短则返回 None。
    """
    try:
        p = Path(nc_path).expanduser().resolve()
        feat, meta = extract_wind_wave_series_from_netcdf(p)
        tlen = int(feat.shape[0])
        if tlen < 2:
            return None
        wind = feat[:, 0].astype(np.float64)
        wave = feat[:, 1].astype(np.float64)
        wp = _smooth_baseline(wind)
        hp = _smooth_baseline(wave)
        return {
            "demo_wind_observed": wind.tolist(),
            "demo_wind_predicted": wp.tolist(),
            "demo_wave_observed": wave.tolist(),
            "demo_wave_predicted": hp.tolist(),
            "wind_wave_from_companion_npz": True,
            "wind_wave_from_netcdf": True,
            "wind_wave_assessment_note": NC_WIND_ASSESSMENT_NOTE,
            "wind_wave_nc_extract_meta": meta,
        }
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
    out: dict[str, Any] = {
        "status": "success",
        "module": "eddy",
        "mode": "real",
        "source_type": "netcdf_windwave",
        "summary": f"已由本页上传 NC 构建风浪时序上下文（T={len(timeline)}），无需先跑涡旋。",
        "timeline": timeline,
        "peak_score": peak,
        "preview_images": [],
        "geometries": [],
        "generated_at": int(time.time()),
        "meta": {"nc_path": str(p), "wind_wave_extract": meta},
    }
    out.update(companion)
    return out

