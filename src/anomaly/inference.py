"""WindWaveLSTM 在线滑窗推理（与 train/eval 及 anomaly_plot 口径一致）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.anomaly.model import build_model
from src.utils.config import load_yaml, pick_device, resolve_path


def window_steps_from_cfg(cfg: dict[str, Any], data_cfg: dict[str, Any] | None = None) -> tuple[int, int, int]:
    """返回 (window_steps, horizon_steps, time_step_hours)。"""
    pre = (data_cfg or {}).get("anomaly_preprocess") or {}
    step_h = int(pre.get("time_step_hours", 3))
    win_h = int(cfg["data"].get("window_hours", 48))
    hor_h = int(cfg["data"].get("horizon_hours", 1))
    w = max(1, win_h // max(step_h, 1))
    h = max(1, (hor_h + step_h - 1) // max(step_h, 1))
    return w, h, step_h


def smooth_baseline(x: np.ndarray) -> np.ndarray:
    """演示降级：对观测做 3 点平滑（非 LSTM 预报）。"""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 3:
        return x.copy()
    k = np.array([0.25, 0.5, 0.25], dtype=np.float64)
    y = np.convolve(x, k, mode="same")
    y[0] = x[0]
    y[-1] = x[-1]
    return y


def resolve_anomaly_ckpt_path(cfg: dict[str, Any], ckpt_path: str | Path | None = None) -> Path:
    if ckpt_path is not None:
        return resolve_path(ckpt_path)
    out_dir = resolve_path(cfg["paths"]["output_dir"])
    return out_dir / "best.pt"


def load_anomaly_model(
    *,
    config_path: str | Path = "config/anomaly.yaml",
    ckpt_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    cfg = load_yaml(str(config_path))
    dev_str = pick_device(str(device or cfg.get("train", {}).get("device", "cuda")))
    dev = torch.device(dev_str)
    ckpt = resolve_anomaly_ckpt_path(cfg, ckpt_path)
    if not ckpt.is_file():
        raise FileNotFoundError(f"风浪 LSTM 权重不存在: {ckpt}")
    model = build_model(cfg).to(dev)
    try:
        state = torch.load(ckpt, map_location=dev, weights_only=False)
    except TypeError:
        state = torch.load(ckpt, map_location=dev)
    model.load_state_dict(state["model"])
    model.eval()
    meta = {
        "config_path": str(config_path),
        "ckpt_path": str(ckpt),
        "device": dev_str,
        "window_hours": int(cfg["data"].get("window_hours", 48)),
        "horizon_hours": int(cfg["data"].get("horizon_hours", 1)),
    }
    return model, meta


_MODEL_CACHE: dict[tuple[str, str, str], torch.nn.Module] = {}


def get_cached_anomaly_model(
    *,
    config_path: str | Path = "config/anomaly.yaml",
    ckpt_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> tuple[torch.nn.Module, dict[str, Any], torch.device]:
    cfg = load_yaml(str(config_path))
    dev_str = pick_device(str(device or cfg.get("train", {}).get("device", "cuda")))
    ckpt = resolve_anomaly_ckpt_path(cfg, ckpt_path)
    key = (str(resolve_path(config_path)), str(ckpt), dev_str)
    if key not in _MODEL_CACHE:
        model, meta = load_anomaly_model(config_path=config_path, ckpt_path=ckpt, device=dev_str)
        _MODEL_CACHE[key] = model
    else:
        model = _MODEL_CACHE[key]
        meta = {
            "config_path": str(config_path),
            "ckpt_path": str(ckpt),
            "device": dev_str,
            "window_hours": int(cfg["data"].get("window_hours", 48)),
            "horizon_hours": int(cfg["data"].get("horizon_hours", 1)),
        }
    return model, meta, torch.device(dev_str)


@torch.no_grad()
def rolling_predict(
    model: torch.nn.Module,
    series: np.ndarray,
    *,
    window_steps: int,
    horizon_steps: int,
    device: torch.device,
    plot_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """series (T,2) → 目标时刻索引、观测、预测、残差（仅可滑窗段）。"""
    ts = np.asarray(series, dtype=np.float32)
    if ts.ndim != 2 or ts.shape[1] != 2:
        raise ValueError(f"series 须为 (T,2)，实际 {ts.shape}")
    t_len = int(ts.shape[0])
    need = window_steps + horizon_steps
    if t_len < need:
        raise ValueError(f"序列长度 T={t_len} 不足窗口需求 {need}")

    starts = list(range(0, t_len - need + 1, max(1, plot_stride)))
    t_idx: list[int] = []
    obs: list[np.ndarray] = []
    prd: list[np.ndarray] = []
    for s in starts:
        tgt = s + window_steps + horizon_steps - 1
        x = torch.from_numpy(ts[s : s + window_steps]).float().unsqueeze(0).to(device)
        p = model(x).cpu().numpy()[0]
        t_idx.append(tgt)
        obs.append(ts[tgt])
        prd.append(p)
    t_arr = np.asarray(t_idx, dtype=np.int32)
    o_arr = np.stack(obs, axis=0)
    p_arr = np.stack(prd, axis=0)
    return t_arr, o_arr, p_arr, o_arr - p_arr


def _smooth_fallback_series(feat: np.ndarray, *, reason: str, extra_meta: dict[str, Any] | None = None) -> "WindWavePredictionResult":
    wind = feat[:, 0].astype(np.float64)
    wave = feat[:, 1].astype(np.float64)
    meta = {
        "backend": "smooth_fallback",
        "fallback_reason": reason,
        **(extra_meta or {}),
    }
    return WindWavePredictionResult(
        wind_observed=wind.tolist(),
        wind_predicted=smooth_baseline(wind).tolist(),
        wave_observed=wave.tolist(),
        wave_predicted=smooth_baseline(wave).tolist(),
        prediction_backend="smooth_fallback",
        fallback_reason=reason,
        meta=meta,
    )


@dataclass
class WindWavePredictionResult:
    wind_observed: list[float]
    wind_predicted: list[float]
    wave_observed: list[float]
    wave_predicted: list[float]
    prediction_backend: str
    fallback_reason: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def predict_wind_wave_from_series(
    feat: np.ndarray,
    *,
    config_path: str | Path = "config/anomaly.yaml",
    ckpt_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> WindWavePredictionResult:
    """
    从 (T,2) 风/浪序列生成与线上一致的 obs/pred 四列表。
    默认 LSTM 滑窗一步预测；缺权重或 T 不足时降级为 smooth_baseline。
    """
    ts = np.asarray(feat, dtype=np.float32)
    if ts.ndim != 2 or ts.shape[1] != 2 or ts.shape[0] < 2:
        raise ValueError(f"feat 须为长度≥2 的 (T,2)，实际 {ts.shape}")

    cfg = load_yaml(str(config_path))
    data_cfg = load_yaml("config/data.yaml")
    window_steps, horizon_steps, step_h = window_steps_from_cfg(cfg, data_cfg)
    need = window_steps + horizon_steps
    t_len = int(ts.shape[0])

    ckpt = resolve_anomaly_ckpt_path(cfg, ckpt_path)
    if not ckpt.is_file():
        return _smooth_fallback_series(
            ts,
            reason="checkpoint_missing",
            extra_meta={"ckpt_path": str(ckpt), "window_steps": window_steps, "horizon_steps": horizon_steps},
        )
    if t_len < need:
        return _smooth_fallback_series(
            ts,
            reason=f"series_too_short:T={t_len}<need={need}",
            extra_meta={"ckpt_path": str(ckpt), "window_steps": window_steps, "horizon_steps": horizon_steps},
        )

    try:
        model, load_meta, dev = get_cached_anomaly_model(
            config_path=config_path,
            ckpt_path=ckpt,
            device=device,
        )
        _, obs_arr, pred_arr, _ = rolling_predict(
            model,
            ts,
            window_steps=window_steps,
            horizon_steps=horizon_steps,
            device=dev,
            plot_stride=1,
        )
    except Exception as exc:
        return _smooth_fallback_series(
            ts,
            reason=f"lstm_inference_error:{type(exc).__name__}",
            extra_meta={"detail": str(exc), "ckpt_path": str(ckpt)},
        )

    if obs_arr.shape[0] < 2:
        return _smooth_fallback_series(
            ts,
            reason="lstm_valid_steps_lt_2",
            extra_meta={"ckpt_path": str(ckpt)},
        )

    meta = {
        "backend": "lstm",
        "config_path": str(config_path),
        "ckpt_path": str(ckpt),
        "device": str(dev),
        "window_steps": window_steps,
        "horizon_steps": horizon_steps,
        "time_step_hours": step_h,
        "original_T": t_len,
        "valid_T": int(obs_arr.shape[0]),
        "warmup_steps_dropped": int(t_len - obs_arr.shape[0]),
        **load_meta,
    }
    return WindWavePredictionResult(
        wind_observed=obs_arr[:, 0].astype(float).tolist(),
        wind_predicted=pred_arr[:, 0].astype(float).tolist(),
        wave_observed=obs_arr[:, 1].astype(float).tolist(),
        wave_predicted=pred_arr[:, 1].astype(float).tolist(),
        prediction_backend="lstm",
        fallback_reason=None,
        meta=meta,
    )
