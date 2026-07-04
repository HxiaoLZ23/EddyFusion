"""指定超前小时（如 24h）的风浪 LSTM 评估：自回归 rollout + 持续性基线。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.anomaly.model import build_model
from src.preprocess.anomaly_dataset import _build_windows, discover_anomaly_month_dirs, _concat_month_dirs
from src.utils.config import load_yaml, resolve_path


def horizon_steps_from_hours(horizon_hours: int, time_step_hours: int) -> int:
    return max(1, (int(horizon_hours) + max(int(time_step_hours), 1) - 1) // max(int(time_step_hours), 1))


def window_steps_from_cfg(cfg: dict, data_cfg: dict | None) -> tuple[int, int]:
    pre = (data_cfg or {}).get("anomaly_preprocess") or {}
    step_h = int(pre.get("time_step_hours", 3))
    win_h = int(cfg["data"].get("window_hours", 48))
    w = max(1, win_h // max(step_h, 1))
    return w, step_h


def load_split_continuous_series(
    *,
    split: str,
    data_cfg: dict,
) -> tuple[np.ndarray, dict[str, Any]]:
    """从命题方按月 NC 拼接 val/test/train 连续 (T,2) 序列（与 NPZ 划分一致）。"""
    raw_root = resolve_path(data_cfg["paths"]["raw_root"])
    subdir = (data_cfg.get("anomaly_preprocess") or {}).get("subdir") or "风浪异常识别"
    ysplit = data_cfg.get("anomaly_year_split") or {}
    if split == "train":
        tr = ysplit.get("train") or {}
        months = discover_anomaly_month_dirs(
            raw_root,
            subdir,
            year_min=int(tr.get("min_year", 2014)),
            year_max=int(tr.get("max_year", 2023)),
        )
    elif split == "val":
        months = discover_anomaly_month_dirs(
            raw_root,
            subdir,
            years=set(int(x) for x in ysplit.get("val_years", [2025])),
        )
    elif split == "test":
        months = discover_anomaly_month_dirs(
            raw_root,
            subdir,
            years=set(int(x) for x in ysplit.get("test_years", [2024])),
        )
    else:
        raise ValueError(f"未知 split: {split}")
    series, meta = _concat_month_dirs(months)
    meta["split"] = split
    meta["series_shape"] = list(series.shape)
    return np.asarray(series, dtype=np.float32), meta


@torch.no_grad()
def rollout_predict_one(
    model: torch.nn.Module,
    window: np.ndarray,
    *,
    rollout_steps: int,
    device: torch.device,
) -> np.ndarray:
    """从 (W,2) 窗口自回归 rollout_steps 次，返回最后一步预测 (2,)。"""
    win = np.asarray(window, dtype=np.float32).copy()
    if win.ndim != 2 or win.shape[1] != 2:
        raise ValueError(f"window 须为 (W,2)，实际 {win.shape}")
    pred = np.zeros(2, dtype=np.float32)
    for _ in range(max(1, int(rollout_steps))):
        x = torch.from_numpy(win).float().unsqueeze(0).to(device)
        pred = model(x).cpu().numpy()[0].astype(np.float32)
        win = np.vstack([win[1:], pred.reshape(1, 2)])
    return pred


@torch.no_grad()
def run_horizon_eval(
    cfg: dict,
    ckpt: Path,
    device: torch.device,
    *,
    split: str,
    data_cfg: dict | None = None,
    horizon_hours: int = 24,
    stride: int | None = None,
) -> dict[str, Any]:
    """
    在物理量纲上评估指定超前（默认 24h）MAE/RMSE。
    当前 best.pt 为 3h 一步训练；horizon_hours>3h 时使用 **自回归 rollout**。
    """
    data_cfg = data_cfg or load_yaml("config/data.yaml")
    pre = data_cfg.get("anomaly_preprocess") or {}
    step_h = int(pre.get("time_step_hours", 3))
    window_steps, _ = window_steps_from_cfg(cfg, data_cfg)
    horizon_steps = horizon_steps_from_hours(horizon_hours, step_h)
    if stride is None:
        stride = int(pre.get("window_stride", 1))

    series, series_meta = load_split_continuous_series(split=split, data_cfg=data_cfg)
    if series.shape[0] < window_steps + horizon_steps:
        raise ValueError(
            f"{split} 序列 T={series.shape[0]} 不足 window={window_steps}+horizon={horizon_steps}"
        )

    x_all, y_all = _build_windows(series, window_steps, horizon_steps, max(1, stride))
    model = build_model(cfg).to(device)
    try:
        state = torch.load(ckpt, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    train_horizon_steps = horizon_steps_from_hours(int(cfg["data"].get("horizon_hours", 1)), step_h)
    preds: list[np.ndarray] = []
    pers: list[np.ndarray] = []
    for i in range(x_all.shape[0]):
        win = x_all[i]
        preds.append(
            rollout_predict_one(model, win, rollout_steps=horizon_steps, device=device)
        )
        pers.append(win[-1].astype(np.float32))

    pred_arr = np.stack(preds, axis=0)
    pers_arr = np.stack(pers, axis=0)
    y_arr = np.asarray(y_all, dtype=np.float64)

    err = pred_arr.astype(np.float64) - y_arr
    err_p = pers_arr.astype(np.float64) - y_arr
    mae = np.abs(err).mean(axis=0)
    rmse = np.sqrt((err**2).mean(axis=0))
    mae_p = np.abs(err_p).mean(axis=0)

    label_stats: dict[str, float] = {}
    for j, name in enumerate(("wind", "wave")):
        col = y_arr[:, j]
        label_stats[f"label_{name}_mean"] = float(col.mean())
        label_stats[f"label_{name}_std"] = float(col.std())
        label_stats[f"label_{name}_min"] = float(col.min())
        label_stats[f"label_{name}_max"] = float(col.max())
        label_stats[f"persistence_mae_{name}"] = float(mae_p[j])
    pers_avg = float(mae_p.mean())
    mae_avg = float(mae.mean())

    return {
        "mae_wind": float(mae[0]),
        "mae_wave": float(mae[1]),
        "mae_avg": mae_avg,
        "rmse_wind": float(rmse[0]),
        "rmse_wave": float(rmse[1]),
        "rmse_avg": float(rmse.mean()),
        "split": split,
        **label_stats,
        "persistence_mae_avg": pers_avg,
        "mae_avg_vs_persistence_ratio": float(mae_avg / max(pers_avg, 1e-9)),
        "n_samples": int(x_all.shape[0]),
        "horizon_hours": int(horizon_hours),
        "horizon_steps": int(horizon_steps),
        "window_steps": int(window_steps),
        "time_step_hours": int(step_h),
        "window_stride": int(stride),
        "eval_mode": "autoregressive_rollout",
        "train_horizon_hours": int(train_horizon_steps * step_h),
        "target_units": "wind_mps_wave_m",
        "normalization": "none",
        "eval_note": (
            f"权重按 {train_horizon_steps * step_h}h 一步训练；本评估对 {horizon_hours}h 采用 {horizon_steps} 步自回归 rollout。"
            "持续性基线为窗口末时刻外推。MAE 为物理单位，无 StandardScaler。"
        ),
        "series_meta": series_meta,
    }
