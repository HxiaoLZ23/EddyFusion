"""格点场风浪 3h（可配置超前）LSTM 评估：逐格点滑窗 + 掩膜聚合 MAE/RMSE。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.anomaly.horizon_eval import horizon_steps_from_hours, window_steps_from_cfg
from src.anomaly.model import build_model
from src.preprocess.anomaly_dataset import (
    _build_windows,
    _concat_month_dirs,
    _pick_dataarray,
    discover_anomaly_month_dirs,
)
from src.preprocess.netcdf_io import open_netcdf_dataset
from src.utils.config import load_yaml, resolve_path


def _align_wave_to_wind(wave_thw: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """将 SWH 格网对齐到风场网格（双线性）。"""
    wave = np.asarray(wave_thw, dtype=np.float32)
    t_len = int(wave.shape[0])
    out = np.empty((t_len, target_h, target_w), dtype=np.float32)
    for ti in range(t_len):
        out[ti] = cv2.resize(wave[ti], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return out


def extract_wind_wave_grid_from_month(month_dir: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """
    从单月 oper+wave 目录读取格点场。
    返回 field (T,H,W,2)：[|U10|, SWH]；陆海 NaN 保留。
    """
    md = Path(month_dir)
    oper_candidates = sorted(md.glob("*oper*.nc"))
    wave_candidates = sorted(md.glob("*wave*.nc"))
    if not oper_candidates:
        raise FileNotFoundError(f"未找到 oper NC: {md}")

    meta: dict[str, Any] = {"month_dir": str(md)}
    wind_thw: np.ndarray | None = None
    wave_thw: np.ndarray | None = None

    ds, tmp = open_netcdf_dataset(oper_candidates[0])
    try:
        u_da = _pick_dataarray(ds, ["u10", "U10", "10u", "uwnd", "u_wind"], required=False)
        v_da = _pick_dataarray(ds, ["v10", "V10", "10v", "vwnd", "v_wind"], required=False)
        if u_da is None or v_da is None:
            raise KeyError(f"oper 缺 u10/v10: {md}")
        u = np.asarray(u_da.values, dtype=np.float64)
        v = np.asarray(v_da.values, dtype=np.float64)
        wind_thw = np.sqrt(u**2 + v**2).astype(np.float32)
        meta["wind_shape"] = list(wind_thw.shape)
    finally:
        ds.close()
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)  # type: ignore[arg-type]
            except OSError:
                pass

    if wave_candidates:
        ds, tmp = open_netcdf_dataset(wave_candidates[0])
        try:
            swh_da = _pick_dataarray(
                ds, ["swh", "SWH", "hs", "wave_height", "significant_wave_height"], required=False
            )
            if swh_da is None:
                raise KeyError(f"wave 缺 swh: {md}")
            wave_thw = np.asarray(swh_da.values, dtype=np.float32)
            meta["wave_shape"] = list(wave_thw.shape)
        finally:
            ds.close()
            if tmp is not None:
                try:
                    tmp.unlink(missing_ok=True)  # type: ignore[arg-type]
                except OSError:
                    pass

    assert wind_thw is not None
    t_w, h_w, w_w = wind_thw.shape
    if wave_thw is None:
        wave_aligned = np.full((t_w, h_w, w_w), np.nan, dtype=np.float32)
        meta["used_wave_fallback"] = True
    else:
        if wave_thw.shape[1:] != (h_w, w_w):
            wave_aligned = _align_wave_to_wind(wave_thw, h_w, w_w)
            meta["wave_resampled"] = True
        else:
            wave_aligned = wave_thw
            meta["wave_resampled"] = False
        t_len = min(t_w, wave_aligned.shape[0])
        wind_thw = wind_thw[:t_len]
        wave_aligned = wave_aligned[:t_len]
        meta["used_wave_fallback"] = False

    field = np.stack([wind_thw, wave_aligned], axis=-1)
    meta["field_shape"] = list(field.shape)
    return field, meta


def _concat_month_grid_fields(month_dirs: list[Path]) -> tuple[np.ndarray, dict[str, Any]]:
    parts: list[np.ndarray] = []
    meta: dict[str, Any] = {"months_used": 0, "months_skipped": 0}
    for md in month_dirs:
        try:
            field, m = extract_wind_wave_grid_from_month(md)
        except (FileNotFoundError, KeyError):
            meta["months_skipped"] = int(meta["months_skipped"]) + 1
            continue
        if field.shape[0] > 0:
            parts.append(field)
            meta["months_used"] = int(meta["months_used"]) + 1
            if "wind_shape" in m:
                meta["wind_shape"] = m["wind_shape"]
            if "wave_resampled" in m:
                meta["wave_resampled"] = m["wave_resampled"]
    if not parts:
        return np.empty((0, 0, 0, 2), dtype=np.float32), meta
    cat = np.concatenate(parts, axis=0)
    meta["T"] = int(cat.shape[0])
    meta["grid_hw"] = [int(cat.shape[1]), int(cat.shape[2])]
    return cat, meta


def load_split_grid_field(*, split: str, data_cfg: dict) -> tuple[np.ndarray, dict[str, Any]]:
    raw_root = resolve_path(data_cfg["paths"]["raw_root"])
    subdir = (data_cfg.get("anomaly_preprocess") or {}).get("subdir") or "风浪异常识别"
    ysplit = data_cfg.get("anomaly_year_split") or {}
    if split == "train":
        tr = ysplit.get("train") or {}
        months = discover_anomaly_month_dirs(
            raw_root, subdir, year_min=int(tr.get("min_year", 2014)), year_max=int(tr.get("max_year", 2023))
        )
    elif split == "val":
        months = discover_anomaly_month_dirs(raw_root, subdir, years=set(int(x) for x in ysplit.get("val_years", [2025])))
    elif split == "test":
        months = discover_anomaly_month_dirs(raw_root, subdir, years=set(int(x) for x in ysplit.get("test_years", [2024])))
    else:
        raise ValueError(split)
    field, meta = _concat_month_grid_fields(months)
    meta["split"] = split
    return field, meta


def _ocean_point_mask(field: np.ndarray, *, min_finite_frac: float = 0.8) -> np.ndarray:
    """(H,W) bool：时间维上足够多有限值的海洋格点。"""
    finite = np.isfinite(field).all(axis=-1)  # (T,H,W)
    frac = finite.mean(axis=0)
    return frac >= float(min_finite_frac)


def _subsample_mask(mask: np.ndarray, space_stride: int) -> np.ndarray:
    if space_stride <= 1:
        return mask
    ss = int(space_stride)
    sub = np.zeros_like(mask, dtype=bool)
    sub[::ss, ::ss] = mask[::ss, ::ss]
    return sub


@torch.no_grad()
def run_grid_eval(
    cfg: dict,
    ckpt: Path,
    device: torch.device,
    *,
    split: str,
    data_cfg: dict | None = None,
    horizon_hours: int = 3,
    time_stride: int = 4,
    space_stride: int = 2,
    batch_size: int = 4096,
    min_finite_frac: float = 0.8,
) -> dict[str, Any]:
    """
    格点场 3h 一步（默认）MAE/RMSE：对每个有效格点独立滑窗，再聚合。
    与区域平均 eval 不同，此处保留 (H,W) 空间结构。
    """
    data_cfg = data_cfg or load_yaml("config/data.yaml")
    pre = data_cfg.get("anomaly_preprocess") or {}
    step_h = int(pre.get("time_step_hours", 3))
    window_steps, _ = window_steps_from_cfg(cfg, data_cfg)
    horizon_steps = horizon_steps_from_hours(horizon_hours, step_h)

    field, series_meta = load_split_grid_field(split=split, data_cfg=data_cfg)
    if field.size == 0:
        raise ValueError(f"{split} 格点场为空")
    t_len, h_gr, w_gr, _ = field.shape
    need = window_steps + horizon_steps
    if t_len < need:
        raise ValueError(f"T={t_len} 不足 window+horizon={need}")

    omask = _subsample_mask(_ocean_point_mask(field, min_finite_frac=min_finite_frac), space_stride)
    ij = np.argwhere(omask)
    n_pts = int(ij.shape[0])
    if n_pts < 1:
        raise ValueError("无有效海洋格点")

    model = build_model(cfg).to(device)
    try:
        state = torch.load(ckpt, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    starts = list(range(0, t_len - need + 1, max(1, int(time_stride))))
    abs_err_sum = np.zeros(2, dtype=np.float64)
    sq_err_sum = np.zeros(2, dtype=np.float64)
    pers_abs_sum = np.zeros(2, dtype=np.float64)
    n_eval = 0

    for s in starts:
        tgt_t = s + window_steps + horizon_steps - 1
        win = field[s : s + window_steps]  # (W,H,G,2)
        y_true = field[tgt_t]  # (H,W,2)
        pers_ref = field[s + window_steps - 1]

        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        yp: list[np.ndarray] = []
        for i, j in ij:
            ts_ij = win[:, i, j, :]
            if not np.isfinite(ts_ij).all():
                continue
            yt = y_true[i, j]
            if not np.isfinite(yt).all():
                continue
            xs.append(ts_ij.astype(np.float32))
            ys.append(yt.astype(np.float32))
            yp.append(pers_ref[i, j].astype(np.float32))

        if not xs:
            continue
        x_arr = np.stack(xs, axis=0)
        y_arr = np.stack(ys, axis=0)
        p_arr = np.stack(yp, axis=0)

        preds: list[np.ndarray] = []
        for b0 in range(0, x_arr.shape[0], batch_size):
            xb = torch.from_numpy(x_arr[b0 : b0 + batch_size]).float().to(device)
            preds.append(model(xb).cpu().numpy())
        pred_arr = np.concatenate(preds, axis=0)

        err = pred_arr.astype(np.float64) - y_arr.astype(np.float64)
        err_p = p_arr.astype(np.float64) - y_arr.astype(np.float64)
        abs_err_sum += np.abs(err).sum(axis=0)
        sq_err_sum += (err**2).sum(axis=0)
        pers_abs_sum += np.abs(err_p).sum(axis=0)
        n_eval += int(y_arr.shape[0])

    if n_eval < 1:
        raise ValueError("无有效格点-时刻样本")

    mae = abs_err_sum / n_eval
    rmse = np.sqrt(sq_err_sum / n_eval)
    mae_p = pers_abs_sum / n_eval
    mae_avg = float(mae.mean())
    pers_avg = float(mae_p.mean())

    # 标签统计：在有效格点与可用时刻上
    sample_vals = field[np.isfinite(field).all(axis=-1)]
    label_w = sample_vals[:, 0]
    label_h = sample_vals[:, 1]

    return {
        "mae_wind": float(mae[0]),
        "mae_wave": float(mae[1]),
        "mae_avg": mae_avg,
        "rmse_wind": float(rmse[0]),
        "rmse_wave": float(rmse[1]),
        "rmse_avg": float(rmse.mean()),
        "persistence_mae_wind": float(mae_p[0]),
        "persistence_mae_wave": float(mae_p[1]),
        "persistence_mae_avg": pers_avg,
        "mae_avg_vs_persistence_ratio": float(mae_avg / max(pers_avg, 1e-9)),
        "split": split,
        "n_grid_points": n_pts,
        "n_eval_samples": int(n_eval),
        "horizon_hours": int(horizon_hours),
        "horizon_steps": int(horizon_steps),
        "window_steps": int(window_steps),
        "time_step_hours": int(step_h),
        "time_stride": int(time_stride),
        "space_stride": int(space_stride),
        "eval_mode": "grid_point_independent",
        "label_wind_mean": float(label_w.mean()),
        "label_wind_std": float(label_w.std()),
        "label_wave_mean": float(label_h.mean()),
        "label_wave_std": float(label_h.std()),
        "target_units": "wind_mps_wave_m",
        "normalization": "none",
        "eval_note": (
            f"格点独立滑窗；SWH 必要时双线性对齐到风场网格。"
            f"默认 time_stride={time_stride}、space_stride={space_stride} 以控制算量；"
            "MAE 为所有有效格点×时刻样本的 micro-average。"
        ),
        "series_meta": series_meta,
    }
