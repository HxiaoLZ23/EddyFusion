"""水文 Z-score 与物理量纲（热力图/曲线展示用）。"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.utils.config import load_yaml, resolve_path

FEATURE_UNITS: dict[str, str] = {
    "temp": "°C",
    "sal": "PSU",
    "u": "m/s",
    "v": "m/s",
}


def feature_unit(name: str) -> str:
    return FEATURE_UNITS.get((name or "").strip().lower(), "")


def feature_units_map(names: list[str]) -> dict[str, str]:
    return {n: feature_unit(n) for n in names}


def stats_vectors_usable(mean: np.ndarray, std: np.ndarray, *, n_channels: int) -> bool:
    mu = np.asarray(mean, dtype=np.float64).reshape(-1)
    sd = np.maximum(np.asarray(std, dtype=np.float64).reshape(-1), 0.0)
    if mu.size < n_channels or sd.size < n_channels:
        return False
    mu = mu[:n_channels]
    sd = sd[:n_channels]
    return bool(np.isfinite(mu).all() and np.isfinite(sd).all() and (sd > 1e-8).all())


def resolve_zscore_stats(
    *,
    config_path: str,
    materialize_meta: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """返回 (mean_1d, std_1d, feature_names)，与 target_features 顺序对齐。"""
    cfg = load_yaml(config_path)
    names = list(cfg["data"]["target_features"])
    meta = materialize_meta or {}

    if "zscore_mean_1d" in meta and "zscore_std_1d" in meta:
        mu = np.asarray(meta["zscore_mean_1d"], dtype=np.float64).reshape(-1)
        sd = np.asarray(meta["zscore_std_1d"], dtype=np.float64).reshape(-1)
        feats = [str(x) for x in (meta.get("zscore_features") or names)]
        if stats_vectors_usable(mu, sd, n_channels=len(names)):
            idx = [feats.index(n) if n in feats else i for i, n in enumerate(names)]
            return mu[idx], np.maximum(sd[idx], 1e-6), names

    sp = resolve_path("data/processed/stats/hydro_zscore.npz")
    if sp.is_file():
        z = np.load(sp)
        mu = np.asarray(z["mean"], dtype=np.float64).reshape(-1)
        sd = np.asarray(z["std"], dtype=np.float64).reshape(-1)
        feats = [str(x) for x in z["features"].tolist()] if "features" in z.files else names
        if stats_vectors_usable(mu, sd, n_channels=len(names)):
            idx = [feats.index(n) if n in feats else i for i, n in enumerate(names)]
            return mu[idx], np.maximum(sd[idx], 1e-6), names

    return None


def denorm_array(z: np.ndarray, feat_index: int, mean_1d: np.ndarray, std_1d: np.ndarray) -> np.ndarray:
    return z.astype(np.float64) * float(std_1d[feat_index]) + float(mean_1d[feat_index])


def finite_vmin_vmax(arr: np.ndarray, *, floor_zero: bool = False) -> tuple[float, float]:
    flat = np.asarray(arr, dtype=np.float64).ravel()
    mask = np.isfinite(flat)
    if not mask.any():
        return (0.0, 1.0)
    vmin = float(np.min(flat[mask]))
    vmax = float(np.max(flat[mask]))
    if floor_zero:
        vmin = 0.0
    if vmin >= vmax:
        vmax = vmin + 1.0
    return vmin, vmax
