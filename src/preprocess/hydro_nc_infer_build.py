"""从海域要素风格 NetCDF（可多日文件拼接）构建水文推理用 X/y 滑窗，供 Streamlit 临时 NPZ。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.preprocess.hydro_nc_stack import (
    apply_zscore,
    build_windows,
    stack_hydro_fields,
    zscore_fit,
)
from src.utils.config import resolve_path


def sort_hydro_nc_paths(paths: list[Path]) -> list[Path]:
    """按 YYYYMMDD 风格 stem 排序，其它文件名置后。"""

    def key(p: Path) -> tuple:
        s = p.stem
        if s.isdigit() and len(s) >= 8:
            return (0, s)
        return (1, str(p))

    return sorted(paths, key=key)


def required_window_length(cfg: dict[str, Any]) -> int:
    """单条 ConvLSTM 样本所需连续时间步：input_steps + output_steps。"""
    d = cfg["data"]
    return int(d["input_steps"]) + int(d["output_steps"])


def peek_total_time_steps(nc_paths: list[str | Path], feats: list[str]) -> int:
    """
    仅统计多文件拼接后的时间维总长（读元数据，不加载全量格点）。
    使用首个 input_feature 对应的变量名解析 time 维。
    """
    from src.preprocess.hydro_nc_stack import load_variable_map
    from src.preprocess.netcdf_io import open_netcdf_dataset

    if not nc_paths:
        return 0
    vm = load_variable_map()
    ch_map: dict[str, list[str]] = vm.get("channels", {})
    feat0 = feats[0]
    cands = ch_map.get(feat0, [feat0])
    total = 0
    for fp in sort_hydro_nc_paths([Path(p).resolve() for p in nc_paths]):
        ds, tmp_copy = open_netcdf_dataset(fp)
        try:
            da = None
            for name in cands:
                if name in ds:
                    da = ds[name]
                    break
            if da is None:
                raise KeyError(f"{fp.name}: 无变量匹配 {feat0}（候选 {cands}）")
            found = False
            for tn in ("time", "Time", "TIME", "t", "ocean_time", "dt_model"):
                if tn in da.sizes:
                    total += int(da.sizes[tn])
                    found = True
                    break
            if not found:
                total += int(da.shape[0])
        finally:
            ds.close()
            if tmp_copy is not None:
                try:
                    tmp_copy.unlink(missing_ok=True)  # type: ignore[arg-type]
                except OSError:
                    pass
    return total


def build_hydro_xy_from_netcdf_paths(
    nc_paths: list[Path],
    cfg: dict[str, Any],
    *,
    window_stride: int = 1,
    max_windows: int | None = 256,
    stats_npz_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    返回 X (N,T_in,H,W,C)、y (N,T_out,H,W,C)，已与训练管线一致做 Z-score。
    要求 cfg.data.input_features == target_features（四要素）。
    """
    d = cfg["data"]
    feats = list(d["input_features"])
    targets = list(d["target_features"])
    if feats != targets:
        raise ValueError("当前 NC 入口仅支持 input_features 与 target_features 完全一致（与 hydro_hycom 一致）。")
    tin = int(d["input_steps"])
    tout = int(d["output_steps"])
    gh_cfg, gw_cfg = int(d["grid_shape"][0]), int(d["grid_shape"][1])

    paths_sorted = sort_hydro_nc_paths([Path(p).resolve() for p in nc_paths])
    field, meta = stack_hydro_fields(paths_sorted, feats)
    th, tw = int(field.shape[1]), int(field.shape[2])
    if (th, tw) != (gh_cfg, gw_cfg):
        meta["grid_warning"] = (
            f"NC 网格为 {th}×{tw}，与配置 grid_shape [{gh_cfg}, {gw_cfg}] 不一致，ConvLSTM 推理可能报错。"
        )

    need = required_window_length(cfg)
    if field.shape[0] < need:
        raise ValueError(
            f"拼接后时间长度 T={field.shape[0]}，小于 input_steps+output_steps={need}。"
            f"请继续向缓冲区追加按时的日文件，或减小配置中的 input_steps/output_steps（需匹配已训练权重）。"
        )

    x, y = build_windows(field, tin, tout, stride=max(1, int(window_stride)))
    del field

    if max_windows is not None and x.shape[0] > int(max_windows):
        x = x[: int(max_windows)].copy()
        y = y[: int(max_windows)].copy()
        meta["windows_truncated_to"] = int(max_windows)

    sp = resolve_path(stats_npz_path) if stats_npz_path else resolve_path("data/processed/stats/hydro_zscore.npz")
    if sp.is_file():
        z = np.load(sp)
        mean, std = z["mean"], z["std"]
        apply_zscore(x, mean, std)
        apply_zscore(y, mean, std)
        meta["normalize"] = "hydro_zscore_stats"
        meta["stats_npz"] = str(sp)
    else:
        mean, std = zscore_fit(x)
        apply_zscore(x, mean, std)
        apply_zscore(y, mean, std)
        meta["normalize"] = "per_upload_fit"
        meta["normalize_note"] = (
            "未找到 data/processed/stats/hydro_zscore.npz，已用本次上传滑窗估计 mean/std，"
            "与离线训练全局统计不一致时 NRMSE 仅作演示参考。"
        )

    meta["n_windows"] = int(x.shape[0])
    meta["input_steps"] = tin
    meta["output_steps"] = tout
    return x, y, meta
