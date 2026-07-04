"""双路 MP4：单次打开 NC、按 time 批量切片；纯 numpy 后处理可进多进程（不传 xarray）。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.eddy.nc_to_bgr import (
    _isel_time,
    _norm_to_u8,
    _pick_dataarray,
    _spatial_only_vals,
    _time_dim_name,
    _time_label_for_index,
    fair_adt_triple_bgr_u8,
)
from src.eddy.stacked_physics import (
    build_physics_stacked_hw7,
    build_physics_stacked_hw8,
    relative_vorticity_and_okubo_weiss_from_uv,
)
from src.preprocess.netcdf_io import open_netcdf_dataset

_TRIPLE_CANDIDATES: list[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], str]] = [
    (("adt", "ADT"), ("ugos", "UGOS"), ("vgos", "VGOS"), "adt_ugos_vgos"),
    (("sla", "SLA", "ssh", "SSH"), ("ugos", "UGOS", "ssu", "SSU"), ("vgos", "VGOS", "ssv", "SSV"), "sla_uv"),
    (("sst", "SST"), ("ssu", "SSU"), ("ssv", "SSV"), "sst_ssu_ssv"),
]


def _resolve_triple_on_open_dataset(ds: Any) -> tuple[Any, Any, Any, str | None, str]:
    last_err: Exception | None = None
    for n0, n1, n2, tag in _TRIPLE_CANDIDATES:
        try:
            ch0 = _pick_dataarray(ds, n0)
            ch1 = _pick_dataarray(ds, n1)
            ch2 = _pick_dataarray(ds, n2)
        except KeyError as e:
            last_err = e
            continue
        tname = _time_dim_name(ch0)
        return ch0, ch1, ch2, tname, tag
    raise ValueError(
        f"无法从 NC 匹配 ADT/流场或 SST/流场变量组合。最后错误: {last_err!r}；"
        f"data_vars={list(ds.data_vars)}"
    ) from last_err


def probe_netcdf_time_meta(nc_path: str | Path) -> dict[str, Any]:
    """仅探测时间维长度与通道组合（打开一次 NC）。"""
    path = Path(nc_path).expanduser().resolve()
    ds, tmp_copy = open_netcdf_dataset(path)
    meta: dict[str, Any] = {"nc_path": str(path)}
    if tmp_copy is not None:
        meta["opened_via_tmp"] = str(tmp_copy)
    try:
        ch0, _ch1, _ch2, tname, tag = _resolve_triple_on_open_dataset(ds)
        meta["channel_triple"] = tag
        if tname is not None:
            meta["time_dim"] = tname
            meta["time_len"] = int(ch0.sizes[tname])
        else:
            meta["time_dim"] = None
            meta["time_len"] = 1
        return meta
    finally:
        ds.close()
        if tmp_copy is not None:
            try:
                tmp_copy.unlink(missing_ok=True)  # type: ignore[arg-type]
            except OSError:
                pass


def extract_triple_slices_batch(
    nc_path: str | Path,
    indices: list[int],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]]:
    """
    打开 NC 一次，对多个 time_index 批量 isel，返回 [(adt,u,v,meta), ...]。
    meta 为每帧字段；文件级字段在首帧 meta 中重复携带。
    """
    if not indices:
        return []
    path = Path(nc_path).expanduser().resolve()
    ds, tmp_copy = open_netcdf_dataset(path)
    file_meta: dict[str, Any] = {"nc_path": str(path)}
    if tmp_copy is not None:
        file_meta["opened_via_tmp"] = str(tmp_copy)

    out: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = []
    try:
        ch0, ch1, ch2, tname, tag = _resolve_triple_on_open_dataset(ds)
        file_meta["channel_triple"] = tag

        if tname is None:
            a0 = _spatial_only_vals(ch0)
            u0 = _spatial_only_vals(ch1)
            v0 = _spatial_only_vals(ch2)
            if a0.shape != u0.shape or a0.shape != v0.shape:
                raise ValueError(f"三通道形状不一致 {a0.shape} {u0.shape} {v0.shape}")
            frame_meta = {**file_meta, "time_dim": None, "time_len": 1}
            for ti in indices:
                m = {**frame_meta, "time_index": int(ti), "time_label": f"步 {ti}"}
                out.append((a0.astype(np.float64), u0.astype(np.float64), v0.astype(np.float64), m))
            return out

        T = int(ch0.sizes[tname])
        file_meta["time_dim"] = tname
        file_meta["time_len"] = T
        clamped = [max(0, min(int(ti), T - 1)) for ti in indices]

        try:
            da0 = ch0.isel({tname: clamped})
            da1 = ch1.isel({tname: clamped})
            da2 = ch2.isel({tname: clamped})
            stack0 = np.asarray(da0.values, dtype=np.float64)
            stack1 = np.asarray(da1.values, dtype=np.float64)
            stack2 = np.asarray(da2.values, dtype=np.float64)
            if stack0.ndim == 2:
                stack0 = stack0[np.newaxis, ...]
                stack1 = stack1[np.newaxis, ...]
                stack2 = stack2[np.newaxis, ...]
        except Exception:
            stack0 = np.stack([_isel_time(ch0, tname, ti) for ti in clamped], axis=0)
            stack1 = np.stack([_isel_time(ch1, tname, ti) for ti in clamped], axis=0)
            stack2 = np.stack([_isel_time(ch2, tname, ti) for ti in clamped], axis=0)

        n = int(stack0.shape[0])
        for i in range(n):
            a0 = np.asarray(stack0[i], dtype=np.float64)
            u0 = np.asarray(stack1[i], dtype=np.float64)
            v0 = np.asarray(stack2[i], dtype=np.float64)
            if a0.shape != u0.shape or a0.shape != v0.shape:
                raise ValueError(f"三通道形状不一致 {a0.shape} {u0.shape} {v0.shape}")
            ti = clamped[i]
            m = {**file_meta, "time_index": ti}
            tl = _time_label_for_index(ch0, tname, ti)
            m["time_label"] = tl or f"步 {ti}"
            out.append((a0, u0, v0, m))
        return out
    finally:
        ds.close()
        if tmp_copy is not None:
            try:
                tmp_copy.unlink(missing_ok=True)  # type: ignore[arg-type]
            except OSError:
                pass


def triple_to_bgr_u8(a0: np.ndarray, u0: np.ndarray, v0: np.ndarray) -> np.ndarray:
    rgb = np.stack([_norm_to_u8(a0), _norm_to_u8(u0), _norm_to_u8(v0)], axis=-1)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def build_dual_frame_from_triple(
    a0: np.ndarray,
    u0: np.ndarray,
    v0: np.ndarray,
    meta: dict[str, Any],
    *,
    need_8ch: bool = False,
    physics_stack_channels: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    """由浮点三通道生成 (底图 BGR, YOLO 输入, plot 可选, meta)。无 xarray / 无文件 I/O。

    ``physics_stack_channels``：3 / 7 / 8；若为 None 则 ``need_8ch=True``→8，否则→3（兼容旧调用）。
    """
    ch = int(physics_stack_channels) if physics_stack_channels is not None else (8 if need_8ch else 3)
    out_meta = dict(meta)
    if ch == 3:
        bgr_vis = fair_adt_triple_bgr_u8(a0)
        out_meta["inference_stack"] = "fair_adt_x3"
    else:
        bgr_vis = triple_to_bgr_u8(a0, u0, v0)

    if ch == 8:
        zeta, ow = relative_vorticity_and_okubo_weiss_from_uv(u0, v0)
        hw8 = build_physics_stacked_hw8(a0, u0, v0, zeta, ow)
        yolo_in = np.clip(np.asarray(hw8, dtype=np.float64) * 255.0, 0.0, 255.0).astype(np.uint8)
        base = np.asarray(bgr_vis, dtype=np.uint8)
        plot_bgr: np.ndarray | None = base
        out_meta["inference_input_channels"] = 8
        out_meta["inference_stack"] = "physics_hw8_from_nc"
    elif ch == 7:
        zeta, ow = relative_vorticity_and_okubo_weiss_from_uv(u0, v0)
        hw7 = build_physics_stacked_hw7(a0, u0, v0, zeta, ow)
        yolo_in = np.clip(np.asarray(hw7, dtype=np.float64) * 255.0, 0.0, 255.0).astype(np.uint8)
        base = np.asarray(bgr_vis, dtype=np.uint8)
        plot_bgr = base
        out_meta["inference_input_channels"] = 7
        out_meta["inference_stack"] = "physics_hw7_from_nc"
    else:
        yolo_in = np.asarray(bgr_vis, dtype=np.uint8)
        base = yolo_in.copy()
        plot_bgr = None
        out_meta["inference_input_channels"] = 3
        out_meta.setdefault("inference_stack", "fair_adt_x3")
    out_meta["time_label"] = out_meta.get("time_label") or f"步 {out_meta.get('time_index', 0)}"
    return base, yolo_in, plot_bgr, out_meta


def dual_extract_worker_count(n_frames: int) -> int:
    """EDDY_DUAL_EXTRACT_WORKERS：0=自动，1=禁用多进程。"""
    if n_frames <= 1:
        return 1
    raw = os.environ.get("EDDY_DUAL_EXTRACT_WORKERS", "0").strip()
    try:
        w = int(raw)
    except ValueError:
        w = 0
    if w == 1:
        return 1
    if w > 1:
        return min(w, n_frames)
    cpu = os.cpu_count() or 4
    return max(1, min(4, cpu - 1, n_frames))


def _mp_build_dual_payload(
    payload: tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    a0, u0, v0, meta, stack_ch = payload
    return build_dual_frame_from_triple(a0, u0, v0, meta, physics_stack_channels=int(stack_ch))


def build_dual_frames_parallel(
    slices: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]],
    *,
    need_8ch: bool = False,
    physics_stack_channels: int | None = None,
    workers: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]]:
    n = len(slices)
    if n == 0:
        return []
    w = workers if workers is not None else dual_extract_worker_count(n)
    stack_ch = int(physics_stack_channels) if physics_stack_channels is not None else (8 if need_8ch else 3)
    payloads = [(a0, u0, v0, meta, stack_ch) for a0, u0, v0, meta in slices]
    if w <= 1:
        return [_mp_build_dual_payload(p) for p in payloads]

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=w) as pool:
        return list(pool.map(_mp_build_dual_payload, payloads, chunksize=max(1, n // (w * 4))))
