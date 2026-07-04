"""从命题方涡旋 NetCDF 生成 OW 伪标签 + YOLO-seg 训练集。

流水线（每 NC 文件 × 每个 time 索引）::

    adt / ugos / vgos 切片
        → okubo_weiss_and_vorticity → ζ, W
        → OW 阈值或多分位投票 → bool mask
        → 连通域轮廓 + ζ 符号 → YOLO-seg 多边形标签 (.txt)
        → build_input_rgb → 训练 PNG（与 input_mode 有关，与标签可解耦）

V6 约定
-------
- ``--input-mode {leakage,fair,triplet}``：仅影响 **images/** PNG，不影响 OW 伪标签逻辑。
- 归一化 **方案 A**：每个时间片、每个变量独立 P2/P98（禁止 triplet 三时刻合并分位）。
- Fair-B0 常用 ``--single-percentile 24``（OW P24）+ ``--input-mode fair``（ADT×3）。

入口::

    python -m src.preprocess.eddy_dataset --export-yolo -- ...
    python -m src.preprocess.eddy_yolo_export --out data/processed/eddy ...

详见 ``docs/架构与方法/涡旋_OW至YOLO伪标签开发参考.md``。
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from src.eddy.stacked_physics import (
    ablation_profile_channels,
    build_physics_stacked_ablation,
)
from src.preprocess.eddy_physics import (
    multi_percentile_vote_mask,
    okubo_weiss_and_vorticity,
    single_threshold_mask,
)
from src.utils.config import load_yaml, project_root, resolve_path
from src.utils.xarray_nc_open import open_xr_dataset_compat

# ---------------------------------------------------------------------------
# 命题方 NC 文件名 → train/val/test（与 服创数据集/数据集说明.md 一致）
# ---------------------------------------------------------------------------

_EDDY_TRAIN_STEMS = frozenset(
    {"19930101_20021231", "20030101_20121231", "20130101_20221231"}
)
_EDDY_TEST_STEM = "20230101_20231231"
_EDDY_VAL_STEM = "20240101_20241231"

INPUT_MODES = frozenset({"leakage", "fair", "triplet"})

_SPATIAL_DIM_NAMES = frozenset({"latitude", "longitude", "lat", "lon"})


def nc_path_to_split(nc_path: Path) -> str | None:
    """由 NC 文件名 stem 推断 Ultralytics 划分；未在表内则返回 None（跳过）。"""
    stem = nc_path.stem
    if stem in _EDDY_TRAIN_STEMS:
        return "train"
    if stem == _EDDY_TEST_STEM:
        return "test"
    if stem == _EDDY_VAL_STEM:
        return "val"
    return None


# ---------------------------------------------------------------------------
# NetCDF 读取辅助
# ---------------------------------------------------------------------------


def _pick_da(ds, names: tuple[str, ...]):
    """按候选别名（大小写不敏感）取 xarray DataArray。"""
    lower = {str(k).lower(): k for k in ds.data_vars}
    for n in names:
        if n.lower() in lower:
            return ds[lower[n.lower()]]
    raise KeyError(f"未找到变量（候选 {names}），实际: {list(ds.data_vars)}")


def _to_hw(arr: np.ndarray) -> np.ndarray:
    """单 time 切片应为 2D 空间场 (H, W)。"""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"期望 2D 空间场，得到 shape={a.shape}")
    return a


def _resolve_time_dim_name(adt_da) -> str:
    """从 ADT DataArray 的 dims 中找出非 lat/lon 的维度名（即 time）。"""
    tdim = [d for d in adt_da.dims if d not in _SPATIAL_DIM_NAMES]
    if not tdim:
        raise RuntimeError("未找到时间维度")
    return str(tdim[0])


# ---------------------------------------------------------------------------
# 训练图像：方案 A 归一化与 RGB 堆叠（input_mode）
# ---------------------------------------------------------------------------


def norm_field_scheme_a(field: np.ndarray, *, p_lo: float, p_hi: float) -> np.ndarray:
    """V6 方案 A：单时间片、单变量独立 P2/P98 → uint8 [0,255]。"""
    x = np.asarray(field, dtype=np.float64)
    xf = x[np.isfinite(x)]
    if xf.size == 0:
        return np.zeros_like(x, dtype=np.uint8)
    lo, hi = np.percentile(xf, (p_lo, p_hi))
    if hi <= lo:
        hi = lo + 1e-9
    y = np.clip((x - lo) / (hi - lo), 0, 1)
    y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
    return (y * 255).astype(np.uint8)


def _rgb_from_fields(
    adt: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    p_lo: float,
    p_hi: float,
) -> np.ndarray:
    """leakage 模式：R/G/B 分别为 ADT、U、V 各自方案 A 归一化（训练时可窥见流场）。"""
    out = np.zeros((adt.shape[0], adt.shape[1], 3), dtype=np.uint8)
    out[:, :, 0] = norm_field_scheme_a(adt, p_lo=p_lo, p_hi=p_hi)
    out[:, :, 1] = norm_field_scheme_a(u, p_lo=p_lo, p_hi=p_hi)
    out[:, :, 2] = norm_field_scheme_a(v, p_lo=p_lo, p_hi=p_hi)
    return out


def build_input_rgb(
    input_mode: str,
    adt: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    adt_prev: np.ndarray | None = None,
    adt_next: np.ndarray | None = None,
    p_lo: float,
    p_hi: float,
) -> np.ndarray:
    """
    构造与伪标签同帧配对的 RGB PNG（H×W×3 uint8）。

    - fair：norm(ADT) 复制三通道（论文 / 线上一致，Fair-B0）
    - leakage：ADT/U/V 各一通道
    - triplet：norm(ADT(t-k)), norm(ADT(t)), norm(ADT(t+k))
    """
    mode = str(input_mode).lower()
    if mode not in INPUT_MODES:
        raise ValueError(f"input_mode 须为 {sorted(INPUT_MODES)}")
    if mode == "leakage":
        return _rgb_from_fields(adt, u, v, p_lo=p_lo, p_hi=p_hi)
    if mode == "fair":
        ch = norm_field_scheme_a(adt, p_lo=p_lo, p_hi=p_hi)
        return np.stack([ch, ch, ch], axis=-1)
    if adt_prev is None or adt_next is None:
        raise ValueError("triplet 需要 adt_prev/adt_next")
    return np.stack(
        [
            norm_field_scheme_a(adt_prev, p_lo=p_lo, p_hi=p_hi),
            norm_field_scheme_a(adt, p_lo=p_lo, p_hi=p_hi),
            norm_field_scheme_a(adt_next, p_lo=p_lo, p_hi=p_hi),
        ],
        axis=-1,
    )


# ---------------------------------------------------------------------------
# 伪标签：OW mask → YOLO-seg 多边形
# ---------------------------------------------------------------------------


def _contours_to_yolo_lines(
    mask_bool: np.ndarray,
    zeta: np.ndarray,
    *,
    min_area_px: int,
    max_area_frac: float,
    approx_eps_frac: float,
    max_instances: int,
) -> list[tuple[int, list[float]]]:
    """
    将 OW 二值 mask 转为 YOLO-seg 标签行。

    返回 [(class_id, [x1,y1,x2,y2,...])]，坐标归一化到 [0,1]。
    类别由连通域内相对涡度 ζ 均值符号决定：ζ<0 → 0 气旋，否则 1 反气旋。
    """
    H, W = mask_bool.shape
    if H < 4 or W < 4:
        return []
    labeled, nlab = ndimage.label(mask_bool)
    if nlab == 0:
        return []

    max_area = float(max_area_frac * H * W)
    # 按面积从大到小处理，优先保留大涡旋
    comps: list[tuple[int, int]] = []
    for lab in range(1, nlab + 1):
        comps.append((int((labeled == lab).sum()), lab))
    comps.sort(key=lambda x: -x[0])

    lines: list[tuple[int, list[float]]] = []
    for area, lab in comps:
        if area < min_area_px or area > max_area:
            continue
        comp = (labeled == lab).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) < float(min_area_px):
            continue
        peri = cv2.arcLength(cnt, True)
        poly = cv2.approxPolyDP(cnt, float(approx_eps_frac) * peri, True)
        if poly.shape[0] < 3:
            continue
        pts = poly.reshape(-1, 2).astype(np.float64)
        pts[:, 0] = np.clip(pts[:, 0] / W, 0, 1)
        pts[:, 1] = np.clip(pts[:, 1] / H, 0, 1)
        flat = [coord for x, y in pts for coord in (float(x), float(y))]

        m = labeled == lab
        mz = float(np.nanmean(zeta[m])) if np.any(m) else 0.0
        # 北半球常见约定；若与命题方 META 定义冲突需再对齐
        cls = 0 if mz < 0 else 1
        lines.append((cls, flat))
        if len(lines) >= max_instances:
            break
    return lines


def _ow_eddy_mask(
    u: np.ndarray,
    v: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    single_percentile: float | None,
    vote_percentiles: tuple[float, ...],
    vote_min: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """由 u,v 计算 ζ、W，再得到涡旋候选 bool mask。返回 (zeta, ow, mask)。"""
    zeta, ow = okubo_weiss_and_vorticity(u, v, lat, lon)
    if single_percentile is not None:
        mask = single_threshold_mask(ow, float(single_percentile))
    else:
        mask = multi_percentile_vote_mask(ow, vote_percentiles, min_votes=int(vote_min))
    return zeta, ow, mask


def _write_yolo_label_file(lbl_path: Path, lines: list[tuple[int, list[float]]]) -> None:
    """写入 Ultralytics segmentation 格式：每行 ``cls x1 y1 x2 y2 ...``。"""
    with lbl_path.open("w", encoding="utf-8") as f:
        for cls, poly in lines:
            parts = [str(cls)] + [f"{x:.6f}" for x in poly]
            f.write(" ".join(parts) + "\n")


# ---------------------------------------------------------------------------
# dataset.yaml 与跨目录标签同步（Fair 图 + Leakage 标签）
# ---------------------------------------------------------------------------


def _write_dataset_yaml(
    eddy_root: Path,
    *,
    rel_path_posix: str | None = None,
    channels: int = 3,
    include_test: bool = True,
) -> None:
    """生成 Ultralytics 用 dataset.yaml（path 相对仓库根或 out 目录）。"""
    eddy_root.mkdir(parents=True, exist_ok=True)
    if rel_path_posix is None:
        try:
            rel_path_posix = eddy_root.resolve().relative_to(project_root().resolve()).as_posix()
        except ValueError:
            rel_path_posix = eddy_root.as_posix()
    ch_line = "" if int(channels) == 3 else f"channels: {int(channels)}\n"
    test_line = "test: images/test\n" if include_test else ""
    txt = f"""# 由 eddy_dataset --export-yolo 生成；path 相对于本文件所在目录
path: {rel_path_posix}
train: images/train
val: images/val
{test_line}{ch_line}names:
  0: eddy_cyclonic
  1: eddy_anticyclonic
"""
    (eddy_root / "dataset.yaml").write_text(txt, encoding="utf-8")


def _sync_labels_from(src_root: Path, dst_root: Path, split: str) -> int:
    """从另一导出目录复制 labels/{split}（Fair-B0：fair 图复用 leakage 伪标签）。"""
    n = 0
    src = src_root / "labels" / split
    dst = dst_root / "labels" / split
    dst.mkdir(parents=True, exist_ok=True)
    if not src.is_dir():
        return 0
    for lbl in src.glob("*.txt"):
        shutil.copy2(lbl, dst / lbl.name)
        n += 1
    return n


# ---------------------------------------------------------------------------
# 时间维过滤（日历窗口 / 边界 / triplet 邻帧 / B0 center_offset_max）
# ---------------------------------------------------------------------------


def _time_at_index(ds, tname: str, t_idx: int) -> datetime:
    import pandas as pd

    val = ds[tname].isel({tname: t_idx}).values
    return pd.Timestamp(val).to_pydatetime()


def _in_calendar_range(ts: datetime, start: str | None, end: str | None) -> bool:
    d = ts.date()
    if start is not None and d < datetime.strptime(start, "%Y-%m-%d").date():
        return False
    if end is not None and d > datetime.strptime(end, "%Y-%m-%d").date():
        return False
    return True


def _passes_center_offset_max(
    ds,
    tname: str,
    t_idx: int,
    T: int,
    *,
    center_offset_max: int,
    time_start: str | None,
    time_end: str | None,
) -> bool:
    """V6 B0：中心 t 当且仅当 t±k（k=1..center_offset_max）均在日历内且索引合法。"""
    kmax = int(center_offset_max)
    if kmax <= 0:
        return True
    for off in range(1, kmax + 1):
        for ni in (t_idx - off, t_idx + off):
            if ni < 0 or ni >= T:
                return False
            if time_start is not None or time_end is not None:
                ts_n = _time_at_index(ds, tname, ni)
                if not _in_calendar_range(ts_n, time_start, time_end):
                    return False
    return True


def _passes_triplet_offset(t_idx: int, T: int, triplet_offset: int) -> bool:
    k = int(triplet_offset)
    if k < 1:
        raise ValueError("triplet_offset 须 >= 1")
    return t_idx - k >= 0 and t_idx + k < T


def _should_export_time_index(
    ds,
    tname: str,
    t_idx: int,
    T: int,
    *,
    input_mode: str,
    time_start: str | None,
    time_end: str | None,
    skip_boundary_days: bool,
    center_offset_max: int,
    triplet_offset: int,
) -> bool:
    """合并所有 time 索引层面的跳过条件。"""
    ts = _time_at_index(ds, tname, t_idx)
    if not _in_calendar_range(ts, time_start, time_end):
        return False
    if skip_boundary_days and int(center_offset_max) <= 0 and (t_idx == 0 or t_idx >= T - 1):
        return False
    if not _passes_center_offset_max(
        ds,
        tname,
        t_idx,
        T,
        center_offset_max=center_offset_max,
        time_start=time_start,
        time_end=time_end,
    ):
        return False
    if input_mode == "triplet" and not _passes_triplet_offset(t_idx, T, triplet_offset):
        return False
    return True


# ---------------------------------------------------------------------------
# 单帧导出：伪标签 + PNG（+ 可选 npy 堆叠）
# ---------------------------------------------------------------------------


def _export_one_frame(
    *,
    ds,
    tname: str,
    t_idx: int,
    nc_stem: str,
    split: str,
    out_root: Path,
    input_mode: str,
    triplet_offset: int,
    rgb_percentiles: tuple[float, float],
    single_percentile: float | None,
    vote_percentiles: tuple[float, ...],
    vote_min: int,
    min_area_px: int,
    max_area_frac: float,
    approx_eps_frac: float,
    max_instances: int,
    lat: np.ndarray,
    lon: np.ndarray,
    adt_da,
    ug_da,
    vg_da,
    skip_labels: bool,
    labels_only: bool,
    stack_physics_npy: bool,
    stack_profile: str,
) -> None:
    """处理单个 time 索引：写 labels/*.txt，按需写 images/*.png 与 *.npy。"""
    a = _to_hw(adt_da.isel({tname: t_idx}).values)
    u = _to_hw(ug_da.isel({tname: t_idx}).values)
    v = _to_hw(vg_da.isel({tname: t_idx}).values)

    zeta, ow, mask = _ow_eddy_mask(
        u, v, lat, lon,
        single_percentile=single_percentile,
        vote_percentiles=vote_percentiles,
        vote_min=vote_min,
    )
    lines = _contours_to_yolo_lines(
        mask,
        zeta,
        min_area_px=min_area_px,
        max_area_frac=max_area_frac,
        approx_eps_frac=approx_eps_frac,
        max_instances=max_instances,
    )

    fname = f"{nc_stem}_t{t_idx:05d}"
    img_p = out_root / "images" / split / f"{fname}.png"
    lbl_p = out_root / "labels" / split / f"{fname}.txt"

    if not skip_labels:
        _write_yolo_label_file(lbl_p, lines)
    if labels_only:
        return

    p_lo, p_hi = float(rgb_percentiles[0]), float(rgb_percentiles[1])
    adt_prev = adt_next = None
    if input_mode == "triplet":
        k = int(triplet_offset)
        adt_prev = _to_hw(adt_da.isel({tname: t_idx - k}).values)
        adt_next = _to_hw(adt_da.isel({tname: t_idx + k}).values)

    rgb = build_input_rgb(
        input_mode,
        a,
        u,
        v,
        adt_prev=adt_prev,
        adt_next=adt_next,
        p_lo=p_lo,
        p_hi=p_hi,
    )
    Image.fromarray(rgb, mode="RGB").save(img_p)

    if stack_physics_npy:
        stack = build_physics_stacked_ablation(
            stack_profile, a, u, v, zeta, ow, p_lo=p_lo, p_hi=p_hi
        )
        np.save(str(img_p.with_suffix(".npy")), stack)


def _export_one_nc_file(
    nc: Path,
    *,
    split: str,
    out_root: Path,
    input_mode: str,
    time_stride: int,
    max_frames_per_file: int | None,
    time_start: str | None,
    time_end: str | None,
    skip_boundary_days: bool,
    center_offset_max: int,
    triplet_offset: int,
    rgb_percentiles: tuple[float, float],
    single_percentile: float | None,
    vote_percentiles: tuple[float, ...],
    vote_min: int,
    min_area_px: int,
    max_area_frac: float,
    approx_eps_frac: float,
    max_instances: int,
    skip_labels: bool,
    labels_only: bool,
    stack_physics_npy: bool,
    stack_profile: str,
) -> int:
    """打开单个 NC，按 stride 遍历 time 维并导出；返回本文件写入帧数。"""
    tmp_nc: Path | None = None
    ds = None
    n_written = 0
    try:
        ds, tmp_nc = open_xr_dataset_compat(nc)
        adt = _pick_da(ds, ("adt", "ADT"))
        ug = _pick_da(ds, ("ugos", "UGOS"))
        vg = _pick_da(ds, ("vgos", "VGOS"))
        lat = ds["latitude"].values if "latitude" in ds.coords else ds["lat"].values
        lon = ds["longitude"].values if "longitude" in ds.coords else ds["lon"].values
        tname = _resolve_time_dim_name(adt)
        T = int(adt.sizes[tname])

        for t_idx in range(0, T, max(1, int(time_stride))):
            if max_frames_per_file is not None and n_written >= max_frames_per_file:
                break
            if not _should_export_time_index(
                ds,
                tname,
                t_idx,
                T,
                input_mode=input_mode,
                time_start=time_start,
                time_end=time_end,
                skip_boundary_days=skip_boundary_days,
                center_offset_max=center_offset_max,
                triplet_offset=triplet_offset,
            ):
                continue

            _export_one_frame(
                ds=ds,
                tname=tname,
                t_idx=t_idx,
                nc_stem=nc.stem,
                split=split,
                out_root=out_root,
                input_mode=input_mode,
                triplet_offset=triplet_offset,
                rgb_percentiles=rgb_percentiles,
                single_percentile=single_percentile,
                vote_percentiles=vote_percentiles,
                vote_min=vote_min,
                min_area_px=min_area_px,
                max_area_frac=max_area_frac,
                approx_eps_frac=approx_eps_frac,
                max_instances=max_instances,
                lat=lat,
                lon=lon,
                adt_da=adt,
                ug_da=ug,
                vg_da=vg,
                skip_labels=skip_labels,
                labels_only=labels_only,
                stack_physics_npy=stack_physics_npy,
                stack_profile=stack_profile,
            )
            n_written += 1
    finally:
        if ds is not None:
            ds.close()
        if tmp_nc is not None:
            try:
                tmp_nc.unlink(missing_ok=True)
            except OSError:
                pass
    return n_written


# ---------------------------------------------------------------------------
# 主入口：遍历目录下全部 NC
# ---------------------------------------------------------------------------


def export_yolo_pseudo(
    *,
    data_config: Path,
    out_root: Path,
    time_stride: int,
    time_stride_val: int | None,
    max_frames_per_file: int | None,
    vote_percentiles: tuple[float, ...],
    vote_min: int,
    single_percentile: float | None,
    min_area_px: int,
    max_area_frac: float,
    approx_eps_frac: float,
    max_instances: int,
    rgb_percentiles: tuple[float, float],
    stack_physics_npy: bool = False,
    physics_channels: int | None = None,
    stack_profile: str = "8",
    input_mode: str = "leakage",
    nc_stem: str | None = None,
    split_filter: str | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
    skip_boundary_days: bool = False,
    triplet_offset: int = 1,
    center_offset_max: int = 0,
    labels_only: bool = False,
    skip_labels: bool = False,
    copy_labels_from: Path | None = None,
    include_test: bool = True,
) -> int:
    """
    批量导出 YOLO-seg 数据集到 ``out_root``。

    目录结构::

        out_root/
          dataset.yaml
          images/{train,val,test}/*.png
          labels/{train,val,test}/*.txt

    返回成功写入的帧数（含 labels_only 仅标签帧）。
    """
    cfg = load_yaml(data_config)
    raw = resolve_path(cfg.get("paths", {}).get("raw_root", "服创数据集"))
    sub = cfg.get("paths", {}).get("eddy_subdir", "中尺度涡识别")
    eddy_dir = raw / sub
    if not eddy_dir.is_dir():
        raise FileNotFoundError(f"涡旋目录不存在: {eddy_dir}")

    mode = str(input_mode).lower()
    if mode not in INPUT_MODES:
        raise ValueError(f"未知 input_mode: {input_mode}")

    out_root = resolve_path(out_root)
    split_names = ("train", "val", "test") if include_test else ("train", "val")
    for sp in split_names:
        (out_root / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / sp).mkdir(parents=True, exist_ok=True)

    if stack_physics_npy:
        profile = str(stack_profile).strip()
        mc_channels = ablation_profile_channels(profile)
        if physics_channels is not None and int(physics_channels) != mc_channels:
            raise ValueError(
                f"--physics-channels={physics_channels} 与 --stack-profile={profile} 不一致"
                f"（profile 对应 {mc_channels} 通道）"
            )
    else:
        mc_channels = 3
        profile = "3"
    _write_dataset_yaml(out_root, channels=mc_channels, include_test=include_test)

    if copy_labels_from is not None:
        src = resolve_path(copy_labels_from)
        for sp in split_names:
            _sync_labels_from(src, out_root, sp)

    nc_files = sorted(eddy_dir.glob("*.nc"))
    if nc_stem is not None:
        nc_files = [p for p in nc_files if p.stem == nc_stem]
        if not nc_files:
            raise FileNotFoundError(f"未找到 {nc_stem}.nc")

    n_written = 0
    for nc in nc_files:
        split = nc_path_to_split(nc)
        if split is None:
            print(f"跳过（未匹配命题方划分表）: {nc.name}")
            continue
        if split_filter is not None and split != split_filter:
            continue
        if not include_test and split == "test":
            continue

        stride = int(time_stride)
        if split == "val" and time_stride_val is not None:
            stride = max(1, int(time_stride_val))

        n_written += _export_one_nc_file(
            nc,
            split=split,
            out_root=out_root,
            input_mode=mode,
            time_stride=stride,
            max_frames_per_file=max_frames_per_file,
            time_start=time_start,
            time_end=time_end,
            skip_boundary_days=skip_boundary_days,
            center_offset_max=int(center_offset_max),
            triplet_offset=int(triplet_offset),
            rgb_percentiles=rgb_percentiles,
            single_percentile=single_percentile,
            vote_percentiles=vote_percentiles,
            vote_min=vote_min,
            min_area_px=min_area_px,
            max_area_frac=max_area_frac,
            approx_eps_frac=approx_eps_frac,
            max_instances=max_instances,
            skip_labels=skip_labels,
            labels_only=labels_only,
            stack_physics_npy=stack_physics_npy,
            stack_profile=profile,
        )

    print(f"导出完成: 共 {n_written} 帧 -> {out_root} (input_mode={mode})")
    print(f"dataset.yaml: {out_root / 'dataset.yaml'}")
    return n_written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OW 伪标签导出 YOLO-seg（V6 input-mode / P24）")
    p.add_argument("--data-config", type=str, default="config/data.yaml")
    p.add_argument("--out", type=str, default="data/processed/eddy")
    p.add_argument(
        "--input-mode",
        type=str,
        default="leakage",
        choices=sorted(INPUT_MODES),
        help="leakage|fair|triplet（方案 A norm；仅影响 PNG，不影响 OW 标签）",
    )
    p.add_argument("--nc-stem", type=str, default=None, help="只处理指定 stem 的 .nc")
    p.add_argument("--split", type=str, default=None, choices=("train", "val", "test"))
    p.add_argument("--time-start", type=str, default=None, help="日历过滤起始 YYYY-MM-DD")
    p.add_argument("--time-end", type=str, default=None, help="日历过滤结束 YYYY-MM-DD")
    p.add_argument("--skip-boundary-days", action="store_true")
    p.add_argument(
        "--triplet-offset",
        type=int,
        default=1,
        help="triplet 模式 ADT(t-k,t,t+k) 的 k；B0 扫 1/3/5",
    )
    p.add_argument(
        "--center-offset-max",
        type=int,
        default=0,
        help="要求 t±k（k=1..N）均在日历内；Fair-B0 常用 5",
    )
    p.add_argument("--labels-only", action="store_true", help="只写 labels，不写 PNG")
    p.add_argument("--skip-labels", action="store_true", help="只写 PNG，不写 labels")
    p.add_argument(
        "--copy-labels-from",
        type=str,
        default=None,
        help="从另一 out 目录复制 labels（fair 图 + leakage 标签）",
    )
    p.add_argument("--no-test-split", action="store_true")
    p.add_argument("--time-stride", type=int, default=15, help="train/test 时间步采样间隔")
    p.add_argument("--time-stride-val", type=int, default=None, help="val 专用 stride，默认同 train")
    p.add_argument("--max-frames-per-file", type=int, default=None)
    p.add_argument(
        "--vote-percentiles",
        type=str,
        default="12,18,24,30",
        help="多分位 OW 投票阈值列表（与 --single-percentile 二选一）",
    )
    p.add_argument("--vote-min", type=int, default=2, help="OW 投票最少命中分位数个数")
    p.add_argument(
        "--single-percentile",
        type=float,
        default=None,
        help="单分位 OW 阈值，如 24 即 P24（Fair-B0）",
    )
    p.add_argument("--min-area-px", type=int, default=80)
    p.add_argument("--max-area-frac", type=float, default=0.15)
    p.add_argument("--approx-eps-frac", type=float, default=0.002, help="Douglas-Peucker 相对周长系数")
    p.add_argument("--max-instances", type=int, default=40, help="单帧最多导出实例数")
    p.add_argument("--rgb-p-lo", type=float, default=2.0)
    p.add_argument("--rgb-p-hi", type=float, default=98.0)
    p.add_argument("--stack-physics-npy", action="store_true", help="额外保存多通道 npy（7/8ch 消融）")
    p.add_argument("--stack-profile", type=str, default="8")
    p.add_argument("--physics-channels", type=int, default=None)
    return p


def main_argv(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.labels_only and args.skip_labels:
        raise SystemExit("不能同时 --labels-only 与 --skip-labels")
    vps = tuple(float(x.strip()) for x in args.vote_percentiles.split(",") if x.strip())
    copy_from = Path(args.copy_labels_from) if args.copy_labels_from else None
    n = export_yolo_pseudo(
        data_config=project_root() / args.data_config,
        out_root=Path(args.out),
        time_stride=args.time_stride,
        time_stride_val=args.time_stride_val,
        max_frames_per_file=args.max_frames_per_file,
        vote_percentiles=vps,
        vote_min=args.vote_min,
        single_percentile=args.single_percentile,
        min_area_px=args.min_area_px,
        max_area_frac=args.max_area_frac,
        approx_eps_frac=args.approx_eps_frac,
        max_instances=args.max_instances,
        rgb_percentiles=(args.rgb_p_lo, args.rgb_p_hi),
        stack_physics_npy=bool(args.stack_physics_npy),
        physics_channels=args.physics_channels,
        stack_profile=str(args.stack_profile),
        input_mode=str(args.input_mode),
        nc_stem=args.nc_stem,
        split_filter=args.split,
        time_start=args.time_start,
        time_end=args.time_end,
        skip_boundary_days=bool(args.skip_boundary_days),
        triplet_offset=int(args.triplet_offset),
        center_offset_max=int(args.center_offset_max),
        labels_only=bool(args.labels_only),
        skip_labels=bool(args.skip_labels),
        copy_labels_from=copy_from,
        include_test=not bool(args.no_test_split),
    )
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main_argv())
