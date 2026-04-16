"""从命题方涡旋 NetCDF（adt/ugos/vgos）生成 OW 投票伪标签 + YOLO-seg 标签与三通道 PNG。"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from src.preprocess.eddy_physics import (
    multi_percentile_vote_mask,
    okubo_weiss_and_vorticity,
    single_threshold_mask,
)
from src.utils.config import load_yaml, project_root, resolve_path

# 与 `服创数据集/数据集说明.md` 文件名一致（stem，无 .nc）
_EDDY_TRAIN_STEMS = frozenset(
    {"19930101_20021231", "20030101_20121231", "20130101_20221231"}
)
_EDDY_TEST_STEM = "20230101_20231231"
_EDDY_VAL_STEM = "20240101_20241231"


def nc_path_to_split(nc_path: Path) -> str | None:
    stem = nc_path.stem
    if stem in _EDDY_TRAIN_STEMS:
        return "train"
    if stem == _EDDY_TEST_STEM:
        return "test"
    if stem == _EDDY_VAL_STEM:
        return "val"
    return None


def _pick_da(ds, names: tuple[str, ...]):
    import xarray as xr

    lower = {str(k).lower(): k for k in ds.data_vars}
    for n in names:
        if n.lower() in lower:
            return ds[lower[n.lower()]]
    raise KeyError(f"未找到变量（候选 {names}），实际: {list(ds.data_vars)}")


def _to_hw(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"期望 2D 空间场，得到 shape={a.shape}")
    return a


def _rgb_from_fields(
    adt: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    p_lo: float,
    p_hi: float,
) -> np.ndarray:
    """三通道 uint8：各通道在分位裁剪后线性拉伸。"""
    out = np.zeros((adt.shape[0], adt.shape[1], 3), dtype=np.uint8)

    def norm1(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        xf = x[np.isfinite(x)]
        if xf.size == 0:
            return np.zeros_like(x, dtype=np.uint8)
        lo, hi = np.percentile(xf, (p_lo, p_hi))
        if hi <= lo:
            hi = lo + 1e-9
        y = np.clip((x - lo) / (hi - lo), 0, 1)
        # NetCDF 中常见局部 NaN/Inf；统一压到 0，避免 cast 警告与脏像素传播。
        y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
        return (y * 255).astype(np.uint8)

    out[:, :, 0] = norm1(adt)
    out[:, :, 1] = norm1(u)
    out[:, :, 2] = norm1(v)
    return out


def _contours_to_yolo_lines(
    mask_bool: np.ndarray,
    zeta: np.ndarray,
    *,
    min_area_px: int,
    max_area_frac: float,
    approx_eps_frac: float,
    max_instances: int,
) -> list[tuple[int, list[float]]]:
    """返回 [(cls, [x1,y1,x2,y2,...] 归一化 0–1), ...]。"""
    H, W = mask_bool.shape
    if H < 4 or W < 4:
        return []
    labeled, nlab = ndimage.label(mask_bool)
    if nlab == 0:
        return []
    lines: list[tuple[int, list[float]]] = []
    max_area = float(max_area_frac * H * W)
    comps: list[tuple[int, int]] = []
    for lab in range(1, nlab + 1):
        a = int((labeled == lab).sum())
        comps.append((a, lab))
    comps.sort(key=lambda x: -x[0])

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
        eps = float(approx_eps_frac) * peri
        poly = cv2.approxPolyDP(cnt, eps, True)
        if poly.shape[0] < 3:
            continue
        pts = poly.reshape(-1, 2).astype(np.float64)
        pts[:, 0] /= W
        pts[:, 1] /= H
        pts[:, 0] = np.clip(pts[:, 0], 0, 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, 1)
        flat: list[float] = []
        for x, y in pts:
            flat.extend([float(x), float(y)])
        m = labeled == lab
        mz = float(np.nanmean(zeta[m])) if np.any(m) else 0.0
        # 北半球常见：气旋式相对涡度多为负、反气旋为正（与具体 META 定义以命题方为准）
        cls = 0 if mz < 0 else 1
        lines.append((cls, flat))
        if len(lines) >= max_instances:
            break
    return lines


def _write_dataset_yaml(eddy_root: Path) -> None:
    txt = """# 由 eddy_dataset --export-yolo 生成；path 相对于本文件所在目录
path: data/processed/eddy
train: images/train
val: images/val
names:
  0: eddy_cyclonic
  1: eddy_anticyclonic
"""
    eddy_root.mkdir(parents=True, exist_ok=True)
    (eddy_root / "dataset.yaml").write_text(txt, encoding="utf-8")


def export_yolo_pseudo(
    *,
    data_config: Path,
    out_root: Path,
    time_stride: int,
    max_frames_per_file: int | None,
    vote_percentiles: tuple[float, ...],
    vote_min: int,
    single_percentile: float | None,
    min_area_px: int,
    max_area_frac: float,
    approx_eps_frac: float,
    max_instances: int,
    rgb_percentiles: tuple[float, float],
) -> int:
    import xarray as xr

    cfg = load_yaml(data_config)
    raw = resolve_path(cfg.get("paths", {}).get("raw_root", "服创数据集"))
    sub = cfg.get("paths", {}).get("eddy_subdir", "中尺度涡识别")
    eddy_dir = raw / sub
    if not eddy_dir.is_dir():
        raise FileNotFoundError(f"涡旋目录不存在: {eddy_dir}")

    out_root = resolve_path(out_root)
    for sp in ("train", "val", "test"):
        (out_root / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / sp).mkdir(parents=True, exist_ok=True)

    _write_dataset_yaml(out_root)
    n_written = 0

    for nc in sorted(eddy_dir.glob("*.nc")):
        split = nc_path_to_split(nc)
        if split is None:
            print(f"跳过（未匹配命题方划分表）: {nc.name}")
            continue
        ds = xr.open_dataset(nc)
        try:
            adt = _pick_da(ds, ("adt", "ADT"))
            ug = _pick_da(ds, ("ugos", "UGOS"))
            vg = _pick_da(ds, ("vgos", "VGOS"))
            lat = ds["latitude"].values if "latitude" in ds.coords else ds["lat"].values
            lon = ds["longitude"].values if "longitude" in ds.coords else ds["lon"].values
            _sp = {"latitude", "longitude", "lat", "lon"}
            tdim = [d for d in adt.dims if d not in _sp]
            if not tdim:
                raise RuntimeError("未找到时间维度")
            tname = tdim[0]
            T = int(adt.sizes[tname])
            n_this = 0
            for t_idx in range(0, T, int(time_stride)):
                if max_frames_per_file is not None and n_this >= max_frames_per_file:
                    break
                a = _to_hw(adt.isel({tname: t_idx}).values)
                u = _to_hw(ug.isel({tname: t_idx}).values)
                v = _to_hw(vg.isel({tname: t_idx}).values)
                zeta, ow = okubo_weiss_and_vorticity(u, v, lat, lon)
                if single_percentile is not None:
                    mask = single_threshold_mask(ow, float(single_percentile))
                else:
                    mask = multi_percentile_vote_mask(
                        ow, vote_percentiles, min_votes=int(vote_min)
                    )
                lines = _contours_to_yolo_lines(
                    mask,
                    zeta,
                    min_area_px=int(min_area_px),
                    max_area_frac=float(max_area_frac),
                    approx_eps_frac=float(approx_eps_frac),
                    max_instances=int(max_instances),
                )
                rgb = _rgb_from_fields(a, u, v, p_lo=rgb_percentiles[0], p_hi=rgb_percentiles[1])
                stem = nc.stem
                fname = f"{stem}_t{t_idx:05d}"
                img_p = out_root / "images" / split / f"{fname}.png"
                lbl_p = out_root / "labels" / split / f"{fname}.txt"
                cv2.imwrite(str(img_p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                with lbl_p.open("w", encoding="utf-8") as f:
                    for cls, poly in lines:
                        parts = [str(cls)] + [f"{x:.6f}" for x in poly]
                        f.write(" ".join(parts) + "\n")
                n_written += 1
                n_this += 1
        finally:
            ds.close()

    print(f"导出完成: 共 {n_written} 帧 -> {out_root}")
    print(f"dataset.yaml: {out_root / 'dataset.yaml'}")
    return n_written


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OW 伪标签导出 YOLO-seg（adt/ugos/vgos）")
    p.add_argument("--data-config", type=str, default="config/data.yaml")
    p.add_argument(
        "--out",
        type=str,
        default="data/processed/eddy",
        help="输出根目录（其下 images/{train,val,test}/ 与 labels/）",
    )
    p.add_argument("--time-stride", type=int, default=30, help="时间维抽样步长（日数据建议 ≥7 控量）")
    p.add_argument("--max-frames-per-file", type=int, default=None, help="每个 nc 最多导出帧数（烟测用，如 5）")
    p.add_argument(
        "--vote-percentiles",
        type=str,
        default="12,18,24,30",
        help="多阈值投票用的 OW 分位数列表（逗号分隔）",
    )
    p.add_argument("--vote-min", type=int, default=2, help="至少几个分位掩膜重合为真")
    p.add_argument(
        "--single-percentile",
        type=float,
        default=None,
        help="若指定则不用投票，仅用单一 OW 低分位掩膜",
    )
    p.add_argument("--min-area-px", type=int, default=80, help="连通域最小像素面积")
    p.add_argument("--max-area-frac", type=float, default=0.15, help="单实例最大占全图面积比例")
    p.add_argument("--approx-eps-frac", type=float, default=0.002, help="Douglas–Peucker epsilon = 周长 * 该系数")
    p.add_argument("--max-instances", type=int, default=40, help="每帧最多实例数")
    p.add_argument("--rgb-p-lo", type=float, default=2.0)
    p.add_argument("--rgb-p-hi", type=float, default=98.0)
    return p


def main_argv(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    vps = tuple(float(x.strip()) for x in args.vote_percentiles.split(",") if x.strip())
    n = export_yolo_pseudo(
        data_config=project_root() / args.data_config,
        out_root=Path(args.out),
        time_stride=args.time_stride,
        max_frames_per_file=args.max_frames_per_file,
        vote_percentiles=vps,
        vote_min=args.vote_min,
        single_percentile=args.single_percentile,
        min_area_px=args.min_area_px,
        max_area_frac=args.max_area_frac,
        approx_eps_frac=args.approx_eps_frac,
        max_instances=args.max_instances,
        rgb_percentiles=(args.rgb_p_lo, args.rgb_p_hi),
    )
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main_argv())
