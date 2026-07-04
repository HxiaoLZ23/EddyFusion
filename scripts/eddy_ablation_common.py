"""涡旋消融实验共享：加载样本、推理、标签与 OW 掩膜。"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


@dataclass
class Instance:
    cls: int
    mask: np.ndarray
    conf: float = 1.0


@dataclass
class Sample:
    split: str
    png: Path
    npy: Path | None
    label: Path
    nc_stem: str
    time_idx: int
    rgb: np.ndarray
    adt: np.ndarray
    u: np.ndarray
    v: np.ndarray
    zeta: np.ndarray
    ow: np.ndarray
    grad_mag: np.ndarray
    gt: list[Instance]


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def parse_stem(stem: str) -> tuple[str, int]:
    m = re.match(r"^(.+)_t(\d+)$", stem)
    if not m:
        raise ValueError(f"无法解析 stem: {stem}")
    return m.group(1), int(m.group(2))


def read_label(path: Path, h: int, w: int) -> list[Instance]:
    out: list[Instance] = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cls = int(float(parts[0]))
        vals = [float(x) for x in parts[1:]]
        pts = np.asarray(vals, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= w
        pts[:, 1] *= h
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
        if int(mask.sum()) > 0:
            out.append(Instance(cls=cls, mask=mask.astype(bool)))
    return out


def load_fields(nc_root: Path, nc_stem: str, time_idx: int, cache: dict) -> tuple[np.ndarray, ...]:
    from src.preprocess.eddy_physics import okubo_weiss_and_vorticity
    from src.utils.xarray_nc_open import open_xr_dataset_compat

    key = nc_stem
    if key not in cache:
        nc = nc_root / f"{nc_stem}.nc"
        ds, tmp = open_xr_dataset_compat(nc)
        cache[key] = (ds, tmp, nc)
    ds, tmp, _ = cache[key]
    try:
        lower = {str(k).lower(): k for k in ds.data_vars}
        adt = ds[lower["adt"]]
        ug = ds[lower["ugos"]]
        vg = ds[lower["vgos"]]
        lat = ds["latitude"].values if "latitude" in ds.coords else ds["lat"].values
        lon = ds["longitude"].values if "longitude" in ds.coords else ds["lon"].values
        sp = {"latitude", "longitude", "lat", "lon"}
        tdim = [d for d in adt.dims if d not in sp][0]
        a = np.asarray(adt.isel({tdim: time_idx}).values, dtype=np.float64)
        u = np.asarray(ug.isel({tdim: time_idx}).values, dtype=np.float64)
        v = np.asarray(vg.isel({tdim: time_idx}).values, dtype=np.float64)
        zeta, ow = okubo_weiss_and_vorticity(u, v, lat, lon)
        gx = np.zeros_like(a)
        gy = np.zeros_like(a)
        gx[:, 1:-1] = (a[:, 2:] - a[:, :-2]) * 0.5
        gy[1:-1, :] = (a[2:, :] - a[:-2, :]) * 0.5
        grad_mag = np.sqrt(gx * gx + gy * gy + 1e-12)
        return a, u, v, zeta, ow, grad_mag
    finally:
        pass


def close_cache(cache: dict) -> None:
    for _k, (_ds, tmp, _nc) in list(cache.items()):
        try:
            _ds.close()
        except Exception:
            pass
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
    cache.clear()


def load_samples(
    dataset_root: Path,
    enh_root: Path | None,
    nc_root: Path,
    splits: list[str],
) -> list[Sample]:
    cache: dict = {}
    samples: list[Sample] = []
    try:
        for split in splits:
            for png in sorted((dataset_root / "images" / split).glob("*.png")):
                nc_stem, time_idx = parse_stem(png.stem)
                rgb = np.asarray(Image.open(png).convert("RGB"))
                h, w = rgb.shape[:2]
                label = dataset_root / "labels" / split / f"{png.stem}.txt"
                gt = read_label(label, h, w)
                adt, u, v, zeta, ow, grad_mag = load_fields(nc_root, nc_stem, time_idx, cache)
                npy = None
                if enh_root is not None:
                    p = enh_root / "images" / split / f"{png.stem}.npy"
                    npy = p if p.is_file() else None
                samples.append(
                    Sample(
                        split=split,
                        png=png,
                        npy=npy,
                        label=label,
                        nc_stem=nc_stem,
                        time_idx=time_idx,
                        rgb=rgb,
                        adt=adt,
                        u=u,
                        v=v,
                        zeta=zeta,
                        ow=ow,
                        grad_mag=grad_mag,
                        gt=gt,
                    )
                )
    finally:
        close_cache(cache)
    return samples


def predict_instances(model: Any, sample: Sample, *, use_npy: bool, conf: float) -> list[Instance]:
    if use_npy and sample.npy is not None:
        inp = np.load(sample.npy).astype(np.float32)
    else:
        inp = str(sample.png)
    res = model.predict(inp, conf=conf, verbose=False)[0]
    h, w = sample.rgb.shape[:2]
    instances: list[Instance] = []
    if res.masks is None or res.boxes is None:
        return instances
    classes = res.boxes.cls.cpu().numpy().astype(int) if res.boxes.cls is not None else np.zeros(0, dtype=int)
    confs = res.boxes.conf.cpu().numpy().astype(float) if res.boxes.conf is not None else np.ones(len(classes))
    for i, poly_list in enumerate(res.masks.xy):
        if i >= len(classes) or len(poly_list) < 3:
            continue
        pts = np.asarray(poly_list, dtype=np.float32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
        if int(mask.sum()) == 0:
            continue
        instances.append(Instance(cls=int(classes[i]), mask=mask.astype(bool), conf=float(confs[i])))
    return instances


def union_pred_mask(pred: list[Instance]) -> np.ndarray:
    if not pred:
        return np.zeros((1, 1), dtype=bool)
    h, w = pred[0].mask.shape
    out = np.zeros((h, w), dtype=bool)
    for p in pred:
        out |= p.mask
    return out


def vote_mask(ow: np.ndarray, percentiles: tuple[float, ...], vote_min: int) -> np.ndarray:
    from src.preprocess.eddy_physics import multi_percentile_vote_mask

    return multi_percentile_vote_mask(ow, percentiles, min_votes=vote_min)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def _resize_hw(field: np.ndarray, h: int, w: int) -> np.ndarray:
    """将物理场对齐到 PNG/标签分辨率。"""
    arr = np.asarray(field, dtype=np.float64)
    if arr.shape[0] == h and arr.shape[1] == w:
        return arr
    return cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)


def ow_rule_instances(
    ow: np.ndarray,
    zeta: np.ndarray,
    h: int,
    w: int,
    *,
    vote_percentiles: tuple[float, ...] = (12.0, 18.0, 24.0, 30.0),
    vote_min: int = 2,
    single_percentile: float | None = None,
    min_area_px: int = 80,
    max_area_frac: float = 0.15,
    approx_eps_frac: float = 0.002,
    max_instances: int = 40,
) -> list[Instance]:
    """推理期 OW 规则链（与 eddy_yolo_export 同参）。"""
    from src.preprocess.eddy_physics import single_threshold_mask
    from src.preprocess.eddy_yolo_export import _contours_to_yolo_lines

    ow_h = _resize_hw(ow, h, w)
    zeta_h = _resize_hw(zeta, h, w)
    if single_percentile is not None:
        mask = single_threshold_mask(ow_h, float(single_percentile))
    else:
        mask = vote_mask(ow_h, vote_percentiles, vote_min)
    lines = _contours_to_yolo_lines(
        mask,
        zeta_h,
        min_area_px=min_area_px,
        max_area_frac=max_area_frac,
        approx_eps_frac=approx_eps_frac,
        max_instances=max_instances,
    )
    out: list[Instance] = []
    for cls, poly in lines:
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= w
        pts[:, 1] *= h
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
        if int(m.sum()) > 0:
            out.append(Instance(cls=int(cls), mask=m.astype(bool), conf=1.0))
    return out


def instance_area_px(inst: Instance) -> int:
    return int(inst.mask.sum())


def boundary_roughness(instances: list[Instance]) -> float:
    """边界粗糙度：周长/√面积（越大越锯齿；圆≈3.54）。"""
    if not instances:
        return 0.0
    vals: list[float] = []
    for inst in instances:
        m = (inst.mask.astype(np.uint8) * 255)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        peri = float(cv2.arcLength(cnt, True))
        if area > 1.0:
            vals.append(peri / np.sqrt(area))
    return float(np.mean(vals)) if vals else 0.0


def boundary_grad_on_mask(instances: list[Instance], grad_mag: np.ndarray) -> float:
    """边界处 ADT 梯度幅值均值（越小通常表示边界更贴等值线）。"""
    g = _resize_hw(grad_mag, instances[0].mask.shape[0], instances[0].mask.shape[1]) if instances else grad_mag
    vals: list[float] = []
    for inst in instances:
        m = inst.mask.astype(np.uint8)
        edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        ys, xs = np.where(edge > 0)
        if ys.size == 0:
            continue
        vals.append(float(np.nanmean(g[ys, xs])))
    return float(np.mean(vals)) if vals else 0.0


def merge_error_count(gt: list[Instance], pred: list[Instance], *, iou_thr: float = 0.25) -> int:
    """一个预测实例与 ≥2 个 GT 重叠 → 粘连/欠分割。"""
    n = 0
    for p in pred:
        hits = 0
        for g in gt:
            if mask_iou(p.mask, g.mask) >= iou_thr:
                hits += 1
        if hits >= 2:
            n += 1
    return n


def split_error_count(gt: list[Instance], pred: list[Instance], *, iou_thr: float = 0.25) -> int:
    """一个 GT 与 ≥2 个预测重叠 → 过分割。"""
    n = 0
    for g in gt:
        hits = 0
        for p in pred:
            if mask_iou(p.mask, g.mask) >= iou_thr:
                hits += 1
        if hits >= 2:
            n += 1
    return n


def small_instance_recall(
    gt: list[Instance],
    pred: list[Instance],
    *,
    area_percentile: float = 33.0,
    iou_thr: float = 0.5,
) -> tuple[float, int]:
    """小涡旋（面积低于 GT 分位数）召回。"""
    if not gt:
        return 0.0, 0
    areas = sorted(instance_area_px(g) for g in gt)
    idx = max(0, min(len(areas) - 1, int(round(len(areas) * area_percentile / 100.0)) - 1))
    thr = areas[idx] if areas else 0
    small = [g for g in gt if instance_area_px(g) <= thr]
    if not small:
        return 0.0, 0
    tp = 0
    used_p: set[int] = set()
    for g in small:
        best = 0.0
        best_pi = -1
        for pi, p in enumerate(pred):
            if pi in used_p or p.cls != g.cls:
                continue
            iou = mask_iou(p.mask, g.mask)
            if iou > best:
                best = iou
                best_pi = pi
        if best >= iou_thr and best_pi >= 0:
            tp += 1
            used_p.add(best_pi)
    return tp / len(small), len(small)


def aggregate_match(rows: list[dict[str, float]]) -> dict[str, float]:
    tp = sum(r.get("tp", 0.0) for r in rows)
    fp = sum(r.get("fp", 0.0) for r in rows)
    fn = sum(r.get("fn", 0.0) for r in rows)
    ious = [r["mean_iou"] for r in rows if r.get("mean_iou", 0) > 0]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou_matched": float(np.mean(ious)) if ious else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_images": float(len(rows)),
    }


def match_metrics(gt: list[Instance], pred: list[Instance], *, iou_thr: float = 0.5) -> dict[str, float]:
    pairs: list[tuple[float, int, int]] = []
    for pi, p in enumerate(pred):
        for gi, g in enumerate(gt):
            if p.cls != g.cls:
                continue
            pairs.append((mask_iou(p.mask, g.mask), pi, gi))
    pairs.sort(reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    ious: list[float] = []
    for iou, pi, gi in pairs:
        if iou < iou_thr or pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        ious.append(iou)
    tp = len(ious)
    fp = max(0, len(pred) - tp)
    fn = max(0, len(gt) - tp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "n_pred": float(len(pred)),
        "n_gt": float(len(gt)),
    }
