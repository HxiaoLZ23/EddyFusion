#!/usr/bin/env python3
"""无人工真值条件下的涡旋 3ch vs 7ch 分场景与物理一致性分析。

本脚本不替代 ``src.eddy.eval`` 的标准 mAP 口径；它面向论文/答辩补充分析：

- 用伪标签作为弱参考，统计分场景 precision / recall / FP per image / mean IoU；
- 用 ADT/UGOS/VGOS 派生物理场，统计预测 mask 的 OW、涡度符号、边界梯度一致性；
- 自动挑选若干 7ch 相对 3ch 更有解释力的案例，生成对比图。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
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


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO / p)


def _parse_stem(stem: str) -> tuple[str, int]:
    m = re.match(r"^(.+)_t(\d+)$", stem)
    if not m:
        raise ValueError(f"无法解析样本 stem: {stem}")
    return m.group(1), int(m.group(2))


def _pick_da(ds: Any, names: tuple[str, ...]) -> Any:
    lower = {str(k).lower(): k for k in ds.data_vars}
    for name in names:
        if name.lower() in lower:
            return ds[lower[name.lower()]]
    raise KeyError(names)


def _load_fields(nc_root: Path, nc_stem: str, time_idx: int, cache: dict[str, tuple[Any, Any]]) -> tuple[np.ndarray, ...]:
    from src.preprocess.eddy_physics import okubo_weiss_and_vorticity
    from src.utils.xarray_nc_open import open_xr_dataset_compat

    if nc_stem not in cache:
        ds, tmp = open_xr_dataset_compat(nc_root / f"{nc_stem}.nc")
        cache[nc_stem] = (ds, tmp)
    ds, _tmp = cache[nc_stem]

    adt = _pick_da(ds, ("adt", "ADT"))
    ug = _pick_da(ds, ("ugos", "UGOS"))
    vg = _pick_da(ds, ("vgos", "VGOS"))
    spatial = {"latitude", "longitude", "lat", "lon"}
    tdim = [d for d in adt.dims if d not in spatial][0]
    a = np.asarray(adt.isel({tdim: time_idx}).values, dtype=np.float64)
    u = np.asarray(ug.isel({tdim: time_idx}).values, dtype=np.float64)
    v = np.asarray(vg.isel({tdim: time_idx}).values, dtype=np.float64)
    lat = ds["latitude"].values if "latitude" in ds.coords else ds["lat"].values
    lon = ds["longitude"].values if "longitude" in ds.coords else ds["lon"].values
    zeta, ow = okubo_weiss_and_vorticity(u, v, lat, lon)
    gx, gy = np.gradient(np.nan_to_num(a))
    grad_mag = np.sqrt(gx * gx + gy * gy + 1e-12)
    return a, u, v, zeta, ow, grad_mag


def _close_cache(cache: dict[str, tuple[Any, Any]]) -> None:
    for ds, tmp in cache.values():
        ds.close()
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


def _read_label(path: Path, h: int, w: int) -> list[Instance]:
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


def _load_samples(dataset_root: Path, enh_root: Path, nc_root: Path, splits: list[str]) -> list[Sample]:
    cache: dict[str, tuple[Any, Any]] = {}
    samples: list[Sample] = []
    try:
        for split in splits:
            for png in sorted((dataset_root / "images" / split).glob("*.png")):
                nc_stem, time_idx = _parse_stem(png.stem)
                rgb = np.asarray(Image.open(png).convert("RGB"))
                h, w = rgb.shape[:2]
                label = dataset_root / "labels" / split / f"{png.stem}.txt"
                gt = _read_label(label, h, w)
                adt, u, v, zeta, ow, grad_mag = _load_fields(nc_root, nc_stem, time_idx, cache)
                npy = enh_root / "images" / split / f"{png.stem}.npy"
                samples.append(
                    Sample(
                        split=split,
                        png=png,
                        npy=npy if npy.is_file() else None,
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
        _close_cache(cache)
    return samples


def _predict_instances(model: Any, sample: Sample, *, use_npy: bool, conf: float) -> list[Instance]:
    inp: str | np.ndarray
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


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def _match_counts(gt: list[Instance], pred: list[Instance], *, iou_thr: float = 0.5) -> dict[str, float]:
    pairs: list[tuple[float, int, int]] = []
    for pi, p in enumerate(pred):
        for gi, g in enumerate(gt):
            if p.cls != g.cls:
                continue
            pairs.append((_iou(p.mask, g.mask), pi, gi))
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
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "matched_mean_iou": float(np.mean(ious)) if ious else 0.0,
    }


def _quantile_bucket(v: float, lo: float, hi: float, low_name: str, mid_name: str, high_name: str) -> str:
    if v <= lo:
        return low_name
    if v >= hi:
        return high_name
    return mid_name


def _image_features(samples: list[Sample]) -> dict[str, dict[str, float | str]]:
    raw: dict[str, dict[str, float]] = {}
    for s in samples:
        gray = np.asarray(Image.fromarray(s.rgb).convert("L"), dtype=np.float32) / 255.0
        speed = np.sqrt(s.u * s.u + s.v * s.v)
        areas = [float(g.mask.mean()) for g in s.gt]
        raw[s.png.stem] = {
            "rgb_std": float(np.nanstd(gray)),
            "speed_p90": float(np.nanpercentile(speed, 90)),
            "grad_p90": float(np.nanpercentile(s.grad_mag, 90)),
            "instance_count": float(len(s.gt)),
            "median_area": float(np.median(areas)) if areas else 0.0,
        }
    keys = ["rgb_std", "speed_p90", "grad_p90", "instance_count", "median_area"]
    qs = {
        k: (
            float(np.quantile([v[k] for v in raw.values()], 0.33)),
            float(np.quantile([v[k] for v in raw.values()], 0.67)),
        )
        for k in keys
    }
    out: dict[str, dict[str, float | str]] = {}
    for stem, feat in raw.items():
        lo_rgb, hi_rgb = qs["rgb_std"]
        lo_speed, hi_speed = qs["speed_p90"]
        lo_grad, hi_grad = qs["grad_p90"]
        lo_cnt, hi_cnt = qs["instance_count"]
        lo_area, hi_area = qs["median_area"]
        out[stem] = dict(feat)
        out[stem].update(
            {
                "contrast_group": _quantile_bucket(feat["rgb_std"], lo_rgb, hi_rgb, "low_contrast", "mid_contrast", "high_contrast"),
                "flow_group": _quantile_bucket(feat["speed_p90"], lo_speed, hi_speed, "weak_flow", "mid_flow", "strong_flow"),
                "grad_group": _quantile_bucket(feat["grad_p90"], lo_grad, hi_grad, "smooth_grad", "mid_grad", "complex_grad"),
                "density_group": _quantile_bucket(feat["instance_count"], lo_cnt, hi_cnt, "few_eddies", "mid_eddies", "many_eddies"),
                "size_group": _quantile_bucket(feat["median_area"], lo_area, hi_area, "small_targets", "mid_targets", "large_targets"),
            }
        )
    return out


def _physical_scores(sample: Sample, pred: list[Instance]) -> dict[str, float]:
    if not pred:
        return {
            "ow_low_frac": 0.0,
            "zeta_sign_consistency": 0.0,
            "boundary_grad_mean": 0.0,
            "n_pred": 0.0,
        }
    ow_thr = float(np.nanpercentile(sample.ow, 20))
    scores_ow: list[float] = []
    scores_zeta: list[float] = []
    scores_grad: list[float] = []
    kernel = np.ones((3, 3), dtype=np.uint8)
    for inst in pred:
        m = inst.mask
        if not np.any(m):
            continue
        scores_ow.append(float(np.nanmean(sample.ow[m] <= ow_thr)))
        z = sample.zeta[m]
        if inst.cls == 0:
            scores_zeta.append(float(np.nanmean(z < 0)))
        else:
            scores_zeta.append(float(np.nanmean(z >= 0)))
        eroded = cv2.erode(m.astype(np.uint8), kernel, iterations=1).astype(bool)
        boundary = np.logical_and(m, np.logical_not(eroded))
        scores_grad.append(float(np.nanmean(sample.grad_mag[boundary])) if np.any(boundary) else 0.0)
    return {
        "ow_low_frac": float(np.mean(scores_ow)) if scores_ow else 0.0,
        "zeta_sign_consistency": float(np.mean(scores_zeta)) if scores_zeta else 0.0,
        "boundary_grad_mean": float(np.mean(scores_grad)) if scores_grad else 0.0,
        "n_pred": float(len(pred)),
    }


def _aggregate(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    groups = sorted({str(r[group_key]) for r in rows})
    out: list[dict[str, Any]] = []
    for group in groups:
        for model in ("3ch", "7ch"):
            sub = [r for r in rows if str(r[group_key]) == group and r["model"] == model]
            if not sub:
                continue
            tp = sum(float(r["tp"]) for r in sub)
            fp = sum(float(r["fp"]) for r in sub)
            fn = sum(float(r["fn"]) for r in sub)
            out.append(
                {
                    "group_key": group_key,
                    "group": group,
                    "model": model,
                    "n_images": len(sub),
                    "precision": tp / (tp + fp) if tp + fp else 0.0,
                    "recall": tp / (tp + fn) if tp + fn else 0.0,
                    "fp_per_image": fp / len(sub),
                    # mean_iou 保留漏检惩罚：TP=0 的图按 0 计；matched_mean_iou 只看已匹配实例的形状质量。
                    "mean_iou": float(np.mean([float(r["mean_iou"]) for r in sub])),
                    "matched_mean_iou": float(
                        np.mean([float(r["matched_mean_iou"]) for r in sub if float(r["tp"]) > 0])
                    )
                    if any(float(r["tp"]) > 0 for r in sub)
                    else 0.0,
                    "ow_low_frac": float(np.mean([float(r["ow_low_frac"]) for r in sub])),
                    "zeta_sign_consistency": float(np.mean([float(r["zeta_sign_consistency"]) for r in sub])),
                    "boundary_grad_mean": float(np.mean([float(r["boundary_grad_mean"]) for r in sub])),
                }
            )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: list[dict[str, Any]], *, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(f"# {title}\n\n无数据。\n", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    lines = [f"# {title}", "", "| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        vals: list[str] = []
        for c in cols:
            v = r[c]
            vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metric(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _plot_outputs(
    *,
    out: Path,
    agg_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    samples_by_stem: dict[str, Sample],
    pred_cache: dict[tuple[str, str], list[Instance]],
) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.35], hspace=0.35, wspace=0.28)

    def _bar(ax, group_key: str, metric: str, title: str) -> None:
        rows = [r for r in agg_rows if r["group_key"] == group_key]
        groups = sorted({r["group"] for r in rows})
        x = np.arange(len(groups))
        w = 0.36
        for offset, model, color in [(-w / 2, "3ch", "#5B8FD8"), (w / 2, "7ch", "#D97B4A")]:
            vals = []
            for g in groups:
                hit = next((r for r in rows if r["group"] == g and r["model"] == model), None)
                vals.append(float(hit[metric]) if hit else 0.0)
            ax.bar(x + offset, vals, w, label=model, color=color)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1.02 if metric != "fp_per_image" else None)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)

    _bar(fig.add_subplot(gs[0, 0]), "contrast_group", "recall", "按对比度分组 Recall")
    _bar(fig.add_subplot(gs[0, 1]), "flow_group", "fp_per_image", "按背景流分组 FP/image")
    _bar(fig.add_subplot(gs[0, 2]), "size_group", "matched_mean_iou", "按目标大小分组 matched IoU")
    _bar(fig.add_subplot(gs[1, 0]), "grad_group", "ow_low_frac", "OW 低值一致性")
    _bar(fig.add_subplot(gs[1, 1]), "density_group", "zeta_sign_consistency", "涡度符号一致性")
    _bar(fig.add_subplot(gs[1, 2]), "flow_group", "boundary_grad_mean", "预测边界 grad(ADT)")

    case_candidates: list[tuple[float, str]] = []
    by_stem: dict[str, dict[str, dict[str, Any]]] = {}
    for r in sample_rows:
        by_stem.setdefault(str(r["stem"]), {})[str(r["model"])] = r
    for stem, pair in by_stem.items():
        if "3ch" not in pair or "7ch" not in pair:
            continue
        score = (
            _metric(pair["7ch"], "zeta_sign_consistency") - _metric(pair["3ch"], "zeta_sign_consistency")
            + _metric(pair["7ch"], "ow_low_frac") - _metric(pair["3ch"], "ow_low_frac")
            + _metric(pair["3ch"], "fp") * 0.05
            - _metric(pair["7ch"], "fp") * 0.05
        )
        case_candidates.append((score, stem))
    case_stems = [s for _score, s in sorted(case_candidates, reverse=True)[:3]]

    for j, stem in enumerate(case_stems):
        sample = samples_by_stem[stem]
        ax = fig.add_subplot(gs[2, j])
        ax.imshow(sample.rgb)
        colors = {"3ch": "#3B82F6", "7ch": "#F97316"}
        for model in ("3ch", "7ch"):
            for inst in pred_cache.get((stem, model), [])[:20]:
                contours, _ = cv2.findContours(inst.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    pts = cnt.reshape(-1, 2)
                    ax.plot(pts[:, 0], pts[:, 1], color=colors[model], linewidth=0.8, alpha=0.9)
        ax.set_title(f"{stem}\n蓝=3ch, 橙=7ch", fontsize=8)
        ax.axis("off")

    fig.suptitle("7ch 无 Mask：分场景弱监督指标与物理一致性分析", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.015,
        "说明：伪标签用于弱参考匹配；OW/涡度/边界梯度用于无人工真值条件下的物理一致性补充评价。",
        ha="center",
        fontsize=9,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="3ch vs 7ch 分场景与物理一致性补充分析")
    ap.add_argument("--dataset-root", default="AutoDL/dataset/eddy")
    ap.add_argument("--enh-root", default="AutoDL/dataset/eddy_enh7")
    ap.add_argument("--nc-root", default="服创数据集/中尺度涡识别")
    ap.add_argument("--ckpt3", default="outputs/eddy_cloud_fair/last.pt")
    ap.add_argument("--ckpt7", default="outputs/eddy_enh7_cloud_fair/best.pt")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--conf", type=float, default=0.25, help="默认预测置信度阈值")
    ap.add_argument("--conf3", type=float, default=None, help="3ch 单独置信度阈值；默认使用 --conf")
    ap.add_argument("--conf7", type=float, default=None, help="7ch 单独置信度阈值；默认使用 --conf")
    ap.add_argument("--out-prefix", default="submission/tables/eddy_scene_physics_cloud_fair")
    ap.add_argument("--fig", default="submission/figures/eddy_7ch_advantage_analysis.png")
    args = ap.parse_args()

    dataset_root = _resolve(args.dataset_root)
    enh_root = _resolve(args.enh_root)
    nc_root = _resolve(args.nc_root)
    ckpt3 = _resolve(args.ckpt3)
    ckpt7 = _resolve(args.ckpt7)
    if not ckpt3.is_file():
        raise FileNotFoundError(ckpt3)
    if not ckpt7.is_file():
        raise FileNotFoundError(ckpt7)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    samples = _load_samples(dataset_root, enh_root, nc_root, splits)
    features = _image_features(samples)

    from ultralytics import YOLO

    conf3 = float(args.conf if args.conf3 is None else args.conf3)
    conf7 = float(args.conf if args.conf7 is None else args.conf7)
    models = {
        "3ch": (YOLO(str(ckpt3)), False, conf3),
        "7ch": (YOLO(str(ckpt7)), True, conf7),
    }

    sample_rows: list[dict[str, Any]] = []
    pred_cache: dict[tuple[str, str], list[Instance]] = {}
    samples_by_stem: dict[str, Sample] = {s.png.stem: s for s in samples}
    for sample in samples:
        feat = features[sample.png.stem]
        for model_name, (model, use_npy, conf) in models.items():
            pred = _predict_instances(model, sample, use_npy=use_npy, conf=conf)
            pred_cache[(sample.png.stem, model_name)] = pred
            match = _match_counts(sample.gt, pred)
            phys = _physical_scores(sample, pred)
            sample_rows.append(
                {
                    "split": sample.split,
                    "stem": sample.png.stem,
                    "model": model_name,
                    **{k: v for k, v in feat.items()},
                    **match,
                    **phys,
                }
            )

    group_keys = ["split", "contrast_group", "flow_group", "grad_group", "density_group", "size_group"]
    agg_rows: list[dict[str, Any]] = []
    for key in group_keys:
        agg_rows.extend(_aggregate(sample_rows, key))

    out_prefix = _resolve(args.out_prefix)
    _write_csv(out_prefix.with_suffix(".samples.csv"), sample_rows)
    _write_csv(out_prefix.with_suffix(".groups.csv"), agg_rows)
    _write_md(out_prefix.with_suffix(".groups.md"), agg_rows, title="涡旋 3ch vs 7ch 分场景与物理一致性分析")
    summary = {
        "note": "无人工真值：伪标签为弱参考，物理一致性为补充指标。",
        "splits": splits,
        "conf": float(args.conf),
        "conf3": conf3,
        "conf7": conf7,
        "dataset_root": str(dataset_root),
        "enh_root": str(enh_root),
        "ckpt3": str(ckpt3),
        "ckpt7": str(ckpt7),
        "group_metrics": agg_rows,
    }
    out_prefix.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    _plot_outputs(
        out=_resolve(args.fig),
        agg_rows=agg_rows,
        sample_rows=sample_rows,
        samples_by_stem=samples_by_stem,
        pred_cache=pred_cache,
    )
    print(out_prefix.with_suffix(".json"))
    print(out_prefix.with_suffix(".groups.csv"))
    print(out_prefix.with_suffix(".groups.md"))
    print(_resolve(args.fig))


if __name__ == "__main__":
    main()
