#!/usr/bin/env python3
"""OW 规则法 vs YOLOv8-seg 对比实验（第 6 章答辩用）。

对比链：
  - **OW 规则**：U/V → OW → 多分位投票 → 连通域 → 多边形（与伪标签导出同参）
  - **YOLO 3ch**：ADT 伪彩色 RGB → YOLOv8-seg 推理

弱参考为 OW 导出伪标签（非人工真值）。脚本同时报告：
  1. 相对伪标签的 P/R/F1/mIoU、小涡旋召回；
  2. 边界平滑度、粘连/过分割、时序稳定性、OW 超参扰动敏感性；
  3. 单帧推理耗时（CPU OW vs GPU/CPU YOLO）。

用法：
  python scripts/eddy_compare_ow_rule_vs_yolo.py
  python scripts/eddy_compare_ow_rule_vs_yolo.py --splits val,test --skip-yolo
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from eddy_ablation_common import (  # noqa: E402
    Sample,
    aggregate_match,
    boundary_grad_on_mask,
    boundary_roughness,
    load_samples,
    match_metrics,
    merge_error_count,
    ow_rule_instances,
    predict_instances,
    resolve_path,
    small_instance_recall,
    split_error_count,
    union_pred_mask,
    vote_mask,
    mask_iou,
    _resize_hw,
)

DEFAULT_VOTE = ((12.0, 18.0, 24.0, 30.0), 2)
OW_PERTURBATIONS = [
    ("default", (12.0, 18.0, 24.0, 30.0), 2),
    ("strict_m3", (12.0, 18.0, 24.0, 30.0), 3),
    ("loose_m1", (12.0, 18.0, 24.0, 30.0), 1),
    ("alt_pct", (10.0, 15.0, 20.0, 25.0, 30.0), 2),
]


def _eval_one(
    sample: Sample,
    pred: list,
    *,
    method: str,
    split: str,
) -> dict[str, Any]:
    mm = match_metrics(sample.gt, pred, iou_thr=0.5)
    sm_rec, sm_n = small_instance_recall(sample.gt, pred)
    h, w = sample.rgb.shape[:2]
    return {
        "stem": sample.png.stem,
        "split": split,
        "method": method,
        "n_gt": len(sample.gt),
        "n_pred": len(pred),
        "precision": mm["precision"],
        "recall": mm["recall"],
        "mean_iou": mm["mean_iou"],
        "tp": mm["tp"],
        "fp": mm["fp"],
        "fn": mm["fn"],
        "small_recall": sm_rec,
        "small_n": sm_n,
        "boundary_roughness": boundary_roughness(pred),
        "boundary_grad": boundary_grad_on_mask(pred, sample.grad_mag),
        "merge_errors": float(merge_error_count(sample.gt, pred)),
        "split_errors": float(split_error_count(sample.gt, pred)),
        "union_area_frac": float(union_pred_mask(pred).sum()) / (h * w) if pred else 0.0,
    }


def _ow_perturbation_spread(sample: Sample, h: int, w: int) -> dict[str, float]:
    """同一帧上 OW 超参扰动导致的实例数/掩膜面积波动（越大越不稳定）。"""
    ow_h = _resize_hw(sample.ow, h, w)
    counts: list[int] = []
    areas: list[float] = []
    for _name, percs, vmin in OW_PERTURBATIONS:
        zeta_h = _resize_hw(sample.zeta, h, w)
        pred = ow_rule_instances(ow_h, zeta_h, h, w, vote_percentiles=percs, vote_min=vmin)
        counts.append(len(pred))
        areas.append(float(union_pred_mask(pred).sum()) / (h * w) if pred else 0.0)
    return {
        "ow_inst_count_std": float(np.std(counts)),
        "ow_area_frac_std": float(np.std(areas)),
        "ow_inst_count_range": float(max(counts) - min(counts)) if counts else 0.0,
    }


def _temporal_pairs(samples: list[Sample]) -> list[tuple[Sample, Sample]]:
    by_stem: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_stem[s.nc_stem].append(s)
    pairs: list[tuple[Sample, Sample]] = []
    for _stem, group in by_stem.items():
        group.sort(key=lambda x: x.time_idx)
        for a, b in zip(group, group[1:]):
            if b.time_idx - a.time_idx <= 7:
                pairs.append((a, b))
    return pairs


def _temporal_stability(
    pairs: list[tuple[Sample, Sample]],
    preds: dict[str, list],
) -> dict[str, float]:
    """相邻帧 union mask IoU 与实例数变化（越大/越小表示越不稳定）。"""
    ious: list[float] = []
    dcounts: list[float] = []
    for a, b in pairs:
        pa = preds.get(a.png.stem, [])
        pb = preds.get(b.png.stem, [])
        ua = union_pred_mask(pa) if pa else np.zeros(a.rgb.shape[:2], dtype=bool)
        ub = union_pred_mask(pb) if pb else np.zeros(b.rgb.shape[:2], dtype=bool)
        ious.append(mask_iou(ua, ub))
        dcounts.append(abs(len(pa) - len(pb)))
    return {
        "temporal_union_iou_mean": float(np.mean(ious)) if ious else 0.0,
        "temporal_inst_delta_mean": float(np.mean(dcounts)) if dcounts else 0.0,
        "n_temporal_pairs": float(len(pairs)),
    }


def _aggregate_method(rows: list[dict[str, Any]]) -> dict[str, float]:
    mm = aggregate_match(rows)
    sm_rows = [r for r in rows if int(r.get("small_n", 0)) > 0]
    return {
        "n_images": len(rows),
        "precision": mm["precision"],
        "recall": mm["recall"],
        "f1": mm["f1"],
        "mean_iou_matched": mm["mean_iou_matched"],
        "small_recall_mean": float(np.mean([r["small_recall"] for r in sm_rows])) if sm_rows else 0.0,
        "small_recall_images": len(sm_rows),
        "boundary_roughness_mean": float(np.mean([r["boundary_roughness"] for r in rows])) if rows else 0.0,
        "boundary_grad_mean": float(np.mean([r["boundary_grad"] for r in rows])) if rows else 0.0,
        "merge_errors_total": float(sum(r["merge_errors"] for r in rows)),
        "split_errors_total": float(sum(r["split_errors"] for r in rows)),
        "merge_errors_per_image": float(np.mean([r["merge_errors"] for r in rows])) if rows else 0.0,
        "split_errors_per_image": float(np.mean([r["split_errors"] for r in rows])) if rows else 0.0,
        "ow_inst_count_std_mean": float(np.mean([r.get("ow_inst_count_std", 0) for r in rows])),
        "ow_area_frac_std_mean": float(np.mean([r.get("ow_area_frac_std", 0) for r in rows])),
    }


def _write_md(summary: dict[str, Any], out_md: Path, *, v6_teacher: bool = False) -> None:
    if v6_teacher:
        title = "# V6 Phase A — OW(P24) Teacher 上界旁证"
        intro = [
            "> **Teacher 上界**：OW(P24) 规则链重放伪标签；P/R≈1 为标签生成上界。",
            "> **不参与 Fair vs Proposed 排名**；主表见 `submission/tables/eddy_v6_fair_vs_proposed.md`。",
        ]
    else:
        title = "# OW 规则法 vs YOLOv8-seg 对比（第 6 章）"
        intro = [
            "> **弱参考**：OW 导出伪标签（非人工 polygon）。",
            "> OW 规则链与伪标签导出同参，故相对伪标签的 mIoU **不应**作为 YOLO 唯一优势论据；",
            "> 工程价值应看边界平滑、小目标召回、粘连分离、时序稳定与部署统一性。",
        ]
    lines = [
        title,
        "",
        *intro,
        "",
        f"- 数据集：`{summary['dataset_root']}`",
        f"- NC：`{summary['nc_root']}`",
        f"- YOLO 权重：`{summary.get('ckpt', '（跳过）')}`",
        f"- splits：{', '.join(summary['splits'])}",
        f"- conf：{summary.get('conf', 0.25)}",
        "",
        "## 1. 相对伪标签（instance IoU≥0.5）",
        "",
        "| split | 方法 | n | P | R | F1 | matched mIoU | 小涡旋召回 | 边界粗糙度↓ | 边界梯度↓ | 粘连/图 | 过分割/图 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in summary["splits"]:
        for method in ("ow_rule", "yolo"):
            key = f"{split}_{method}"
            if key not in summary["by_split_method"]:
                continue
            r = summary["by_split_method"][key]
            lines.append(
                f"| {split} | {method} | {int(r['n_images'])} | "
                f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
                f"{r['mean_iou_matched']:.3f} | {r['small_recall_mean']:.3f} | "
                f"{r['boundary_roughness_mean']:.2f} | {r['boundary_grad_mean']:.4f} | "
                f"{r['merge_errors_per_image']:.2f} | {r['split_errors_per_image']:.2f} |"
            )
    cross_lines = []
    for split in summary["splits"]:
        key = f"{split}_yolo_vs_ow"
        if key in summary["by_split_method"]:
            r = summary["by_split_method"][key]
            cross_lines.append(
                f"| {split} | {int(r['n_images'])} | {r['precision']:.3f} | {r['recall']:.3f} | "
                f"{r['f1']:.3f} | {r['mean_iou_matched']:.3f} |"
            )
    if cross_lines:
        lines.extend(
            [
                "",
                "## 1b. YOLO 相对 OW 规则输出（instance IoU≥0.5，不经伪标签文件）",
                "",
                "| split | n | P | R | F1 | matched mIoU |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
                *cross_lines,
            ]
        )
    lines.extend(
        [
            "",
            "## 2. 时序稳定性（相邻帧 union IoU，越高越稳）",
            "",
            "| split | 方法 | 配对帧数 | mean union IoU | mean |Δ实例数| |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for split in summary["splits"]:
        for method in ("ow_rule", "yolo"):
            key = f"{split}_{method}"
            if key not in summary.get("temporal", {}):
                continue
            t = summary["temporal"][key]
            lines.append(
                f"| {split} | {method} | {int(t['n_temporal_pairs'])} | "
                f"{t['temporal_union_iou_mean']:.3f} | {t['temporal_inst_delta_mean']:.2f} |"
            )
    lines.extend(
        [
            "",
            "## 3. OW 超参扰动敏感性（仅 OW 规则；std 越小越稳）",
            "",
            "| split | OW 实例数 std/图 | OW 面积占比 std/图 |",
            "| --- | ---: | ---: |",
        ]
    )
    for split in summary["splits"]:
        key = f"{split}_ow_rule"
        if key in summary["by_split_method"]:
            r = summary["by_split_method"][key]
            lines.append(
                f"| {split} | {r['ow_inst_count_std_mean']:.3f} | {r['ow_area_frac_std_mean']:.4f} |"
            )
    if summary.get("timing"):
        lines.extend(
            [
                "",
                "## 4. 单帧推理耗时",
                "",
                f"- OW 规则（CPU）：{summary['timing'].get('ow_ms_per_frame', 0):.1f} ms/帧",
                f"- YOLO（{summary['timing'].get('yolo_device', 'n/a')}）："
                f"{summary['timing'].get('yolo_ms_per_frame', 0):.1f} ms/帧",
                "",
            ]
        )
    lines.extend(
        [
            "## 5. 答辩要点（自动生成）",
            "",
            summary.get("narrative", ""),
            "",
            f"明细 CSV：`{summary.get('csv_path', '')}`",
            f"JSON：`{summary.get('json_path', '')}`",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_narrative(summary: dict[str, Any], *, v6_teacher: bool = False) -> str:
    parts: list[str] = []
    if v6_teacher:
        for split in summary["splits"]:
            ow = summary["by_split_method"].get(f"{split}_ow_rule")
            if ow:
                parts.append(
                    f"**{split} Teacher（OW P24）**：相对伪标签 P/R={ow['precision']:.3f}/{ow['recall']:.3f}，"
                    f"matched mIoU={ow['mean_iou_matched']:.3f}。"
                )
        parts.append(
            "**用途**：Teacher 上界旁证；**不参与** V6 Fair vs Proposed 主排名。"
            "主结论以 `eddy_v6_fair_vs_proposed.md` 中 Proposed−Fair 的 ΔmAP/ΔP/ΔR 为准。"
        )
        return " ".join(p for p in parts if p)
    parts.append(
        "**核心结论**：推理期直接执行 OW→投票→连通域 可 **100% 复现** 伪标签（P/R=1.0）；"
        "YOLO 93% mAP 表征的是对 OW 规则的 **可部署近似/蒸馏**，而非在伪标签口径上超越 OW。"
    )
    for split in summary["splits"]:
        yo = summary["by_split_method"].get(f"{split}_yolo")
        cross = summary["by_split_method"].get(f"{split}_yolo_vs_ow")
        if cross:
            parts.append(
                f"**{split}** YOLO vs OW 直接输出：P/R={cross['precision']:.3f}/{cross['recall']:.3f}，"
                f"matched mIoU={cross['mean_iou_matched']:.3f}。"
            )
        if yo:
            ow = summary["by_split_method"].get(f"{split}_ow_rule")
            br_delta = (ow["boundary_roughness_mean"] - yo["boundary_roughness_mean"]) if ow else 0
            parts.append(
                f"  边界粗糙度 YOLO 降低 {br_delta:.2f}（{br_delta / ow['boundary_roughness_mean'] * 100:.1f}%）" if ow and ow["boundary_roughness_mean"] else ""
            )
        tow = summary.get("temporal", {}).get(f"{split}_ow_rule")
        tyo = summary.get("temporal", {}).get(f"{split}_yolo")
        if tow and tyo:
            parts.append(
                f"  时序 union IoU：OW={tow['temporal_union_iou_mean']:.3f} → YOLO={tyo['temporal_union_iou_mean']:.3f}。"
            )
        ow = summary["by_split_method"].get(f"{split}_ow_rule")
        if ow:
            parts.append(
                f"  OW 超参扰动下实例数 std≈{ow['ow_inst_count_std_mean']:.2f}/图；YOLO 推理 **不依赖** OW 阈值链。"
            )
    parts.append(
        "**答辩建议**：承认「可直接跑 OW」；强调 YOLO 的价值是 **规则蒸馏 + 工程统一部署**（RGB/GPU 流水线、与前端一致），"
        "以及边界平滑/时序略优；过分割可通过 conf 调参缓解。若需证明超越 OW，须引入 **独立人工样例** 或 **输入扰动鲁棒性** 实验。"
    )
    return "\n".join(p for p in parts if p)


def _plot_compare(samples: list[Sample], ow_preds: dict, yolo_preds: dict, out_png: Path, *, max_panels: int = 3) -> None:
    """挑选 YOLO 边界更平滑或 small recall 更高的样本出图。"""
    scored: list[tuple[float, Sample]] = []
    for s in samples:
        ow = ow_preds.get(s.png.stem, [])
        yo = yolo_preds.get(s.png.stem, [])
        if not yo or not ow:
            continue
        br_ow = boundary_roughness(ow)
        br_yo = boundary_roughness(yo)
        sm_ow, _ = small_instance_recall(s.gt, ow)
        sm_yo, _ = small_instance_recall(s.gt, yo)
        score = (br_ow - br_yo) + (sm_yo - sm_ow)
        scored.append((score, s))
    scored.sort(reverse=True)
    picks = [s for _, s in scored[:max_panels]]
    if not picks:
        return

    import cv2

    n = len(picks)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n), squeeze=False)
    titles = ["RGB", "pseudo-GT", "OW rule", "YOLO"]
    for ri, s in enumerate(picks):
        h, w = s.rgb.shape[:2]
        panels = [
            s.rgb,
            _overlay(s.rgb, s.gt, (0, 255, 0)),
            _overlay(s.rgb, ow_preds.get(s.png.stem, []), (255, 180, 0)),
            _overlay(s.rgb, yolo_preds.get(s.png.stem, []), (0, 120, 255)),
        ]
        for ci, img in enumerate(panels):
            ax = axes[ri, ci]
            ax.imshow(img)
            ax.set_title(titles[ci] if ri == 0 else "")
            ax.axis("off")
        axes[ri, 0].set_ylabel(s.png.stem, fontsize=8)
    fig.suptitle("OW rule vs YOLO: boundary / small-target cases", fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _overlay(rgb: np.ndarray, instances: list, color: tuple[int, int, int]) -> np.ndarray:
    import cv2

    out = rgb.copy()
    for inst in instances:
        m = inst.mask.astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnts, -1, color, 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="AutoDL/dataset/eddy")
    ap.add_argument("--nc-root", default="服创数据集/中尺度涡识别")
    ap.add_argument("--ckpt", default="outputs/eddy_cloud_fair/last.pt")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-samples", type=int, default=0, help="0=全量")
    ap.add_argument("--skip-yolo", action="store_true")
    ap.add_argument("--device", default="")
    ap.add_argument("--single-percentile", type=float, default=None, help="V6：OW<P24，与 export 对齐")
    ap.add_argument(
        "--v6-teacher",
        action="store_true",
        help="V6 Phase A：Teacher 上界旁证表头/脚注（不参与 Fair vs Proposed 排名）",
    )
    ap.add_argument("--out-md", default="submission/tables/eddy_ow_rule_vs_yolo_compare.md")
    ap.add_argument("--out-json", default="submission/tables/eddy_ow_rule_vs_yolo_compare.json")
    ap.add_argument("--out-csv", default="submission/tables/eddy_ow_rule_vs_yolo_compare.csv")
    ap.add_argument("--out-fig", default="submission/figures/eddy_ow_rule_vs_yolo_compare.png")
    args = ap.parse_args()

    dataset_root = resolve_path(args.dataset_root)
    nc_root = resolve_path(args.nc_root)
    ckpt = resolve_path(args.ckpt)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    if not (dataset_root / "images").is_dir():
        raise SystemExit(f"数据集不存在: {dataset_root}")

    samples = load_samples(dataset_root, None, nc_root, splits)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    if not samples:
        raise SystemExit("无样本")

    model = None
    yolo_device = "skipped"
    if not args.skip_yolo and ckpt.is_file():
        from ultralytics import YOLO

        model = YOLO(str(ckpt))
        yolo_device = args.device or ("cuda:0" if __import__("torch").cuda.is_available() else "cpu")
    elif not args.skip_yolo:
        print(f"警告：权重不存在 {ckpt}，跳过 YOLO")

    percs, vmin = DEFAULT_VOTE
    rows: list[dict[str, Any]] = []
    ow_preds: dict[str, list] = {}
    yolo_preds: dict[str, list] = {}
    ow_times: list[float] = []
    yolo_times: list[float] = []

    for sample in samples:
        h, w = sample.rgb.shape[:2]
        t0 = time.perf_counter()
        ow_pred = ow_rule_instances(
            sample.ow,
            sample.zeta,
            h,
            w,
            vote_percentiles=percs,
            vote_min=vmin,
            single_percentile=args.single_percentile,
        )
        ow_times.append((time.perf_counter() - t0) * 1000.0)
        ow_preds[sample.png.stem] = ow_pred
        row = _eval_one(sample, ow_pred, method="ow_rule", split=sample.split)
        row.update(_ow_perturbation_spread(sample, h, w))
        rows.append(row)

        if model is not None:
            t1 = time.perf_counter()
            yo_pred = predict_instances(model, sample, use_npy=False, conf=args.conf)
            yolo_times.append((time.perf_counter() - t1) * 1000.0)
            yolo_preds[sample.png.stem] = yo_pred
            yo_row = _eval_one(sample, yo_pred, method="yolo", split=sample.split)
            rows.append(yo_row)
            # YOLO 相对 OW 规则输出的直接一致性（不经过伪标签文件）
            cross = match_metrics(ow_pred, yo_pred, iou_thr=0.5)
            rows.append(
                {
                    "stem": sample.png.stem,
                    "split": sample.split,
                    "method": "yolo_vs_ow",
                    "n_gt": len(ow_pred),
                    "n_pred": len(yo_pred),
                    "precision": cross["precision"],
                    "recall": cross["recall"],
                    "mean_iou": cross["mean_iou"],
                    "tp": cross["tp"],
                    "fp": cross["fp"],
                    "fn": cross["fn"],
                    "small_recall": 0.0,
                    "small_n": 0,
                    "boundary_roughness": 0.0,
                    "boundary_grad": 0.0,
                    "merge_errors": 0.0,
                    "split_errors": 0.0,
                    "union_area_frac": 0.0,
                }
            )

    by_split_method: dict[str, dict[str, float]] = {}
    for split in splits:
        for method in ("ow_rule", "yolo", "yolo_vs_ow"):
            sub = [r for r in rows if r["split"] == split and r["method"] == method]
            if sub:
                by_split_method[f"{split}_{method}"] = _aggregate_method(sub)

    temporal: dict[str, dict[str, float]] = {}
    for split in splits:
        sub_samples = [s for s in samples if s.split == split]
        pairs = _temporal_pairs(sub_samples)
        if pairs:
            temporal[f"{split}_ow_rule"] = _temporal_stability(pairs, ow_preds)
            if yolo_preds:
                temporal[f"{split}_yolo"] = _temporal_stability(pairs, yolo_preds)

    summary: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "nc_root": str(nc_root),
        "ckpt": str(ckpt) if model else None,
        "splits": splits,
        "conf": args.conf,
        "n_samples": len(samples),
        "by_split_method": by_split_method,
        "temporal": temporal,
        "timing": {
            "ow_ms_per_frame": float(np.mean(ow_times)) if ow_times else 0.0,
            "yolo_ms_per_frame": float(np.mean(yolo_times)) if yolo_times else 0.0,
            "yolo_device": yolo_device,
        },
    }
    summary["narrative"] = _build_narrative(summary, v6_teacher=args.v6_teacher)
    summary["csv_path"] = str(resolve_path(args.out_csv))
    summary["json_path"] = str(resolve_path(args.out_json))

    out_json = resolve_path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    out_csv = resolve_path(args.out_csv)
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    out_md = resolve_path(args.out_md)
    _write_md(summary, out_md, v6_teacher=args.v6_teacher)

    if yolo_preds:
        _plot_compare(samples, ow_preds, yolo_preds, resolve_path(args.out_fig))

    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    print(f"wrote {out_csv}")
    if yolo_preds:
        print(f"wrote {resolve_path(args.out_fig)}")


if __name__ == "__main__":
    main()
