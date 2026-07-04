#!/usr/bin/env python3
"""E5：预测 union mask 与 OW 投票掩膜的像素 IoU（3ch vs 7ch）。"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from eddy_ablation_common import (
    load_samples,
    predict_instances,
    resolve_path,
    union_pred_mask,
    vote_mask,
    mask_iou,
)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="AutoDL/dataset/eddy")
    ap.add_argument("--enh-root", default="AutoDL/dataset/eddy_enh7")
    ap.add_argument("--nc-root", default="服创数据集/中尺度涡识别")
    ap.add_argument("--ckpt3", default="outputs/eddy_cloud_fair/last.pt")
    ap.add_argument("--ckpt7", default="outputs/eddy_enh7_cloud_fair/best.pt")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--conf3", type=float, default=0.25)
    ap.add_argument("--conf7", type=float, default=0.25)
    ap.add_argument("--out-csv", default="submission/tables/eddy_ablation_E5_vote_overlap.csv")
    ap.add_argument("--out-fig", default="submission/figures/eddy_ablation_E5_vote_iou_hist.png")
    args = ap.parse_args()

    samples = load_samples(
        resolve_path(args.dataset_root),
        resolve_path(args.enh_root),
        resolve_path(args.nc_root),
        [s.strip() for s in args.splits.split(",") if s.strip()],
    )

    from ultralytics import YOLO

    m3 = YOLO(str(resolve_path(args.ckpt3)))
    m7 = YOLO(str(resolve_path(args.ckpt7)))

    rows: list[dict] = []
    for s in samples:
        vote = vote_mask(s.ow, (12.0, 18.0, 24.0, 30.0), 2)
        for model_name, model, use_npy, conf in (
            ("3ch", m3, False, args.conf3),
            ("7ch", m7, True, args.conf7),
        ):
            pred = predict_instances(model, s, use_npy=use_npy, conf=conf)
            pm = union_pred_mask(pred)
            rows.append(
                {
                    "split": s.split,
                    "stem": s.png.stem,
                    "model": model_name,
                    "conf": conf,
                    "vote_mask_iou": round(mask_iou(pm, vote), 4),
                    "pred_area_frac": round(float(pm.sum()) / pm.size, 6),
                    "vote_area_frac": round(float(vote.sum()) / vote.size, 6),
                    "n_pred": len(pred),
                }
            )

    out = resolve_path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = resolve_path(args.out_csv).with_suffix(".summary.json")
    import json

    summ = {}
    for model in ("3ch", "7ch"):
        sub = [r["vote_mask_iou"] for r in rows if r["model"] == model]
        summ[model] = {
            "mean_vote_iou": float(np.mean(sub)),
            "median_vote_iou": float(np.median(sub)),
            "p25": float(np.quantile(sub, 0.25)),
            "p75": float(np.quantile(sub, 0.75)),
        }
    summary.write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for model, color in (("3ch", "#2563eb"), ("7ch", "#dc2626")):
        sub = [r["vote_mask_iou"] for r in rows if r["model"] == model]
        ax.hist(sub, bins=20, alpha=0.55, label=model, color=color)
    ax.set_xlabel("IoU(pred union, OW vote mask)")
    ax.legend()
    out_fig = resolve_path(args.out_fig)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print(f"wrote {summary}")
    print(f"wrote {out_fig}")


if __name__ == "__main__":
    main()
