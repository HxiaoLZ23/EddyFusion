#!/usr/bin/env python3
"""E2 补充：3ch vs 7ch 置信度扫描（precision/recall，无 mAP）。"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from eddy_ablation_common import load_samples, match_metrics, predict_instances, resolve_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="AutoDL/dataset/eddy")
    ap.add_argument("--enh-root", default="AutoDL/dataset/eddy_enh7")
    ap.add_argument("--nc-root", default="服创数据集/中尺度涡识别")
    ap.add_argument("--ckpt3", default="outputs/eddy_cloud_fair/last.pt")
    ap.add_argument("--ckpt7", default="outputs/eddy_enh7_cloud_fair/best.pt")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument(
        "--confs",
        default="0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4",
        help="逗号分隔置信度列表（3ch/7ch 共用）",
    )
    ap.add_argument("--out-csv", default="submission/tables/eddy_ablation_E2_conf_sweep.csv")
    ap.add_argument("--out-fig", default="submission/figures/eddy_ablation_E2_conf.png")
    args = ap.parse_args()

    confs = [float(x.strip()) for x in args.confs.split(",") if x.strip()]
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
    for conf in confs:
        for model_name, model, use_npy in (("3ch", m3, False), ("7ch", m7, True)):
            precs, recs, fp_img = [], [], []
            for s in samples:
                pred = predict_instances(model, s, use_npy=use_npy, conf=conf)
                m = match_metrics(s.gt, pred)
                precs.append(m["precision"])
                recs.append(m["recall"])
                fp_img.append(m["fp"] / 1.0)
            n = max(len(samples), 1)
            rows.append(
                {
                    "model": model_name,
                    "conf": conf,
                    "precision_mean": sum(precs) / n,
                    "recall_mean": sum(recs) / n,
                    "fp_per_image_mean": sum(fp_img) / n,
                }
            )

    out_csv = resolve_path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for model_name, color in (("3ch", "#2563eb"), ("7ch", "#dc2626")):
        sub = [r for r in rows if r["model"] == model_name]
        ax.plot(
            [r["recall_mean"] for r in sub],
            [r["precision_mean"] for r in sub],
            "-o",
            label=model_name,
            color=color,
        )
    ax.set_xlabel("mean recall (pseudo-GT match)")
    ax.set_ylabel("mean precision")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_fig = resolve_path(args.out_fig)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_fig}")


if __name__ == "__main__":
    main()
