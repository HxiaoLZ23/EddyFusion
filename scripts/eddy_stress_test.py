#!/usr/bin/env python3
"""E4：标签/输入破坏烟测（检验 3ch 是否依赖伪标签结构）。"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from eddy_ablation_common import (
    load_samples,
    match_metrics,
    predict_instances,
    resolve_path,
    union_pred_mask,
    vote_mask,
    mask_iou,
)

def _prepare_label_dir(samples, mode: str, rng: random.Random) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="eddy_stress_"))
    lbl = tmp / "labels"
    lbl.mkdir(parents=True)
    stems = [s.png.stem for s in samples]
    label_paths = {s.png.stem: s.label for s in samples}
    for stem in stems:
        src = label_paths[stem]
        dst = lbl / f"{stem}.txt"
        if mode == "baseline":
            shutil.copy2(src, dst)
        elif mode == "label_empty":
            dst.write_text("", encoding="utf-8")
        elif mode == "label_shuffle":
            other = rng.choice([t for t in stems if t != stem])
            shutil.copy2(label_paths[other], dst)
        else:
            shutil.copy2(src, dst)
    return lbl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="AutoDL/dataset/eddy")
    ap.add_argument("--enh-root", default="AutoDL/dataset/eddy_enh7")
    ap.add_argument("--nc-root", default="服创数据集/中尺度涡识别")
    ap.add_argument("--ckpt3", default="outputs/eddy_cloud_fair/last.pt")
    ap.add_argument("--ckpt7", default="outputs/eddy_enh7_cloud_fair/best.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-samples", type=int, default=30)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-csv", default="submission/tables/eddy_ablation_E4_stress.csv")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    all_samples = load_samples(
        resolve_path(args.dataset_root),
        resolve_path(args.enh_root),
        resolve_path(args.nc_root),
        [args.split],
    )
    samples = all_samples[: args.max_samples]

    from ultralytics import YOLO

    m3 = YOLO(str(resolve_path(args.ckpt3)))
    m7 = YOLO(str(resolve_path(args.ckpt7)))

    modes = ["baseline", "label_shuffle", "label_empty"]
    rows: list[dict] = []
    for mode in modes:
        lbl_dir = _prepare_label_dir(samples, mode, rng)
        try:
            for s in samples:
                if mode == "baseline":
                    gt_path = s.label
                else:
                    gt_path = lbl_dir / f"{s.png.stem}.txt"
                from eddy_ablation_common import read_label

                h, w = s.rgb.shape[:2]
                gt = read_label(gt_path, h, w)
                vote = vote_mask(s.ow, (12.0, 18.0, 24.0, 30.0), 2)
                for model_name, model, use_npy in (("3ch", m3, False), ("7ch", m7, True)):
                    pred = predict_instances(model, s, use_npy=use_npy, conf=args.conf)
                    m = match_metrics(gt, pred)
                    pm = union_pred_mask(pred)
                    rows.append(
                        {
                            "mode": mode,
                            "model": model_name,
                            "stem": s.png.stem,
                            **m,
                            "vote_mask_iou": mask_iou(pm, vote),
                        }
                    )
        finally:
            shutil.rmtree(lbl_dir.parent, ignore_errors=True)

    # 聚合
    agg: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        agg.setdefault((r["mode"], r["model"]), []).append(r)

    out_rows: list[dict] = []
    for (mode, model), lst in sorted(agg.items()):
        n = len(lst)
        out_rows.append(
            {
                "mode": mode,
                "model": model,
                "n_images": n,
                "precision_mean": sum(x["precision"] for x in lst) / n,
                "recall_mean": sum(x["recall"] for x in lst) / n,
                "mean_iou_mean": sum(x["mean_iou"] for x in lst) / n,
                "vote_mask_iou_mean": sum(x["vote_mask_iou"] for x in lst) / n,
                "n_pred_mean": sum(x["n_pred"] for x in lst) / n,
            }
        )

    out = resolve_path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
