#!/usr/bin/env python3
"""E3：OW 多分位投票超参敏感性（烟测帧，不重训）。"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from eddy_ablation_common import close_cache, load_fields, parse_stem, resolve_path, vote_mask, mask_iou


def _label_stats(label_path: Path, h: int, w: int) -> dict:
    from eddy_ablation_common import read_label

    gt = read_label(label_path, h, w)
    areas = [float(g.mask.sum()) for g in gt]
    return {
        "n_instances": len(gt),
        "total_area_px": sum(areas),
        "mean_vertices": 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="AutoDL/dataset/eddy")
    ap.add_argument("--nc-root", default="服创数据集/中尺度涡识别")
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-samples", type=int, default=20)
    ap.add_argument("--out-csv", default="submission/tables/eddy_ablation_E3_vote_sensitivity.csv")
    args = ap.parse_args()

    root = resolve_path(args.dataset_root)
    nc_root = resolve_path(args.nc_root)
    pngs = sorted((root / "images" / args.split).glob("*.png"))[: args.max_samples]

    configs = [
        ("default_12_18_24_30_m2", (12.0, 18.0, 24.0, 30.0), 2),
        ("strict_m3", (12.0, 18.0, 24.0, 30.0), 3),
        ("loose_m1", (12.0, 18.0, 24.0, 30.0), 1),
        ("round2_10_15_20_25_30_m2", (10.0, 15.0, 20.0, 25.0, 30.0), 2),
    ]

    cache: dict = {}
    rows: list[dict] = []
    try:
        for png in pngs:
            nc_stem, ti = parse_stem(png.stem)
            _a, _u, _v, _z, ow, _gm = load_fields(nc_root, nc_stem, ti, cache)
            h, w = ow.shape
            ref_name, ref_p, ref_m = configs[0]
            ref_mask = vote_mask(ow, ref_p, ref_m)
            ref_area = int(ref_mask.sum())
            for name, percs, vmin in configs:
                m = vote_mask(ow, percs, vmin)
                rows.append(
                    {
                        "stem": png.stem,
                        "config": name,
                        "vote_min": vmin,
                        "percentiles": ",".join(str(int(p)) for p in percs),
                        "mask_area_px": int(m.sum()),
                        "mask_area_frac": round(float(m.sum()) / (h * w), 6),
                        "iou_vs_default": round(mask_iou(m, ref_mask), 4),
                        "label_n_inst": _label_stats(root / "labels" / args.split / f"{png.stem}.txt", h, w)[
                            "n_instances"
                        ],
                    }
                )
    finally:
        close_cache(cache)

    out = resolve_path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 汇总：相对 default 的 area 变化
    summary_path = out.with_name(out.stem + "_summary.md")
    lines = ["# E3 vote 敏感性汇总", ""]
    for name, _, _ in configs:
        sub = [r for r in rows if r["config"] == name]
        ious = [r["iou_vs_default"] for r in sub if r["config"] != "default_12_18_24_30_m2"]
        areas = [r["mask_area_frac"] for r in sub]
        lines.append(f"- **{name}**: mean mask_area_frac={np.mean(areas):.4f}, mean iou_vs_default={np.mean([r['iou_vs_default'] for r in sub]):.4f}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
