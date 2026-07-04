#!/usr/bin/env python3
"""风浪模块 C：数据量纲 + 持续性基线 + 模型 MAE 审计（回应「是否未反归一化」质疑）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.anomaly.dataset import AnomalyNpzDataset
from src.anomaly.eval import _label_and_persistence_stats, run_eval
from src.utils.config import load_yaml, pick_device, resolve_path


def _print_split_audit(split: str, ds: AnomalyNpzDataset) -> None:
    stats = _label_and_persistence_stats(ds)
    print(f"\n=== {split} 数据审计（N={len(ds)}）===")
    print(
        f"  标签风速 |U10|: mean={stats['label_wind_mean']:.3f} std={stats['label_wind_std']:.3f} "
        f"range=[{stats['label_wind_min']:.3f}, {stats['label_wind_max']:.3f}] m/s"
    )
    print(
        f"  标签波高 SWH: mean={stats['label_wave_mean']:.3f} std={stats['label_wave_std']:.3f} "
        f"range=[{stats['label_wave_min']:.3f}, {stats['label_wave_max']:.3f}] m"
    )
    print(
        f"  持续性基线 MAE: wind={stats['persistence_mae_wind']:.4f} wave={stats['persistence_mae_wave']:.4f} "
        f"avg={stats['persistence_mae_avg']:.4f}"
    )
    rel_w = stats["persistence_mae_wind"] / max(stats["label_wind_std"], 1e-9)
    rel_h = stats["persistence_mae_wave"] / max(stats["label_wave_std"], 1e-9)
    print(f"  持续性 MAE / label_std: wind={rel_w:.3f} wave={rel_h:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="风浪 eval 量纲与基线审计")
    ap.add_argument("--config", default="config/anomaly.yaml")
    ap.add_argument("--ckpt", default="outputs/anomaly/best.pt")
    ap.add_argument("--split", choices=("val", "test", "both"), default="both")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = load_yaml("config/data.yaml")
    ckpt = resolve_path(args.ckpt)
    splits = ("val", "test") if args.split == "both" else (args.split,)

    print("结论预览：预处理/anomaly_dataset 与 eval 均无 StandardScaler；若 label 均值在 1~3 m/s、1~2 m 量级，则为物理单位而非 z-score。")

    device = torch.device(pick_device(cfg["train"].get("device", "cuda")))
    all_metrics: dict[str, dict] = {}
    for split in splits:
        key = f"{split}_sequences"
        ds = AnomalyNpzDataset(cfg["paths"][key])
        _print_split_audit(split, ds)
        if ckpt.is_file():
            m = run_eval(cfg, ckpt, device, split=split, data_cfg=data_cfg)
            all_metrics[split] = m
            print(f"  模型 MAE: wind={m['mae_wind']:.4f} wave={m['mae_wave']:.4f} avg={m['mae_avg']:.4f}")
            print(f"  相对持续性: mae_avg/persistence_avg={m['mae_avg_vs_persistence_ratio']:.3f} (<1 表示优于持续性)")
            print(f"  相对标签均值: wind={m['mae_wind']/max(m['label_wind_mean'],1e-9):.3f} wave={m['mae_wave']/max(m['label_wave_mean'],1e-9):.3f}")
        else:
            print(f"  跳过模型 eval：权重不存在 {ckpt}")

    if all_metrics:
        out = resolve_path("outputs/anomaly/eval_audit_summary.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
