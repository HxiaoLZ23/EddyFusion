#!/usr/bin/env python3
"""
格点场 3h 超前 MAE/RMSE（逐格点滑窗，物理量纲）。

默认 subsample：time_stride=4（12h 取样）、space_stride=2（风场格点隔点），
避免 241×321 全格点全时刻过慢。全格点可加 --space-stride 1 --time-stride 1（耗时长）。

示例：
  python scripts/anomaly_eval_grid3h.py --split test
  python scripts/anomaly_eval_grid3h.py --split both --time-stride 8 --space-stride 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.anomaly.grid_eval import run_grid_eval
from src.utils.config import load_yaml, pick_device, resolve_path
from src.utils.metrics import write_metrics_json


def main() -> None:
    ap = argparse.ArgumentParser(description="风浪格点场 3h eval")
    ap.add_argument("--config", default="config/anomaly.yaml")
    ap.add_argument("--data-config", default="config/data.yaml")
    ap.add_argument("--ckpt", default="outputs/anomaly/best.pt")
    ap.add_argument("--split", choices=("val", "test", "both"), default="both")
    ap.add_argument("--horizon-hours", type=int, default=3)
    ap.add_argument("--time-stride", type=int, default=4)
    ap.add_argument("--space-stride", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--out-dir", default="outputs/anomaly")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = load_yaml(args.data_config)
    ckpt = resolve_path(args.ckpt)
    if not ckpt.is_file():
        raise SystemExit(f"权重不存在: {ckpt}")

    device = __import__("torch").device(pick_device(cfg["train"].get("device", "cuda")))
    splits = ("val", "test") if args.split == "both" else (args.split,)
    h = int(args.horizon_hours)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}
    for split in splits:
        metrics = run_grid_eval(
            cfg,
            ckpt,
            device,
            split=split,
            data_cfg=data_cfg,
            horizon_hours=h,
            time_stride=int(args.time_stride),
            space_stride=int(args.space_stride),
            batch_size=int(args.batch_size),
        )
        series_meta = metrics.pop("series_meta", {})
        out_json = out_dir / f"metrics_summary_{split}_grid{h}h.json"
        write_metrics_json(
            out_json,
            module="anomaly",
            level=int(cfg["meta"]["level"]),
            metrics=metrics,
            passed=metrics["mae_avg"] < 0.5,
            tags={"level": int(cfg["meta"]["level"]), "eval_split": split, "grid_eval": True, "horizon_hours": h},
        )
        summary[split] = metrics
        print(f"\n=== {split} 格点场 @ {h}h (points={metrics['n_grid_points']} samples={metrics['n_eval_samples']}) ===")
        print(
            f"  stride time={metrics['time_stride']} space={metrics['space_stride']}  "
            f"grid={series_meta.get('grid_hw')}"
        )
        print(
            f"  MAE  wind={metrics['mae_wind']:.4f}  wave={metrics['mae_wave']:.4f}  avg={metrics['mae_avg']:.4f}"
        )
        print(
            f"  RMSE wind={metrics['rmse_wind']:.4f} wave={metrics['rmse_wave']:.4f} avg={metrics['rmse_avg']:.4f}"
        )
        print(
            f"  Persistence MAE avg={metrics['persistence_mae_avg']:.4f}  "
            f"ratio={metrics['mae_avg_vs_persistence_ratio']:.3f}"
        )
        print(f"  label wind mean={metrics['label_wind_mean']:.3f}±{metrics['label_wind_std']:.3f} m/s")
        print(f"  label wave mean={metrics['label_wave_mean']:.3f}±{metrics['label_wave_std']:.3f} m")
        print(f"  wrote {out_json}")

    audit = out_dir / f"eval_grid{h}h_summary.json"
    audit.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {audit}")


if __name__ == "__main__":
    main()
