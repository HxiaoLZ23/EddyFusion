#!/usr/bin/env python3
"""
风浪 LSTM：24h 超前 MAE/RMSE（自回归 rollout，物理量纲）。

示例：
  python scripts/anomaly_eval_horizon24.py
  python scripts/anomaly_eval_horizon24.py --split test --horizon-hours 24
  python scripts/anomaly_eval_horizon24.py --horizon-hours 12 --out-json outputs/anomaly/metrics_summary_val_horizon12h.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.anomaly.horizon_eval import run_horizon_eval
from src.utils.config import load_yaml, pick_device, resolve_path
from src.utils.metrics import write_metrics_json


def main() -> None:
    ap = argparse.ArgumentParser(description="风浪指定超前小时 eval（默认 24h rollout）")
    ap.add_argument("--config", default="config/anomaly.yaml")
    ap.add_argument("--data-config", default="config/data.yaml")
    ap.add_argument("--ckpt", default="outputs/anomaly/best.pt")
    ap.add_argument("--split", choices=("val", "test", "both"), default="both")
    ap.add_argument("--horizon-hours", type=int, default=24)
    ap.add_argument("--stride", type=int, default=None, help="滑窗步长，默认 data.yaml window_stride")
    ap.add_argument(
        "--out-dir",
        default="outputs/anomaly",
        help="写出 metrics_summary_{split}_horizon{H}h.json",
    )
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
        metrics = run_horizon_eval(
            cfg,
            ckpt,
            device,
            split=split,
            data_cfg=data_cfg,
            horizon_hours=h,
            stride=args.stride,
        )
        series_meta = metrics.pop("series_meta", {})
        level = int(cfg["meta"]["level"])
        out_json = out_dir / f"metrics_summary_{split}_horizon{h}h.json"
        write_metrics_json(
            out_json,
            module="anomaly",
            level=level,
            metrics=metrics,
            passed=metrics["mae_avg"] < 0.5,
            tags={"level": level, "eval_split": split, "horizon_hours": h},
        )
        summary[split] = metrics
        print(f"\n=== {split} @ {h}h rollout (N={metrics['n_samples']}) ===")
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
        print(f"  label wind mean={metrics['label_wind_mean']:.3f} m/s  wave mean={metrics['label_wave_mean']:.3f} m")
        print(f"  wrote {out_json}")
        if series_meta:
            print(f"  series months_used={series_meta.get('months_used')} T={series_meta.get('T')}")

    audit_path = out_dir / f"eval_horizon{h}h_summary.json"
    audit_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {audit_path}")


if __name__ == "__main__":
    main()
