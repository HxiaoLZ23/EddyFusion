from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.anomaly.dataset import AnomalyNpzDataset
from src.anomaly.model import build_model
from src.utils.config import load_yaml, pick_device, resolve_path
from src.utils.metrics import write_metrics_json


@torch.no_grad()
def run_eval(cfg: dict, ckpt: Path, device: torch.device, split: str) -> dict:
    paths = cfg["paths"]
    key = f"{split}_sequences"
    if key not in paths:
        raise KeyError(f"配置 paths 中缺少 {key}，无法评估 split={split}")
    ds = AnomalyNpzDataset(paths[key])
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    model = build_model(cfg).to(device)
    try:
        state = torch.load(ckpt, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    maes = []
    rmses = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        mae = (pred - y).abs().mean(dim=0)
        rmse = torch.sqrt(((pred - y) ** 2).mean(dim=0))
        maes.append(mae.cpu().numpy())
        rmses.append(rmse.cpu().numpy())
    mae = np.stack(maes, axis=0).mean(0)
    rmse = np.stack(rmses, axis=0).mean(0)
    metrics = {
        "mae_wind": float(mae[0]),
        "mae_wave": float(mae[1]),
        "rmse_wind": float(rmse[0]),
        "rmse_wave": float(rmse[1]),
        "mae_avg": float(mae.mean()),
        "rmse_avg": float(rmse.mean()),
        "split": split,
    }
    if split == "val":
        metrics["val_mae_avg"] = metrics["mae_avg"]
    else:
        metrics["test_mae_avg"] = metrics["mae_avg"]
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/anomaly.yaml")
    parser.add_argument("--ckpt", type=str, default="outputs/anomaly/best.pt")
    parser.add_argument(
        "--split",
        type=str,
        choices=("val", "test"),
        default="val",
        help="使用 paths 中 val_sequences 或 test_sequences",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    ckpt = resolve_path(args.ckpt)
    if not ckpt.is_file():
        out = resolve_path(cfg["paths"]["output_dir"])
        last = out / "last.pt"
        hint = f"未找到权重: {ckpt}"
        if last.is_file() and ckpt.name == "best.pt":
            hint += f"\n可改用: --ckpt {last}"
        raise FileNotFoundError(hint)

    device = torch.device(pick_device(cfg["train"].get("device", "cuda")))
    metrics = run_eval(cfg, ckpt, device, split=args.split)

    level = int(cfg["meta"]["level"])
    # 赛题「准确率≥80%」需命题方口径；回归 MAE 作过程指标，阈值占位
    passed = metrics["mae_avg"] < 0.5

    mf = cfg.get("eval", {}).get("metrics_file", "outputs/anomaly/metrics_summary.json")
    mp = resolve_path(mf)
    out_json = mp.parent / f"{mp.stem}_{args.split}{mp.suffix}"
    write_metrics_json(
        out_json,
        module="anomaly",
        level=level,
        metrics=metrics,
        passed=passed,
        tags={"level": level, "eval_split": args.split},
    )
    print(f"wrote {out_json}")
    print("metrics:", metrics)


if __name__ == "__main__":
    main()
