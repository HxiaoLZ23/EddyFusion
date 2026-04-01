from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.anomaly.dataset import AnomalyNpzDataset
from src.anomaly.model import build_model
from src.utils.config import load_yaml, pick_device, resolve_path
from src.utils.metrics import write_metrics_json


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/anomaly.yaml")
    parser.add_argument("--ckpt", type=str, default="outputs/anomaly/best.pt")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    ckpt = resolve_path(args.ckpt)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    device = torch.device(pick_device(cfg["train"].get("device", "cuda")))
    model = build_model(cfg).to(device)
    try:
        state = torch.load(ckpt, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    paths = cfg["paths"]
    ds = AnomalyNpzDataset(paths["test_sequences"])
    loader = DataLoader(ds, batch_size=64, shuffle=False)

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
    }
    # 赛题「准确率≥80%」需命题方口径；此处用回归误差作占位，passed 仅作演示
    passed = metrics["mae_avg"] < 0.5

    level = int(cfg["meta"]["level"])
    out = cfg.get("eval", {}).get("metrics_file", "outputs/anomaly/metrics_summary.json")
    write_metrics_json(
        out,
        module="anomaly",
        level=level,
        metrics=metrics,
        passed=passed,
        tags={"level": level},
    )
    print("metrics:", metrics)


if __name__ == "__main__":
    main()
