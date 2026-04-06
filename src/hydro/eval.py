from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.hydro.dataset import HydroNpzDataset
from src.hydro.model import build_model
from src.utils.config import load_yaml, pick_device, resolve_path
from src.utils.metrics import write_metrics_json


@torch.no_grad()
def run_eval(cfg: dict, ckpt: Path, device: torch.device) -> dict:
    paths = cfg["paths"]
    ds = HydroNpzDataset(paths["val_data"], paths["val_label"])
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    model = build_model(cfg).to(device)
    try:
        state = torch.load(ckpt, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    se = []
    ae = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        se.append(((pred - y) ** 2).mean(dim=(0, 1, 3, 4)).cpu().numpy())
        ae.append(y.abs().mean(dim=(0, 1, 3, 4)).cpu().numpy())
    se = np.stack(se, axis=0).mean(0)
    ae = np.stack(ae, axis=0).mean(0)
    rmse = np.sqrt(se)
    nrmse = rmse / np.maximum(ae, 1e-6)
    names = cfg["data"]["target_features"]
    nrmse_dict = {names[i]: float(nrmse[i]) for i in range(len(names))}
    avg = float(np.mean(nrmse))
    return {
        "rmse_per_feature": {names[i]: float(np.sqrt(se[i])) for i in range(len(names))},
        "nrmse_per_feature": nrmse_dict,
        "val_nrmse_avg": avg,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/hydro.yaml")
    parser.add_argument("--ckpt", type=str, default="outputs/hydro/best.pt")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    ckpt = resolve_path(args.ckpt)
    if not ckpt.is_file():
        out = resolve_path(cfg["paths"]["output_dir"])
        last = out / "last.pt"
        hint = f"未找到权重: {ckpt}"
        if last.is_file() and ckpt.name == "best.pt":
            hint += f"\n若训练曾出现 nan，可能只生成了 last.pt，可改用: --ckpt {last}"
        raise FileNotFoundError(hint)

    device = torch.device(pick_device(cfg["train"].get("device", "cuda")))
    metrics = run_eval(cfg, ckpt, device)

    level = int(cfg["meta"]["level"])
    tags = {
        "level": level,
        "mot.enabled": cfg.get("mot", {}).get("enabled"),
        "attn_res.enabled": cfg.get("attn_res", {}).get("enabled"),
    }
    passed = metrics["val_nrmse_avg"] <= 0.15  # 赛题 MSE 口径近似：以 NRMSE 作参考阈值

    out = cfg.get("eval", {}).get("metrics_file", "outputs/hydro/metrics_summary.json")
    write_metrics_json(
        out,
        module="hydro",
        level=level,
        metrics=metrics,
        passed=passed,
        tags=tags,
    )
    print("metrics:", metrics)


if __name__ == "__main__":
    main()
