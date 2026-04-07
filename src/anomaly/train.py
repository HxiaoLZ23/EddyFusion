from __future__ import annotations

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from src.anomaly.dataset import AnomalyNpzDataset
from src.anomaly.model import build_model
from src.utils.config import load_yaml, pick_device, project_root, resolve_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="风-浪双任务训练")
    parser.add_argument("--config", type=str, default="config/anomaly.yaml")
    parser.add_argument("--synthetic", action="store_true", help="先生成合成 npz")
    args = parser.parse_args()

    root = project_root()
    if args.synthetic:
        from src.preprocess.anomaly_dataset import generate_synthetic_anomaly

        generate_synthetic_anomaly(root / args.config)

    cfg = load_yaml(args.config)
    set_seed(int(cfg["meta"]["seed"]))
    paths = cfg["paths"]
    train_ds = AnomalyNpzDataset(paths["train_sequences"])
    val_ds = AnomalyNpzDataset(paths["val_sequences"])

    tc = cfg["train"]
    device_str = pick_device(tc.get("device", "cuda"))
    device = torch.device(device_str)
    batch_size = int(tc["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(tc["lr"]))
    epochs = int(tc["epochs"])
    use_amp = bool(tc.get("amp", True)) and device_str == "cuda"
    scaler = GradScaler("cuda") if use_amp else None

    out_dir = resolve_path(paths["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    best = float("inf")

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                with autocast("cuda", enabled=device.type == "cuda"):
                    pred = model(x)
                    loss = nn.functional.mse_loss(pred, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(x)
                loss = nn.functional.mse_loss(pred, y)
                loss.backward()
                optimizer.step()
            bs = x.size(0)
            total += float(loss.item()) * bs
            n += bs
        train_loss = total / max(n, 1)

        model.eval()
        vloss = 0.0
        vn = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                loss = nn.functional.mse_loss(pred, y)
                vloss += float(loss.item()) * x.size(0)
                vn += x.size(0)
        val_loss = vloss / max(vn, 1)
        print(f"epoch {ep}/{epochs} train_mse={train_loss:.6f} val_mse={val_loss:.6f}")
        if val_loss < best:
            best = val_loss
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": ep}, best_path)
        if args.synthetic and ep >= 3:
            break


if __name__ == "__main__":
    main()
