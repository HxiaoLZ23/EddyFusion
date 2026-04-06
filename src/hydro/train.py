from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from src.hydro.dataset import HydroNpzDataset
from src.hydro.model import build_model
from src.utils.config import load_yaml, pick_device, project_root, resolve_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def nrmse_batch(pred: torch.Tensor, y: torch.Tensor) -> float:
    """pred, y: B T C H W — 各通道 RMSE/|y|均值 再平均。"""
    with torch.no_grad():
        err = pred - y
        rmse = torch.sqrt((err**2).mean(dim=(0, 2, 3, 4)))  # per-channel over B,T,H,W
        denom = y.abs().mean(dim=(0, 2, 3, 4)).clamp(min=1e-6)
        per_ch = (rmse / denom).mean()
        return float(per_ch.item())


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler | None,
    grad_clip: float,
    grad_accum_steps: int = 1,
) -> float:
    model.train()
    total = 0.0
    n = 0
    accum = max(1, grad_accum_steps)
    optimizer.zero_grad(set_to_none=True)
    micro = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if scaler:
            with autocast("cuda", enabled=device.type == "cuda"):
                pred = model(x)
                loss_full = nn.functional.mse_loss(pred, y)
                loss = loss_full / accum
            scaler.scale(loss).backward()
        else:
            pred = model(x)
            loss_full = nn.functional.mse_loss(pred, y)
            (loss_full / accum).backward()
        bs = x.size(0)
        total += float(loss_full.item()) * bs
        n += bs
        micro += 1
        if micro % accum == 0 or micro == len(loader):
            if scaler:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return total / max(n, 1)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    scores: list[float] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        scores.append(nrmse_batch(pred, y))
    return float(np.mean(scores)) if scores else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="水文训练")
    parser.add_argument(
        "--config",
        type=str,
        default="config/hydro.yaml",
        help="默认配置；若使用 --synthetic 则自动改用 config/hydro_synthetic.yaml",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="使用小型合成数据与 hydro_synthetic 配置先跑通训练",
    )
    args = parser.parse_args()

    root = project_root()
    if args.synthetic:
        from src.preprocess.hydro_dataset import generate_synthetic

        synth_cfg = root / "config/hydro_synthetic.yaml"
        generate_synthetic(synth_cfg)
        cfg = load_yaml(synth_cfg)
    else:
        cfg = load_yaml(args.config)
    set_seed(int(cfg["meta"]["seed"]))

    paths = cfg["paths"]

    train_ds = HydroNpzDataset(paths["train_data"], paths["train_label"])
    val_ds = HydroNpzDataset(paths["val_data"], paths["val_label"])

    tc = cfg["train"]
    device_str = pick_device(tc.get("device", "cuda"))
    device = torch.device(device_str)
    batch_size = int(tc["batch_size"])
    loader_kw = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": int(cfg["data"].get("num_workers", 0)),
        "pin_memory": bool(cfg["data"].get("pin_memory", False)) and device_str == "cuda",
    }
    train_loader = DataLoader(train_ds, **loader_kw)
    val_bs = int(tc.get("eval_batch_size", batch_size))
    val_loader = DataLoader(val_ds, batch_size=val_bs, shuffle=False, num_workers=loader_kw["num_workers"])

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tc["lr"]),
        weight_decay=float(tc["weight_decay"]),
    )
    epochs = int(tc["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    use_amp = bool(tc.get("amp", True)) and device_str == "cuda"
    scaler = GradScaler("cuda") if use_amp else None
    grad_clip = float(tc.get("grad_clip_norm", 0))
    grad_accum = int(tc.get("gradient_accumulation_steps", 1))

    out_dir = resolve_path(paths["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    best_metric = float("inf")

    for ep in range(1, epochs + 1):
        tr_loss = train_epoch(
            model, train_loader, device, optimizer, scaler, grad_clip, grad_accum
        )
        val_nrmse = validate(model, val_loader, device)
        scheduler.step()
        print(f"epoch {ep}/{epochs} train_mse={tr_loss:.6f} val_nrmse={val_nrmse:.6f}")
        if val_nrmse < best_metric:
            best_metric = val_nrmse
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": ep}, best_path)


if __name__ == "__main__":
    main()
