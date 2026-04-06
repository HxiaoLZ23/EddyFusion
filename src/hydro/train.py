from __future__ import annotations

import argparse
import math
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
) -> tuple[float, bool]:
    """返回 (平均 train MSE, 本 epoch 是否至少执行过一次 optimizer.step)。"""
    model.train()
    total = 0.0
    n = 0
    accum = max(1, grad_accum_steps)
    optimizer.zero_grad(set_to_none=True)
    micro = 0
    stepped = False
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
            stepped = True
            optimizer.zero_grad(set_to_none=True)
    return total / max(n, 1), stepped


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    scores: list[float] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        scores.append(nrmse_batch(pred, y))
    if not scores:
        return 0.0
    arr = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        print("警告: 验证集中存在非有限 NRMSE（常为预测或标签含 nan/inf）", flush=True)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        print("警告: 验证 NRMSE 全部为 nan/inf（请检查标签与模型输出）", flush=True)
        return float("nan")
    return float(finite.mean())


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
    n_tr, n_va = len(train_ds), len(val_ds)
    print(f"数据集: train={n_tr} 样本, val={n_va} 样本", flush=True)
    if n_tr == 0:
        raise ValueError(
            "训练集样本数为 0：请检查 config 中 train_data/train_label 路径与预处理是否已生成 npz。"
        )
    if n_va == 0:
        print("警告: 验证集样本数为 0，val_nrmse 将恒为 0", flush=True)

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
    last_path = out_dir / "last.pt"
    best_metric = float("inf")

    x0, y0 = train_ds[0]
    if not torch.isfinite(x0).all() or not torch.isfinite(y0).all():
        print(
            "警告: 训练集第 0 条样本含 nan/inf，将导致 loss 发散；请检查预处理与 Z-score。",
            flush=True,
        )

    for ep in range(1, epochs + 1):
        tr_loss, stepped = train_epoch(
            model, train_loader, device, optimizer, scaler, grad_clip, grad_accum
        )
        val_nrmse = validate(model, val_loader, device)
        if stepped:
            scheduler.step()
        elif ep == 1:
            print(
                "警告: 本 epoch 未执行 optimizer.step()（训练 DataLoader 是否为空？）"
                " 已跳过 scheduler.step()。",
                flush=True,
            )
        print(f"epoch {ep}/{epochs} train_mse={tr_loss:.6f} val_nrmse={val_nrmse:.6f}")

        torch.save(
            {"model": model.state_dict(), "cfg": cfg, "epoch": ep},
            last_path,
        )

        if math.isfinite(val_nrmse) and math.isfinite(tr_loss) and val_nrmse < best_metric:
            best_metric = val_nrmse
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": ep}, best_path)
        elif not math.isfinite(val_nrmse) or not math.isfinite(tr_loss):
            print(
                "提示: 当前指标含 nan/inf，不会写入 best.pt；已更新 last.pt。"
                " 可尝试: 降低 lr、train.amp=false、检查预处理 npz 是否含 nan、"
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 仅缓解碎片。",
                flush=True,
            )

    if not best_path.is_file():
        print(
            f"注意: 全程未得到有限验证指标，未生成 {best_path.name}。"
            f" 仍可用 last.pt 做调试: --ckpt {last_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
