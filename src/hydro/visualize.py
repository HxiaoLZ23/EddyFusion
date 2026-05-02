from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.hydro.dataset import HydroNpzDataset
from src.utils.config import ensure_dir, resolve_path


@torch.no_grad()
def save_hydro_example_plots(
    *,
    model: torch.nn.Module,
    cfg: dict,
    device: torch.device,
    split: str = "val",
    sample_index: int = 0,
    xy: tuple[torch.Tensor, torch.Tensor] | None = None,
    out_dir: str | Path = "outputs/hydro/figures",
    tag: str = "eval",
) -> list[Path]:
    """保存水文可视化图：四要素 t+72 空间图 + 区域均值曲线。

    若传入 ``xy``（单样本 ``x,y`` 均为 ``T×C×H×W`` CPU tensor），则不再 ``np.load``
    重复构造数据集，避免训练过程中周期性出图导致内存翻倍/OOM。
    """
    if xy is not None:
        x, y = xy
        if x.dim() != 4 or y.dim() != 4:
            raise ValueError("xy 应为单样本 (T,C,H,W) 的 x 与 y")
    else:
        paths = cfg["paths"]
        sx = f"{split}_data"
        sy = f"{split}_label"
        ds = HydroNpzDataset(paths[sx], paths[sy])
        if len(ds) == 0:
            return []
        idx = int(max(0, min(sample_index, len(ds) - 1)))
        x, y = ds[idx]

    model.eval()
    try:
        pred = model(x.unsqueeze(0).to(device)).detach().float().cpu().numpy()[0]
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()
    gt = y.detach().float().numpy()

    feats = list(cfg["data"]["target_features"])
    outp = ensure_dir(out_dir)
    t_show = gt.shape[0] - 1  # 默认展示最后一步（常对应 t+72）
    saved: list[Path] = []

    for ci, name in enumerate(feats):
        g = gt[t_show, :, :, ci]
        p = pred[t_show, :, :, ci]
        e = np.abs(p - g)
        vmin = float(min(np.nanmin(g), np.nanmin(p)))
        vmax = float(max(np.nanmax(g), np.nanmax(p)))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = 0.0, 1.0
        elif vmax - vmin < 1e-8:
            vmax = vmin + 1e-8

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(g, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0].set_title(f"{name} GT t+{t_show + 1}")
        axes[1].imshow(p, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1].set_title(f"{name} Pred t+{t_show + 1}")
        axes[2].imshow(e, cmap="magma")
        axes[2].set_title(f"{name} |Err| t+{t_show + 1}")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fp = resolve_path(outp / f"hydro_{tag}_{split}_{name}_map_t{t_show + 1}.png")
        fig.savefig(fp, dpi=200)
        plt.close(fig)
        saved.append(fp)

    # 区域均值曲线（每个要素一张子图）
    t_steps = gt.shape[0]
    tt = np.arange(1, t_steps + 1)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ci, name in enumerate(feats):
        ax = axes[ci // 2, ci % 2]
        gt_mean = gt[:, :, :, ci].mean(axis=(1, 2))
        pd_mean = pred[:, :, :, ci].mean(axis=(1, 2))
        ax.plot(tt, gt_mean, label="GT")
        ax.plot(tt, pd_mean, label="Pred")
        ax.set_title(f"{name} area-mean")
        ax.set_xlabel("Horizon step")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fp_curve = resolve_path(outp / f"hydro_{tag}_{split}_area_mean_curves.png")
    fig.savefig(fp_curve, dpi=200)
    plt.close(fig)
    saved.append(fp_curve)
    plt.close("all")
    return saved

