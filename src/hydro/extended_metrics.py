"""水文扩展评估指标：逐通道 MAE/RMSE/NRMSE、相对持久性 Skill、Pearson、可选反标准化误差。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.hydro.dataset import HydroNpzDataset
from src.hydro.model import build_model


def _persistence_baseline(y: torch.Tensor, x_last: torch.Tensor) -> torch.Tensor:
    """y: (B,T,C,H,W)；x_last: (B,C,H,W) 上一时刻观测，复制为全时域预测。"""
    b, tout, _, h, w = y.shape
    return x_last.unsqueeze(1).expand(b, tout, -1, h, w).contiguous()


@torch.no_grad()
def evaluate_extended_on_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    feature_names: list[str],
    mean_1d: np.ndarray | None = None,
    std_1d: np.ndarray | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """在 val/test loader 上累计扩展指标。"""
    c_out = len(feature_names)
    n_total = 0.0  # 每通道 B*T*H*W 累计（各通道相同）

    sse_m = torch.zeros(c_out, dtype=torch.float64, device="cpu")  # MSE accum (sum squared err)
    sae_m = torch.zeros(c_out, dtype=torch.float64, device="cpu")
    sse_p = torch.zeros(c_out, dtype=torch.float64, device="cpu")
    sae_p = torch.zeros(c_out, dtype=torch.float64, device="cpu")

    # Pearson: E[xy]-E[x]E[y] over flattened B,T,H,W per channel
    sum_p = torch.zeros(c_out, dtype=torch.float64, device="cpu")
    sum_y = torch.zeros(c_out, dtype=torch.float64, device="cpu")
    sum_p2 = torch.zeros(c_out, dtype=torch.float64, device="cpu")
    sum_y2 = torch.zeros(c_out, dtype=torch.float64, device="cpu")
    sum_py = torch.zeros(c_out, dtype=torch.float64, device="cpu")
    sum_abs_y = torch.zeros(c_out, dtype=torch.float64, device="cpu")

    use_physical = mean_1d is not None and std_1d is not None
    mu_phys: torch.Tensor | None = None
    sd_phys: torch.Tensor | None = None
    sse_phys: torch.Tensor | None = None
    sae_phys: torch.Tensor | None = None
    sum_abs_y_phys: torch.Tensor | None = None
    if use_physical:
        mu = np.asarray(mean_1d, dtype=np.float64).reshape(-1)
        sd = np.asarray(std_1d, dtype=np.float64).reshape(-1)
        if mu.shape[0] == c_out == sd.shape[0]:
            mu_phys = torch.from_numpy(mu).view(1, 1, 1, 1, c_out)
            sd_phys = torch.from_numpy(sd).view(1, 1, 1, 1, c_out)
            sse_phys = torch.zeros(c_out, dtype=torch.float64, device="cpu")
            sae_phys = torch.zeros(c_out, dtype=torch.float64, device="cpu")
            sum_abs_y_phys = torch.zeros(c_out, dtype=torch.float64, device="cpu")
        else:
            use_physical = False

    eps = 1e-12
    n_batch = 0
    model.eval()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)

        persist = _persistence_baseline(y, x[:, -1])

        bd = lambda t: torch.permute(t, (0, 1, 3, 4, 2)).contiguous()  # B T H W C
        pn = bd(pred).double()
        yn = bd(y).double()
        per = bd(persist).double()

        # 累计量在 CPU；batch 可能在 CUDA，须在相加前移到 CPU，避免 device 混用
        sse_m += ((pn - yn) ** 2).reshape(-1, c_out).sum(dim=0).cpu()
        sae_m += (pn - yn).abs().reshape(-1, c_out).sum(dim=0).cpu()
        sse_p += ((per - yn) ** 2).reshape(-1, c_out).sum(dim=0).cpu()
        sae_p += (per - yn).abs().reshape(-1, c_out).sum(dim=0).cpu()

        flat = pn.reshape(-1, c_out)
        sum_p += flat.sum(dim=0).cpu()
        sum_y += yn.reshape(-1, c_out).sum(dim=0).cpu()
        sum_p2 += (pn**2).reshape(-1, c_out).sum(dim=0).cpu()
        sum_y2 += (yn**2).reshape(-1, c_out).sum(dim=0).cpu()
        sum_py += (pn * yn).reshape(-1, c_out).sum(dim=0).cpu()
        sum_abs_y += yn.abs().reshape(-1, c_out).sum(dim=0).cpu()
        n_total += float(pn.reshape(-1, c_out).shape[0])

        if use_physical and mu_phys is not None and sd_phys is not None and sse_phys is not None:
            pn_p = pn * sd_phys + mu_phys
            yn_p = yn * sd_phys + mu_phys
            diff_p = pn_p - yn_p
            sse_phys += (diff_p**2).reshape(-1, c_out).sum(dim=0).cpu()
            sae_phys += diff_p.abs().reshape(-1, c_out).sum(dim=0).cpu()
            sum_abs_y_phys += yn_p.abs().reshape(-1, c_out).sum(dim=0).cpu()

        n_batch += 1
        if max_batches is not None and n_batch >= int(max_batches):
            break

    nt = max(n_total, eps)
    mse_m_np = (sse_m / nt).numpy()
    mse_p_np = (sse_p / nt).numpy()
    rmse = {feature_names[i]: float(np.sqrt(mse_m_np[i])) for i in range(c_out)}
    mean_abs_y = (sum_abs_y / nt).numpy()
    nrmse = {
        feature_names[i]: float(np.sqrt(mse_m_np[i]) / max(float(mean_abs_y[i]), eps))
        for i in range(c_out)
    }
    mae = {feature_names[i]: float((sae_m[i] / nt).item()) for i in range(c_out)}
    rmse_pers = {feature_names[i]: float(np.sqrt(mse_p_np[i])) for i in range(c_out)}

    skill = {}
    for i, name in enumerate(feature_names):
        mp = float(mse_p_np[i])
        mm = float(mse_m_np[i])
        skill[name] = float(1.0 - mm / max(mp, eps)) if mp > eps else None

    pearson: dict[str, float | None] = {}
    n_all = nt
    for i, name in enumerate(feature_names):
        n = float(n_all)
        if n <= 1:
            pearson[name] = None
            continue
        s_p = float(sum_p[i])
        s_y = float(sum_y[i])
        s_pp = float(sum_p2[i])
        s_yy = float(sum_y2[i])
        s_py = float(sum_py[i])
        num = n * s_py - s_p * s_y
        vx = n * s_pp - s_p * s_p
        vy = n * s_yy - s_y * s_y
        denom = vx * vy
        if denom <= eps:
            pearson[name] = None
        else:
            pearson[name] = float(num / np.sqrt(max(denom, eps)))

    out: dict[str, Any] = {
        "mae_per_feature": mae,
        "rmse_per_feature": rmse,
        "mse_persistence_per_feature": {feature_names[i]: float(mse_p_np[i]) for i in range(c_out)},
        "rmse_persistence_per_feature": rmse_pers,
        "skill_vs_persistence": skill,
        "pearson_per_feature": pearson,
        "mse_model_per_feature": {feature_names[i]: float(mse_m_np[i]) for i in range(c_out)},
        "nrmse_per_feature": nrmse,
        "mean_abs_y_per_feature": {feature_names[i]: float(mean_abs_y[i]) for i in range(c_out)},
        "n_summed_per_channel_dim": nt,
    }

    if mean_1d is not None and std_1d is not None:
        mu = np.asarray(mean_1d, dtype=np.float64).reshape(-1)
        sd = np.asarray(std_1d, dtype=np.float64).reshape(-1)
        if mu.shape[0] == c_out == sd.shape[0]:
            out["rmse_physical_scale"] = {
                feature_names[i]: float(np.sqrt(mse_m_np[i]) * sd[i]) for i in range(c_out)
            }
            out["note_rmse_physical_scale"] = (
                "RMSE(norm)×std；与下方 rmse_physical 在 z-score 仿射下对误差等价，"
                "但 NRMSE 分母须用反标准化后的 mean(|y|)。"
            )

    if use_physical and sse_phys is not None and sae_phys is not None and sum_abs_y_phys is not None:
        mse_phys_np = (sse_phys / nt).numpy()
        mean_abs_y_phys = (sum_abs_y_phys / nt).numpy()
        rmse_phys = {
            feature_names[i]: float(np.sqrt(mse_phys_np[i])) for i in range(c_out)
        }
        nrmse_phys = {
            feature_names[i]: float(np.sqrt(mse_phys_np[i]) / max(float(mean_abs_y_phys[i]), eps))
            for i in range(c_out)
        }
        mae_phys = {feature_names[i]: float((sae_phys[i] / nt).item()) for i in range(c_out)}
        out["rmse_physical_per_feature"] = rmse_phys
        out["nrmse_physical_per_feature"] = nrmse_phys
        out["mae_physical_per_feature"] = mae_phys
        out["mean_abs_y_physical_per_feature"] = {
            feature_names[i]: float(mean_abs_y_phys[i]) for i in range(c_out)
        }
        out["note_nrmse_physical"] = (
            "pred_phys = pred_z*std+mean，y_phys 同理；NRMSE_phys = RMSE_phys/mean(|y_phys|)，"
            "与 eval.py 的 NRMSE 结构一致，分母在物理量纲（°C、PSU、m/s）。"
        )
        out["rmse_physical_avg"] = float(np.mean([rmse_phys[f] for f in feature_names]))
        out["nrmse_physical_avg"] = float(np.mean([nrmse_phys[f] for f in feature_names]))
        out["mae_physical_avg"] = float(np.mean([mae_phys[f] for f in feature_names]))

    out["rmse_avg"] = float(np.mean([rmse[f] for f in feature_names]))
    out["mae_avg"] = float(np.mean([mae[f] for f in feature_names]))
    out["nrmse_avg"] = float(np.mean([nrmse[f] for f in feature_names]))
    skills_valid = [s for s in skill.values() if s is not None]
    out["skill_avg"] = float(np.mean(skills_valid)) if skills_valid else None

    pears = [p for p in pearson.values() if p is not None]
    out["pearson_avg"] = float(np.mean(pears)) if pears else None

    return out


@torch.no_grad()
def evaluate_checkpoint_extended(
    cfg: dict[str, Any],
    ckpt: Path | str,
    device: torch.device,
    *,
    split: str = "val",
    stats_npz_path: Path | str | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    paths = cfg["paths"]
    sx = f"{split}_data"
    sy = f"{split}_label"
    if sx not in paths or sy not in paths:
        raise KeyError(f"paths 缺少 {sx}/{sy}")
    ds = HydroNpzDataset(paths[sx], paths[sy])
    bs = max(1, int(cfg["train"].get("eval_batch_size", cfg["train"].get("batch_size", 1))))
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)

    ckpt = Path(ckpt)
    model = build_model(cfg).to(device)
    try:
        state = torch.load(ckpt, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])

    mean_1d = None
    std_1d = None
    if stats_npz_path:
        zp = Path(stats_npz_path)
        if zp.is_file():
            z = np.load(zp)
            mean = z["mean"]
            std = z["std"]
            mean_1d = np.asarray(mean.reshape(-1), dtype=np.float64)
            std_1d = np.asarray(std.reshape(-1), dtype=np.float64)
            feats = list(z["features"]) if "features" in z.files else []
            targ = cfg["data"]["target_features"]
            if feats and list(feats) != targ:
                # 仍可尝试按索引截断对齐
                if len(feats) >= len(targ):
                    mean_1d = mean_1d[: len(targ)]
                    std_1d = std_1d[: len(targ)]

    names = list(cfg["data"]["target_features"])
    return evaluate_extended_on_loader(
        model,
        loader,
        device,
        feature_names=names,
        mean_1d=mean_1d,
        std_1d=std_1d,
        max_batches=max_batches,
    )
