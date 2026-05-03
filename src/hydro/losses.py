"""水文训练辅助损失（EOS 近似约束等）。与配置 ``loss.*`` 对齐。"""

from __future__ import annotations

import torch


def eos_ts_residual_coupling(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    *,
    idx_temp: int,
    idx_sal: int,
) -> torch.Tensor:
    """软 EOS / T-S 耦合：在 z-score 空间用当前 batch 的标签估计 ``d(ps)/d(pt)`` 比例，惩罚预测偏差偏离该耦合。

    ``pred``, ``tgt``: ``(..., C)`` 末维为要素；内部对除 ``C`` 外所有维度做聚合（与均值平移后的协方差成比例）。
    """
    yt = tgt[..., idx_temp].reshape(tgt.shape[0], -1)
    ys = tgt[..., idx_sal].reshape(tgt.shape[0], -1)
    pt = pred[..., idx_temp].reshape(pred.shape[0], -1)
    ps = pred[..., idx_sal].reshape(pred.shape[0], -1)

    ym_t = yt.mean(dim=1, keepdim=True)
    ym_s = ys.mean(dim=1, keepdim=True)
    yt_c = yt - ym_t
    ys_c = ys - ym_s
    vn = torch.sum(yt_c**2, dim=1).clamp(min=1e-6)
    lam = torch.sum(ys_c * yt_c, dim=1) / vn
    lam = lam.detach()

    et = pt - yt
    es = ps - ys
    return ((es - lam.unsqueeze(1) * et) ** 2).mean()


def physics_loss_bundle(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    cfg: dict,
    *,
    feature_names: list[str],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """返回 `(total_aux, components)`；``total_aux`` 不含 MSE。"""
    device = pred.device
    zero = torch.zeros((), device=device, dtype=pred.dtype)
    parts: dict[str, torch.Tensor] = {}

    lc = cfg.get("loss", {}) if isinstance(cfg, dict) else {}
    if not lc.get("eos_constraint", {}).get("enabled"):
        return zero, parts

    names = list(feature_names)
    if "temp" not in names or "sal" not in names:
        return zero, parts

    iw = float(lc["eos_constraint"].get("weight", 0.1))
    it = names.index("temp")
    isal = names.index("sal")

    eos = eos_ts_residual_coupling(pred, tgt, idx_temp=it, idx_sal=isal)
    parts["eos_ts"] = eos
    return iw * eos, parts


def hydro_train_loss(pred: torch.Tensor, tgt: torch.Tensor, cfg: dict) -> torch.Tensor:
    """MSE + 可选 MAE 权重 + EOS 耦合。"""
    mse = torch.nn.functional.mse_loss(pred, tgt)
    lc = cfg.get("loss", {}) if isinstance(cfg, dict) else {}
    out = mse

    mw = lc.get("mae_weight")
    if mw is not None and float(mw) > 0:
        out = out + float(mw) * torch.nn.functional.l1_loss(pred, tgt)

    names = cfg.get("data", {}).get("target_features") or ()
    if not isinstance(names, (list, tuple)):
        names = ()
    eos_total, _ = physics_loss_bundle(pred, tgt, cfg, feature_names=list(names))
    out = out + eos_total
    return out
