"""MoT-lite：时间维双流 depthwise Conv3d 专家 + 门控混合（先于 ConvLSTM 编码）。

与 `config/hydro.yaml` / `hydro_hycom_l1.yaml` 中 ``mot.enabled`` 对应。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalMotStem(nn.Module):
    """
    ``x``: ``(B,T,C,H,W)`` → ``(B,T,C,H,W)``。
    在 ``T`` 维上做两组 depthwise（groups=C）卷积后再按像素门控线性混合，
    不改动 ``T`` 长度，可与现有 ``HydroBaseline`` 无损衔接。
    """

    def __init__(
        self,
        *,
        channels: int,
        kernel_short: int = 3,
        kernel_long: int = 7,
        mid_channels: int | None = None,
    ) -> None:
        super().__init__()
        c = int(channels)
        self.kernel_short = int(kernel_short)
        self.kernel_long = int(kernel_long)
        mid = int(mid_channels or min(32, max(8, c // 2)))

        ks = kernel_short | 1
        kl = kernel_long | 1
        pad_s = ks // 2
        pad_l = kl // 2

        self.expert_short = nn.Conv3d(
            c, c, kernel_size=(ks, 1, 1), padding=(pad_s, 0, 0), groups=c, bias=True
        )
        self.expert_long = nn.Conv3d(
            c, c, kernel_size=(kl, 1, 1), padding=(pad_l, 0, 0), groups=c, bias=True
        )
        self.gate = nn.Sequential(
            nn.Conv3d(c, mid, kernel_size=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, c, kernel_size=(1, 1, 1)),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,T,C,H,W) -> (B,C,T,H,W)
        xc = x.permute(0, 2, 1, 3, 4).contiguous()
        e1 = self.expert_short(xc)
        e2 = self.expert_long(xc)
        g = self.gate(xc)
        y = g * e1 + (1.0 - g) * e2
        return y.permute(0, 2, 1, 3, 4).contiguous()


def build_mot_stem(cfg: dict, *, in_channels: int) -> TemporalMotStem | None:
    m = cfg.get("mot") if isinstance(cfg, dict) else None
    if not m or not bool(m.get("enabled")):
        return None
    k1 = int(m.get("kernel_short", m.get("expert_kernel_short", 3)))
    k2 = int(m.get("kernel_long", m.get("expert_kernel_long", 7)))
    mid = m.get("mid_channels") or m.get("branch_hidden_dim")
    kw: dict[str, object] = {"channels": in_channels, "kernel_short": k1, "kernel_long": k2}
    if mid is not None:
        kw["mid_channels"] = int(mid)
    return TemporalMotStem(**kw)
