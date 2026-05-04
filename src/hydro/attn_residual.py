"""Block / Full AttnRes：在 ConvLSTM 时空编码后、解码卷积前堆叠残差注意力块。

与 ``config`` 中 ``attn_res.enabled`` / ``type`` / ``num_blocks`` 对齐；显存友好（通道 + 7×7 空间门控 + 局部 FFN）。"""

from __future__ import annotations

import torch
import torch.nn as nn


class AttnResidualBlock2d(nn.Module):
    """单块：GroupNorm → 通道注意力 → 空间门控 → 3×3 FFN，残差相加。"""

    def __init__(
        self,
        dim: int,
        *,
        reduction: int = 4,
        pre_norm: bool = True,
        deep_ffn: bool = False,
    ) -> None:
        super().__init__()
        self.pre_norm = bool(pre_norm)
        g = min(8, int(dim))
        while dim % g != 0 and g > 1:
            g -= 1
        self.norm = nn.GroupNorm(g, dim)
        mid = max(8, dim // int(reduction))
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, mid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, dim, 1, bias=True),
        )
        self.sa = nn.Conv2d(dim, 1, kernel_size=7, padding=3, bias=True)
        ffn_layers: list[nn.Module] = [
            nn.Conv2d(dim, dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1, bias=True),
        ]
        if deep_ffn:
            ffn_layers += [
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, 3, padding=1, bias=True),
            ]
        self.ffn = nn.Sequential(*ffn_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x) if self.pre_norm else x
        ca = torch.sigmoid(self.ca(h))
        h = h * ca
        h = h * torch.sigmoid(self.sa(h))
        h = self.ffn(h)
        return residual + h


class AttnResNeck(nn.Module):
    """多块串联。"""

    def __init__(
        self,
        dim: int,
        num_blocks: int,
        *,
        reduction: int = 4,
        pre_norm: bool = True,
        deep_ffn: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                AttnResidualBlock2d(
                    dim,
                    reduction=reduction,
                    pre_norm=pre_norm,
                    deep_ffn=deep_ffn,
                )
                for _ in range(int(num_blocks))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x


def build_attn_neck(cfg: dict, *, hidden_dim: int) -> AttnResNeck | None:
    ar = cfg.get("attn_res") if isinstance(cfg, dict) else None
    if not ar or not bool(ar.get("enabled")):
        return None
    t = str(ar.get("type", "block")).lower()
    deep = t == "full"
    n = int(ar.get("num_blocks", 4))
    red = int(ar.get("reduction", 4))
    pre_norm = bool(ar.get("pre_norm", True))
    return AttnResNeck(
        int(hidden_dim),
        n,
        reduction=red,
        pre_norm=pre_norm,
        deep_ffn=deep,
    )


def build_attn_residual(config: dict) -> AttnResNeck | None:
    """兼容旧名；``hidden_dim`` 取自 ``config['model']['hidden_dim']``。"""
    m = config.get("model", {}) if isinstance(config, dict) else {}
    hd = int(m.get("hidden_dim", 128))
    return build_attn_neck(config, hidden_dim=hd)
