from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class WindWaveLSTM(nn.Module):
    """双任务：共享 LSTM 骨干 + 风速头/波高头。"""

    def __init__(self, in_features: int = 2, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            in_features,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.wind_head = nn.Linear(hidden_dim, 1)
        self.wave_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B T F
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        wind = self.wind_head(last)
        wave = self.wave_head(last)
        return torch.cat([wind, wave], dim=1)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True):  # type: ignore[override]
        """兼容旧版单 head 权重（head.weight/head.bias）。"""
        if "head.weight" in state_dict and "head.bias" in state_dict:
            head_w = state_dict.pop("head.weight")
            head_b = state_dict.pop("head.bias")
            if head_w.ndim == 2 and head_w.shape[0] >= 2:
                state_dict["wind_head.weight"] = head_w[0:1, :]
                state_dict["wave_head.weight"] = head_w[1:2, :]
            if head_b.ndim == 1 and head_b.shape[0] >= 2:
                state_dict["wind_head.bias"] = head_b[0:1]
                state_dict["wave_head.bias"] = head_b[1:2]
        return super().load_state_dict(state_dict, strict=strict)


def build_model(cfg: dict[str, Any]) -> WindWaveLSTM:
    m = cfg["model"]
    return WindWaveLSTM(
        in_features=2,
        hidden_dim=int(m["hidden_dim"]),
        num_layers=int(m.get("num_layers", 1)),
    )
