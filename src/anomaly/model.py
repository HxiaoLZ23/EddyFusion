from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class WindWaveLSTM(nn.Module):
    """双任务：共享 LSTM，线性头输出风速、波高下一时刻。"""

    def __init__(self, in_features: int = 2, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            in_features,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B T F
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


def build_model(cfg: dict[str, Any]) -> WindWaveLSTM:
    m = cfg["model"]
    return WindWaveLSTM(
        in_features=2,
        hidden_dim=int(m["hidden_dim"]),
        num_layers=int(m.get("num_layers", 1)),
    )
