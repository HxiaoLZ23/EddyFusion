from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.config import resolve_path


class HydroNpzDataset(Dataset):
    """X: (N,T_in,H,W,C_in), y: (N,T_out,H,W,C_out)"""

    def __init__(self, x_path: str | Path, y_path: str | Path):
        xp = resolve_path(x_path)
        yp = resolve_path(y_path)
        dx = np.load(xp)
        dy = np.load(yp)
        self.x = dx["X"] if "X" in dx else dx[dx.files[0]]
        self.y = dy["y"] if "y" in dy else dy[dy.files[0]]
        assert self.x.shape[0] == self.y.shape[0], "X/y 样本数不一致"

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.x[idx]).float()
        y = torch.from_numpy(self.y[idx]).float()
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.isfinite(y).all():
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        # NHWC -> NCHW for time dimension kept: T H W C -> T C H W
        x = x.permute(0, 3, 1, 2).contiguous()
        y = y.permute(0, 3, 1, 2).contiguous()
        return x, y
