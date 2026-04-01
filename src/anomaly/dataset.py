from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.config import resolve_path


class AnomalyNpzDataset(Dataset):
    """X: (N,T,2) 风速/波高；y: (N,2) 下一时刻目标。"""

    def __init__(self, path: str):
        p = resolve_path(path)
        d = np.load(p)
        self.x = d["X"]
        self.y = d["y"]

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.x[idx]).float()
        y = torch.from_numpy(self.y[idx]).float()
        return x, y
