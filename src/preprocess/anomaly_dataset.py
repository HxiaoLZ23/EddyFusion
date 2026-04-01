from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.utils.config import load_yaml, project_root, resolve_path


def generate_synthetic_anomaly(config_path: str | Path) -> None:
    cfg = load_yaml(config_path)
    rng = np.random.default_rng(int(cfg["meta"]["seed"]))
    paths = cfg["paths"]
    win = int(cfg["data"]["window_hours"])
    n_train, n_val, n_test = 400, 80, 80

    def make(n: int) -> tuple[np.ndarray, np.ndarray]:
        # 特征：风速、波高；目标为下一时刻
        x = rng.standard_normal((n, win, 2)).astype(np.float32)
        # 累积随机游走近似时序
        x = np.cumsum(x, axis=1)
        y = rng.standard_normal((n, 2)).astype(np.float32) * 0.1 + x[:, -1] * 0.9
        return x, y

    key_map = {
        "train": paths["train_sequences"],
        "val": paths["val_sequences"],
        "test": paths["test_sequences"],
    }
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        xp = resolve_path(key_map[split])
        x, y = make(n)
        xp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(xp, X=x, y=y)
        print(f"wrote {xp} X={x.shape} y={y.shape}")

    idx = {
        "source": "synthetic",
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
    }
    tip = resolve_path(paths["typhoon_index"])
    tip.parent.mkdir(parents=True, exist_ok=True)
    with tip.open("w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)
    print(f"wrote {tip}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/anomaly.yaml")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    if args.synthetic:
        generate_synthetic_anomaly(project_root() / args.config)
        return
    raise NotImplementedError("命题方时序预处理待实现，可先 --synthetic")


if __name__ == "__main__":
    main()
