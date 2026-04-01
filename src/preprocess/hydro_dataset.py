from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.utils.config import load_yaml, project_root, resolve_path


def generate_synthetic(config_path: str | Path) -> None:
    """写入 X_*.npz / y_*.npz，形状与 config 中 input_steps、output_steps、grid_shape、特征数一致。"""
    cfg = load_yaml(config_path)
    root = project_root()
    paths = cfg["paths"]
    d = cfg["data"]
    tin = int(d["input_steps"])
    tout = int(d["output_steps"])
    gh, gw = int(d["grid_shape"][0]), int(d["grid_shape"][1])
    c_in = len(d["input_features"])
    c_out = len(d["target_features"])
    rng = np.random.default_rng(int(cfg["meta"]["seed"]))

    n_train, n_val, n_test = 80, 16, 16

    def make_batch(n: int) -> tuple[np.ndarray, np.ndarray]:
        x = rng.standard_normal((n, tin, gh, gw, c_in)).astype(np.float32)
        y = rng.standard_normal((n, tout, gh, gw, c_out)).astype(np.float32)
        # 弱相关：让 y 与 x 最后若干步略相关，便于 loss 下降
        y += 0.05 * x[:, -tout:, :, :, :c_out]
        return x, y

    splits = {
        ("train", n_train),
        ("val", n_val),
        ("test", n_test),
    }
    for name, n in splits:
        x, y = make_batch(n)
        xp = resolve_path(paths[f"{name}_data"])
        yp = resolve_path(paths[f"{name}_label"])
        xp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(xp, X=x)
        np.savez_compressed(yp, y=y)
        print(f"wrote {xp.name} / {yp.name} shape X={x.shape} y={y.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="水文样本构建")
    parser.add_argument("--config", type=str, default="config/hydro.yaml")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="生成与 config 一致的合成 npz（无命题方数据时使用）",
    )
    args = parser.parse_args()
    if args.synthetic:
        generate_synthetic(project_root() / args.config)
        return
    raise NotImplementedError(
        "命题方 NetCDF 预处理待接入：请使用 --synthetic 或实现 NetCDF 滑动窗口逻辑"
    )


if __name__ == "__main__":
    main()
