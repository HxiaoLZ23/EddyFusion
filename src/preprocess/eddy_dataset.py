"""涡旋数据集构建与图文对齐（占位）。"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/data.yaml")
    args = parser.parse_args()
    raise NotImplementedError(f"待实现：eddy 预处理，config={args.config}")


if __name__ == "__main__":
    main()
