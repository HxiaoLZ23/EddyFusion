"""水文滑动窗口样本构建（占位）。"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/data.yaml")
    args = parser.parse_args()
    raise NotImplementedError(f"待实现：hydro 预处理，config={args.config}")


if __name__ == "__main__":
    main()
