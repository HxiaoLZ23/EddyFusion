"""风-浪任务训练入口。"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/anomaly.yaml")
    args = parser.parse_args()
    raise NotImplementedError(f"待实现：src.anomaly.train，config={args.config}")


if __name__ == "__main__":
    main()
