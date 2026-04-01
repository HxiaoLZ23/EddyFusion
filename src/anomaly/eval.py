"""异常检测评估指标输出。"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/anomaly.yaml")
    parser.add_argument("--ckpt", type=str, default="outputs/anomaly/best.pt")
    args = parser.parse_args()
    raise NotImplementedError(f"待实现：anomaly eval，config={args.config} ckpt={args.ckpt}")


if __name__ == "__main__":
    main()
