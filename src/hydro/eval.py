"""水文评估：NRMSE 等，写入 metrics_summary.json。"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/hydro.yaml")
    parser.add_argument("--ckpt", type=str, default="outputs/hydro/best.pt")
    args = parser.parse_args()
    raise NotImplementedError(f"待实现：hydro eval，config={args.config} ckpt={args.ckpt}")


if __name__ == "__main__":
    main()
