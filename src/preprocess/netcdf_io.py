"""NetCDF 读取、插值与物理派生量（占位入口）。"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="NetCDF 预处理入口")
    parser.add_argument("--config", type=str, default="config/data.yaml", help="data.yaml 路径")
    args = parser.parse_args()
    raise NotImplementedError(
        f"待实现：读取 {args.config} 并输出至 data/processed/（见《A09-分阶段详细执行指南》阶段一）"
    )


if __name__ == "__main__":
    main()
