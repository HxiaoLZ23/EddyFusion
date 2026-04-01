from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.utils.config import load_yaml, project_root, resolve_path


def _iter_nc_files(raw_root: Path) -> list[Path]:
    if not raw_root.is_dir():
        return []
    out: list[Path] = []
    for pat in ("*.nc", "*.nc4", "*.cdf"):
        out.extend(sorted(raw_root.rglob(pat)))
    return out


def inspect_file(path: Path) -> None:
    import xarray as xr

    ds = xr.open_dataset(path)
    print(f"=== {path} ===")
    print(ds)
    print("dims:", dict(ds.sizes))
    ds.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="NetCDF 检查 / 预处理入口")
    parser.add_argument("--config", type=str, default="config/data.yaml", help="data.yaml 路径")
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="指定单个 .nc 文件；默认扫描 data/raw 下首个 nc",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    root = project_root()
    raw_root = resolve_path(cfg.get("paths", {}).get("raw_root", "data/raw"))

    if args.path:
        p = Path(args.path)
        if not p.is_absolute():
            p = root / p
        if not p.is_file():
            print(f"文件不存在: {p}", file=sys.stderr)
            sys.exit(1)
        inspect_file(p)
        return

    files = _iter_nc_files(raw_root)
    if not files:
        print(
            f"未在 {raw_root} 发现 NetCDF（*.nc / *.nc4）。请将命题方数据放入该目录或使用 --path。",
            file=sys.stderr,
        )
        sys.exit(2)
    inspect_file(files[0])
    if len(files) > 1:
        print(f"\n（共 {len(files)} 个文件，仅展示第一个；可用 --path 指定）")


if __name__ == "__main__":
    main()
