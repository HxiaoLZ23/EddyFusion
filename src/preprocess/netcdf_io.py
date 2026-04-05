from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.utils.config import load_yaml, project_root, resolve_path


def _resolve_path_arg(root: Path, user: str) -> Path:
    """将命令行路径解析为绝对 Path（支持相对仓库根目录）。"""
    text = user.strip().strip('"').strip("'")
    p = Path(text).expanduser()
    if p.is_absolute():
        return p
    return (root / p).resolve()


def _find_nc_by_basename(raw_root: Path, basename: str, limit: int = 15) -> list[Path]:
    """在 raw_root 下按文件名匹配（用于路径写错时的提示）。"""
    if not raw_root.is_dir():
        return []
    out: list[Path] = []
    for cand in raw_root.rglob(basename):
        if cand.is_file():
            out.append(cand)
            if len(out) >= limit:
                break
    return out


def _list_subdirs(p: Path, max_items: int = 30) -> str:
    if not p.is_dir():
        return f"(不存在: {p})"
    names = sorted(x.name for x in p.iterdir() if x.is_dir())
    if not names:
        return "(无子目录)"
    tail = f"\n  … 共 {len(names)} 个" if len(names) > max_items else ""
    shown = names[:max_items]
    return "\n  - " + "\n  - ".join(shown) + tail


def _iter_nc_files(raw_root: Path) -> list[Path]:
    if not raw_root.is_dir():
        return []
    out: list[Path] = []
    for pat in ("*.nc", "*.nc4", "*.cdf"):
        out.extend(sorted(raw_root.rglob(pat)))
    return out


def inspect_file(path: Path) -> None:
    import xarray as xr

    # 显式 engine，避免部分环境下后端推断异常
    ds = xr.open_dataset(path, engine="netcdf4")
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
        p = _resolve_path_arg(root, args.path)
        if not p.is_file():
            print(f"文件不存在: {p}", file=sys.stderr)
            if raw_root.is_dir():
                print(f"\n「{raw_root.name}」下的一级子目录为：{_list_subdirs(raw_root)}", file=sys.stderr)
                print(
                    "\n提示：若目录名为「水文要素预测」而非「海域要素预测」（或相反），请改正 --path 中的文件夹名。",
                    file=sys.stderr,
                )
            base = Path(args.path).name
            if base.endswith(".nc") and raw_root.is_dir():
                hits = _find_nc_by_basename(raw_root, base)
                if hits:
                    print(f"\n在「{raw_root}」下找到同名文件（可改用其一）：", file=sys.stderr)
                    for h in hits[:10]:
                        print(f"  {h}", file=sys.stderr)
            sys.exit(1)
        inspect_file(p)
        return

    if not raw_root.is_dir():
        print(f"原始数据目录不存在: {raw_root}\n请检查 config/data.yaml 中 paths.raw_root（当前命题方数据为「服创数据集」）。", file=sys.stderr)
        sys.exit(2)

    files = _iter_nc_files(raw_root)
    if not files:
        print(
            f"未在 {raw_root} 发现 NetCDF（*.nc / *.nc4）。请检查命题方数据是否已解压或使用 --path。",
            file=sys.stderr,
        )
        sys.exit(2)

    last_err: Exception | None = None
    for p in files:
        try:
            inspect_file(p)
            if len(files) > 1:
                print(f"\n（共 {len(files)} 个文件，已打开第一个可读文件；可用 --path 指定其它文件）")
            return
        except Exception as e:
            last_err = e
            print(f"[跳过] 无法打开: {p}\n  原因: {e}", file=sys.stderr)
    print(f"所有候选文件均无法打开。最后一个错误: {last_err}", file=sys.stderr)
    sys.exit(3)


if __name__ == "__main__":
    main()
