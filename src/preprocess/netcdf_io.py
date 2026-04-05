from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

from src.utils.config import load_yaml, project_root, resolve_path


def _normalize_user_path(user: str) -> str:
    """统一为正斜杠，避免 Windows 下反斜杠与 Python 字符串转义问题。"""
    return user.strip().strip('"').strip("'").replace("\\", "/")


def _resolve_path_arg(root: Path, user: str) -> Path:
    """相对路径优先相对「仓库根」，其次「当前工作目录」（与 Cursor 运行 cwd 一致时可用）。"""
    normalized = _normalize_user_path(user)
    p = Path(normalized)
    if p.is_absolute():
        return p.resolve()
    for base in (root, Path.cwd()):
        cand = (base / p).resolve()
        if cand.is_file():
            return cand
    return (root / p).resolve()


def _win_extended_path(path: Path) -> str:
    """Windows 下为含中文/长路径添加 \\\\?\\ 前缀，便于底层 C 库打开。"""
    p = path.resolve()
    s = str(p)
    if os.name != "nt":
        return s
    if s.startswith("\\\\?\\"):
        return s
    if s.startswith("\\\\"):
        return "\\\\?\\UNC\\" + s[2:].lstrip("\\")
    return "\\\\?\\" + s


def _win_short_path(path: Path) -> Path | None:
    """尝试获取 8.3 短路径，规避部分 netCDF4 对 Unicode 路径的问题。"""
    if os.name != "nt":
        return None
    try:
        import ctypes

        buf = ctypes.create_unicode_buffer(32768)
        ab = str(path.resolve())
        n = ctypes.windll.kernel32.GetShortPathNameW(ab, buf, len(buf))
        if n and Path(buf.value).is_file():
            return Path(buf.value)
    except Exception:
        return None
    return None


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


def open_netcdf_dataset(path: Path) -> tuple[object, Path | None]:
    """
    在 Windows + 中文路径下，netCDF4 C API 常无法打开含 Unicode 的路径。
    依次尝试：8.3 短路径、普通绝对路径、\\\\?\\ 扩展路径；仍失败则复制到临时英文路径再打开。

    返回 (xarray.Dataset, 若为临时文件则 Path 以便关闭后删除，否则 None)。
    """
    import xarray as xr

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"不是可读文件: {path}")

    tries: list[str] = []
    if os.name == "nt":
        short = _win_short_path(path)
        if short is not None:
            tries.append(str(short))
    tries.append(str(path))
    if os.name == "nt":
        tries.append(_win_extended_path(path))

    seen: set[str] = set()
    last_err: Exception | None = None
    for s in tries:
        if not s or s in seen:
            continue
        seen.add(s)
        try:
            return xr.open_dataset(s, engine="netcdf4"), None
        except Exception as e:
            last_err = e
            continue

    # 最后手段：复制到仅 ASCII 的临时路径（大文件会慢；关闭 Dataset 后由调用方删除）
    tmp_nc = Path(tempfile.gettempdir()) / f"eddyfusion_nc_{uuid.uuid4().hex}.nc"
    try:
        shutil.copy2(path, tmp_nc)
        ds = xr.open_dataset(str(tmp_nc), engine="netcdf4")
        return ds, tmp_nc
    except Exception as e:
        if tmp_nc.exists():
            try:
                tmp_nc.unlink()
            except OSError:
                pass
        raise OSError(
            f"无法用 netcdf4 打开（含短路径/扩展路径/临时副本）: {path}\n最后错误: {last_err!r}; 副本错误: {e!r}"
        ) from e


def inspect_file(path: Path) -> None:
    ds, tmp_copy = open_netcdf_dataset(path)
    try:
        print(f"=== {path} ===")
        if tmp_copy is not None:
            print(f"(已通过临时英文路径打开: {tmp_copy})", flush=True)
        print(ds)
        print("dims:", dict(ds.sizes))
    finally:
        ds.close()
        if tmp_copy is not None:
            try:
                tmp_copy.unlink(missing_ok=True)  # type: ignore[arg-type]
            except OSError:
                pass


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
            print(f"文件不存在（解析为）: {p}", file=sys.stderr)
            print(f"仓库根目录 project_root() = {root}", file=sys.stderr)
            print(f"当前工作目录 cwd = {Path.cwd()}", file=sys.stderr)
            if raw_root.is_dir():
                print(f"\n「{raw_root.name}」下的一级子目录为：{_list_subdirs(raw_root)}", file=sys.stderr)
                print(
                    "\n提示：若目录名为「水文要素预测」与「海域要素预测」不一致，请改正 --path；"
                    "也可在仓库根下执行，或使用正斜杠：服创数据集/海域要素预测/1994/19940101.nc",
                    file=sys.stderr,
                )
            base = Path(_normalize_user_path(args.path)).name
            if base.endswith(".nc") and raw_root.is_dir():
                hits = _find_nc_by_basename(raw_root, base)
                if hits:
                    print(f"\n在「{raw_root}」下找到同名文件（可改用其一）：", file=sys.stderr)
                    for h in hits[:10]:
                        print(f"  {h}", file=sys.stderr)
            sys.exit(1)
        try:
            inspect_file(p)
        except OSError as e:
            print(f"打开失败: {e}", file=sys.stderr)
            print(
                "若文件在资源管理器中存在，多为 Windows 中文路径与 netCDF4 兼容问题；"
                "已自动尝试扩展路径与短路径。仍失败可将 .nc 复制到纯英文路径再试。",
                file=sys.stderr,
            )
            sys.exit(4)
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
