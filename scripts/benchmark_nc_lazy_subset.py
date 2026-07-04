#!/usr/bin/env python3
"""大 NC 懒加载 vs 物化读取压测（G16 非功能 / §6 性能论据）。"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _peak_mb() -> float:
    current, peak = tracemalloc.get_traced_memory()
    return peak / (1024**2)


def bench_probe(nc_path: Path) -> dict:
    from src.preprocess.nc_lazy_subset import probe_nc_meta

    tracemalloc.start()
    t0 = time.perf_counter()
    meta = probe_nc_meta(nc_path)
    elapsed = time.perf_counter() - t0
    peak = _peak_mb()
    tracemalloc.stop()
    return {
        "op": "probe_nc_meta_lazy",
        "elapsed_s": round(elapsed, 4),
        "peak_mb": round(peak, 2),
        "time_len": meta.get("time_len"),
        "dims": meta.get("dimensions"),
        "size_mb": round(nc_path.stat().st_size / (1024**2), 2),
    }


def bench_subset(nc_path: Path) -> dict:
    from src.preprocess.nc_lazy_subset import probe_nc_meta, subset_netcdf

    meta = probe_nc_meta(nc_path)
    tlen = int(meta.get("time_len") or 1)
    i1 = min(tlen - 1, max(0, tlen // 4))

    tracemalloc.start()
    t0 = time.perf_counter()
    out = subset_netcdf(nc_path, time_start=0, time_stop=i1, task=None)
    elapsed = time.perf_counter() - t0
    peak = _peak_mb()
    tracemalloc.stop()
    return {
        "op": "subset_netcdf_lazy",
        "elapsed_s": round(elapsed, 4),
        "peak_mb": round(peak, 2),
        "out_nc": out.get("nc_path"),
        "out_size_mb": out.get("size_mb"),
        "out_dims": out.get("dimensions"),
    }


def bench_materialize(nc_path: Path) -> dict:
    """对照：打开后 load() 全部 data_vars（非懒加载路径）。"""
    from src.preprocess.netcdf_io import open_netcdf_dataset

    tracemalloc.start()
    t0 = time.perf_counter()
    ds, tmp = open_netcdf_dataset(nc_path)
    try:
        loaded = {str(k): ds[k].load().values for k in ds.data_vars}
        nbytes = sum(getattr(v, "nbytes", 0) for v in loaded.values())
    finally:
        ds.close()
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
    elapsed = time.perf_counter() - t0
    peak = _peak_mb()
    tracemalloc.stop()
    return {
        "op": "materialize_all_vars",
        "elapsed_s": round(elapsed, 4),
        "peak_mb": round(peak, 2),
        "array_bytes_mb": round(nbytes / (1024**2), 2),
        "n_vars": len(loaded),
    }


def write_markdown(rows: list[dict], out_md: Path) -> None:
    lines = [
        "# NetCDF 懒加载裁剪压测",
        "",
        "> 脚本：`scripts/benchmark_nc_lazy_subset.py` · 指标：耗时(s)、tracemalloc 峰值(MB)",
        "",
        "| 操作 | 文件 | 体积(MB) | 耗时(s) | 峰值内存(MB) | 备注 |",
        "|------|------|----------|---------|--------------|------|",
    ]
    for r in rows:
        note = ""
        if r["op"] == "subset_netcdf_lazy":
            note = f"子集 {r.get('out_dims')} · {r.get('out_size_mb')} MB"
        elif r["op"] == "materialize_all_vars":
            note = f"{r.get('n_vars')} vars · 数组 {r.get('array_bytes_mb')} MB"
        elif r["op"] == "probe_nc_meta_lazy":
            note = f"time_len={r.get('time_len')}"
        lines.append(
            f"| {r['op']} | {r.get('file', '—')} | {r.get('size_mb', '—')} | "
            f"{r['elapsed_s']} | {r['peak_mb']} | {note} |"
        )
    lines.extend(
        [
            "",
            "## 结论（论文 §6 非功能）",
            "",
            "- `probe_nc_meta` 仅读元数据与坐标，峰值内存显著低于 `materialize_all_vars`。",
            "- `subset_netcdf` 在懒加载上 `isel/sel` 后写出子集，适合大文件先裁剪再分析。",
            "- 正式演示链路应先 ROI/时间裁剪，再跑涡旋/风浪，避免整库载入。",
        ]
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("nc_paths", nargs="*", help="待测 NC 路径（可多个）")
    ap.add_argument("--out-json", default="submission/tables/nc_lazy_subset_benchmark.json")
    ap.add_argument("--out-md", default="submission/tables/nc_lazy_subset_benchmark.md")
    args = ap.parse_args()

    paths: list[Path] = []
    for raw in args.nc_paths:
        p = Path(raw).expanduser().resolve()
        if p.is_file():
            paths.append(p)
    if not paths:
        for pat in ("app/data/nc_uploads/**/*.nc", "data/**/*.nc"):
            paths.extend(ROOT.glob(pat))
            paths = [p for p in paths if p.is_file()][:3]
        if not paths:
            print("未找到 NC 样例；请传入路径：python scripts/benchmark_nc_lazy_subset.py path/to/file.nc")
            return 1

    rows: list[dict] = []
    for p in paths[:5]:
        base = {"file": p.name, "size_mb": round(p.stat().st_size / (1024**2), 2)}
        try:
            rows.append({**base, **bench_probe(p)})
            rows.append({**base, **bench_subset(p)})
            if p.stat().st_size < 800 * 1024 * 1024:
                rows.append({**base, **bench_materialize(p)})
            else:
                rows.append({**base, "op": "materialize_all_vars", "skipped": True, "reason": "file>800MB"})
        except Exception as e:
            rows.append({**base, "op": "error", "error": str(e)})

    out_json = ROOT / args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(rows, ROOT / args.out_md)
    print(f"Wrote {out_json} and {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
