#!/usr/bin/env python3
"""
涡旋物理场 CPU vs CuPy 对照（NetCDF 读取仍在 CPU）。

用法（仓库根）:
  python scripts/benchmark_eddy_physics_cpu_vs_cupy.py --nc-path app/data/nc_uploads/xxx.nc
  python scripts/benchmark_eddy_physics_cpu_vs_cupy.py --synthetic --h 720 --w 1440 --frames 32
  python scripts/benchmark_eddy_physics_cpu_vs_cupy.py --nc-path ... --time-indices 0,10,20

GPU 对照后端（--backend）:
  auto   — 先 CuPy，失败则用 PyTorch CUDA（推荐 Windows + 仅 torch+cu118）
  cupy   — 仅 CuPy（需 CUDA Toolkit 11.8 与 CUDA_PATH，见下方错误说明）
  torch  — 仅 PyTorch CUDA（与 YOLO 共用运行时，无需单独 Toolkit）
  all    — CuPy + Torch 都跑

CuPy 在 Windows 若报 nvrtc64_112_0.dll / complex.cuh：请用 --backend torch 或安装 CUDA Toolkit 11.8。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.eddy.stacked_physics import build_physics_stacked_hw8, relative_vorticity_and_okubo_weiss_from_uv
from src.eddy.stacked_physics_gpu import (
    relative_vorticity_and_okubo_weiss_from_uv_cupy,
    run_physics_cpu,
    run_physics_cupy,
)
from src.eddy.stacked_physics_torch import (
    relative_vorticity_and_okubo_weiss_from_uv_torch,
    run_physics_torch,
)
from src.utils.config import resolve_path
from src.utils.cupy_bootstrap import probe_cupy_runtime


def _timer(fn, *, warmup: int, repeats: int) -> dict[str, float]:
    for _ in range(max(0, warmup)):
        fn()
    samples: list[float] = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "ms_mean": float(arr.mean()),
        "ms_std": float(arr.std()),
        "ms_min": float(arr.min()),
        "ms_max": float(arr.max()),
    }


def _load_frames(nc_path: Path, indices: list[int]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    from src.eddy.nc_to_bgr import extract_triple_scalar_fields_from_netcdf

    out: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    for ti in indices:
        t0 = time.perf_counter()
        adt, u, v, _meta = extract_triple_scalar_fields_from_netcdf(nc_path, time_index=int(ti))
        read_ms = (time.perf_counter() - t0) * 1000.0
        out.append((adt, u, v, read_ms))
    return out


def _synthetic(h: int, w: int, n: int, seed: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        adt = rng.standard_normal((h, w), dtype=np.float64)
        u = rng.standard_normal((h, w), dtype=np.float64) * 0.1
        v = rng.standard_normal((h, w), dtype=np.float64) * 0.1
        out.append((adt, u, v, 0.0))
    return out


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def main() -> int:
    ap = argparse.ArgumentParser(description="涡旋物理场 CPU vs CuPy benchmark")
    ap.add_argument("--nc-path", type=str, default="", help="仓库相对或绝对 NC 路径")
    ap.add_argument("--synthetic", action="store_true", help="无 NC 时用随机场")
    ap.add_argument("--h", type=int, default=512)
    ap.add_argument("--w", type=int, default=512)
    ap.add_argument("--frames", type=int, default=16, help="模拟双路抽帧数")
    ap.add_argument("--time-indices", type=str, default="", help="逗号分隔时次，默认 0..frames-1")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--json-out", type=str, default="")
    ap.add_argument(
        "--backend",
        choices=("auto", "cupy", "torch", "all"),
        default="auto",
        help="GPU 对照后端",
    )
    args = ap.parse_args()

    if args.time_indices.strip():
        indices = [int(x.strip()) for x in args.time_indices.split(",") if x.strip()]
    else:
        indices = list(range(max(1, int(args.frames))))

    if args.synthetic or not args.nc_path.strip():
        frames_data = _synthetic(args.h, args.w, len(indices), seed=42)
        source = f"synthetic_{args.h}x{args.w}"
    else:
        nc = resolve_path(args.nc_path.strip())
        if not nc.is_file():
            print(f"NC 不存在: {nc}", file=sys.stderr)
            return 1
        frames_data = _load_frames(nc, indices)
        source = str(nc)

    read_ms = [x[3] for x in frames_data]
    adt0, u0, v0, _ = frames_data[0]
    shape = adt0.shape

    def cpu_zeta_ow():
        relative_vorticity_and_okubo_weiss_from_uv(u0, v0)

    def cpu_full():
        run_physics_cpu(adt0, u0, v0)

    want_cupy = args.backend in ("auto", "cupy", "all")
    want_torch = args.backend in ("auto", "torch", "all")

    cupy_probe = probe_cupy_runtime() if want_cupy else {"ok": False, "skipped": True}
    cupy_ok = bool(cupy_probe.get("ok"))

    torch_ok = False
    torch_err = ""
    if want_torch:
        try:
            import torch

            torch_ok = bool(torch.cuda.is_available())
            if not torch_ok:
                torch_err = "torch.cuda.is_available() is False"
        except Exception as e:
            torch_err = str(e)

    if args.backend == "cupy" and not cupy_ok:
        print(cupy_probe.get("fix_hint", ""), file=sys.stderr)
        print("CuPy 探测失败:", cupy_probe.get("error"), file=sys.stderr)

    report: dict[str, object] = {
        "source": source,
        "shape": list(shape),
        "n_frames": len(frames_data),
        "time_indices": indices,
        "nc_read_ms_mean": float(np.mean(read_ms)) if any(read_ms) else 0.0,
        "backend_requested": args.backend,
        "cupy_probe": cupy_probe,
        "torch_cuda": torch_ok,
    }

    cpu_z = _timer(cpu_zeta_ow, warmup=args.warmup, repeats=args.repeats)
    cpu_f = _timer(cpu_full, warmup=args.warmup, repeats=args.repeats)
    report["cpu_zeta_ow_single_frame"] = cpu_z
    report["cpu_full_hw8_single_frame"] = cpu_f

    z_cpu = relative_vorticity_and_okubo_weiss_from_uv(u0, v0)

    if cupy_ok:

        def cupy_zeta_ow():
            relative_vorticity_and_okubo_weiss_from_uv_cupy(u0, v0)

        def cupy_full():
            run_physics_cupy(adt0, u0, v0)

        z_gpu, ow_gpu = relative_vorticity_and_okubo_weiss_from_uv_cupy(u0, v0)
        report["cupy_zeta_max_abs_diff_vs_cpu"] = _max_abs_diff(z_cpu[0], z_gpu)
        report["cupy_ow_max_abs_diff_vs_cpu"] = _max_abs_diff(z_cpu[1], ow_gpu)
        cupy_z = _timer(cupy_zeta_ow, warmup=args.warmup, repeats=args.repeats)
        cupy_f = _timer(cupy_full, warmup=args.warmup, repeats=args.repeats)
        report["cupy_zeta_ow_single_frame"] = cupy_z
        report["cupy_full_hw8_single_frame"] = cupy_f
        report["speedup_cupy_zeta_ow"] = round(cpu_z["ms_mean"] / max(cupy_z["ms_mean"], 1e-6), 3)
        report["speedup_cupy_full_hw8"] = round(cpu_f["ms_mean"] / max(cupy_f["ms_mean"], 1e-6), 3)

    if torch_ok:

        def torch_zeta_ow():
            relative_vorticity_and_okubo_weiss_from_uv_torch(u0, v0)

        def torch_full():
            run_physics_torch(adt0, u0, v0)

        z_t, ow_t = relative_vorticity_and_okubo_weiss_from_uv_torch(u0, v0)
        report["torch_zeta_max_abs_diff_vs_cpu"] = _max_abs_diff(z_cpu[0], z_t)
        report["torch_ow_max_abs_diff_vs_cpu"] = _max_abs_diff(z_cpu[1], ow_t)
        torch_z = _timer(torch_zeta_ow, warmup=args.warmup, repeats=args.repeats)
        torch_f = _timer(torch_full, warmup=args.warmup, repeats=args.repeats)
        report["torch_zeta_ow_single_frame"] = torch_z
        report["torch_full_hw8_single_frame"] = torch_f
        report["speedup_torch_zeta_ow"] = round(cpu_z["ms_mean"] / max(torch_z["ms_mean"], 1e-6), 3)
        report["speedup_torch_full_hw8"] = round(cpu_f["ms_mean"] / max(torch_f["ms_mean"], 1e-6), 3)

    if cupy_ok or torch_ok:

        def batch_cpu():
            for adt, u, v, _r in frames_data:
                run_physics_cpu(adt, u, v)

        batch_cpu_t = _timer(batch_cpu, warmup=0, repeats=max(1, args.repeats // 2))
        report["cpu_batch_physics_all_frames"] = batch_cpu_t

        if cupy_ok:

            def batch_cupy():
                for adt, u, v, _r in frames_data:
                    run_physics_cupy(adt, u, v)

            batch_cupy_t = _timer(batch_cupy, warmup=0, repeats=max(1, args.repeats // 2))
            report["cupy_batch_physics_all_frames"] = batch_cupy_t
            report["speedup_cupy_batch_physics"] = round(
                batch_cpu_t["ms_mean"] / max(batch_cupy_t["ms_mean"], 1e-6), 3
            )

        if torch_ok:

            def batch_torch():
                for adt, u, v, _r in frames_data:
                    run_physics_torch(adt, u, v)

            batch_torch_t = _timer(batch_torch, warmup=0, repeats=max(1, args.repeats // 2))
            report["torch_batch_physics_all_frames"] = batch_torch_t
            report["speedup_torch_batch_physics"] = round(
                batch_cpu_t["ms_mean"] / max(batch_torch_t["ms_mean"], 1e-6), 3
            )
    elif want_torch and not torch_ok:
        report["torch_error"] = torch_err

    nz = cpu_z["ms_mean"]
    nf = cpu_f["ms_mean"]
    lines = [
        f"数据源: {source}",
        f"格点: {shape[0]} x {shape[1]}，帧数对照: {len(indices)}",
        f"NC 读取(仅真实文件): 均值 {report['nc_read_ms_mean']:.2f} ms/帧",
        "",
        f"CPU  ζ/OW 单帧: {nz:.2f} ± {cpu_z['ms_std']:.2f} ms",
        f"CPU  完整 HW8: {nf:.2f} ± {cpu_f['ms_std']:.2f} ms",
    ]
    if cupy_ok:
        cz = report["cupy_zeta_ow_single_frame"]["ms_mean"]  # type: ignore[index]
        cf = report["cupy_full_hw8_single_frame"]["ms_mean"]  # type: ignore[index]
        lines.extend(
            [
                f"CuPy ζ/OW 单帧: {cz:.2f} ms（加速 {report['speedup_cupy_zeta_ow']}x）",
                f"CuPy 完整 HW8: {cf:.2f} ms（加速 {report['speedup_cupy_full_hw8']}x）",
                f"CuPy 多帧 batch: {report.get('speedup_cupy_batch_physics', 'n/a')}x",
            ]
        )
    elif want_cupy:
        lines.append(f"\nCuPy 不可用: {cupy_probe.get('error', 'unknown')}")
        if cupy_probe.get("fix_hint"):
            lines.append(str(cupy_probe["fix_hint"]))

    if torch_ok:
        tz = report["torch_zeta_ow_single_frame"]["ms_mean"]  # type: ignore[index]
        tf = report["torch_full_hw8_single_frame"]["ms_mean"]  # type: ignore[index]
        lines.extend(
            [
                f"Torch ζ/OW 单帧: {tz:.2f} ms（加速 {report['speedup_torch_zeta_ow']}x）",
                f"Torch 完整 HW8: {tf:.2f} ms（加速 {report['speedup_torch_full_hw8']}x）",
                f"Torch 多帧 batch: {report.get('speedup_torch_batch_physics', 'n/a')}x",
            ]
        )

    best_sp = 1.0
    if cupy_ok:
        best_sp = max(best_sp, float(report.get("speedup_cupy_full_hw8", 1.0)))  # type: ignore[arg-type]
    if torch_ok:
        best_sp = max(best_sp, float(report.get("speedup_torch_full_hw8", 1.0)))  # type: ignore[arg-type]

    if cupy_ok or torch_ok:
        worth = best_sp >= 1.3 and nf > 5.0
        lines.append(
            "\n建议: "
            + (
                f"值得评估 GPU 物理场（最佳 HW8 加速约 {best_sp:.2f}x）。"
                if worth
                else "暂不建议优先 GPU 化物理场（加速有限；瓶颈更可能在 YOLO/NC I/O）。"
            )
        )
    elif want_torch and not torch_ok:
        lines.append(f"\nPyTorch CUDA 不可用: {torch_err}")

    print("\n".join(lines))
    print("\n--- JSON ---")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
