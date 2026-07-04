#!/usr/bin/env python3
"""MP4 编码 CPU (libx264) vs GPU (h264_nvenc) 对照。仓库根执行。"""

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

from src.eddy.mp4_browser_safe import encode_bgr_frames_to_browser_mp4, mp4_encoder_status


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--fps", type=float, default=1.0)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (args.h, args.w, 3), dtype=np.uint8) for _ in range(args.frames)]
    out_dir = _ROOT / "app" / "data" / "eddy_preview" / "_bench_encode"
    out_dir.mkdir(parents=True, exist_ok=True)

    status = mp4_encoder_status()
    print("编码器探测:", json.dumps(status, ensure_ascii=False, indent=2))

    results = {}
    for mode in ("libx264", "h264_nvenc"):
        out = out_dir / f"bench_{mode}.mp4"
        t0 = time.perf_counter()
        ok, msg = encode_bgr_frames_to_browser_mp4(
            frames,
            fps=args.fps,
            out_path=out,
            encoder=mode,
            allow_nvenc_fallback=False,
        )
        ms = (time.perf_counter() - t0) * 1000.0
        results[mode] = {"ok": ok, "ms": ms, "msg": msg, "size_mb": out.stat().st_size / 1e6 if ok and out.is_file() else 0}
        print(f"{mode}: ok={ok} {ms:.0f} ms — {msg}")

    if results.get("libx264", {}).get("ok") and results.get("h264_nvenc", {}).get("ok"):
        sp = results["libx264"]["ms"] / max(results["h264_nvenc"]["ms"], 1e-6)
        print(f"\nNVENC 相对 libx264 加速约 {sp:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
