"""
从 submission/figures/eddy_demo/eddy_demo.mp4 首帧构造演示用二维物理场 NPZ，
与 `src/eddy/multichannel_fuse.load_fused_bgr_from_npz` 默认键兼容（sla / vorticity / dtdy）。

另写入与风浪演示对齐的一维序列（非命题方实测，仅用于后续接入「风浪预警」页的占位），
键名：`demo_wind_observed`、`demo_wind_predicted`、`demo_wave_observed`、`demo_wave_predicted`。

用法（仓库根）::
    python scripts/gen_eddy_demo_physics_npz.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _laplacian_2d(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.zeros_like(z)
    out[1:-1, 1:-1] = (
        z[0:-2, 1:-1]
        + z[2:, 1:-1]
        + z[1:-1, 0:-2]
        + z[1:-1, 2:]
        - 4.0 * z[1:-1, 1:-1]
    )
    return out


def _sobel_y(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    gy = np.zeros_like(z)
    gy[1:-1, :] = (z[2:, :] - z[:-2, :]) * 0.5
    return gy


def _box_blur_sep(z: np.ndarray, radius: int = 3) -> np.ndarray:
    """可分离盒式平滑，仅 numpy。"""
    z = np.asarray(z, dtype=np.float64)
    k = np.ones(2 * radius + 1, dtype=np.float64) / float(2 * radius + 1)
    t = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, z)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, t)


def _synthetic_basin(h: int, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    sla = 0.25 * np.sin(x * 0.06) * np.cos(y * 0.09) + 0.08 * np.sin((x + y) * 0.03)
    lap = _laplacian_2d(sla)
    gy = _sobel_y(sla)
    return sla, lap, np.abs(gy)


def _ffprobe_wh(mp4: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        str(mp4),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    w, h = out.split("x")
    return int(w), int(h)


def _ffmpeg_first_frame_rgb(mp4: Path) -> np.ndarray:
    w, h = _ffprobe_wh(mp4)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(mp4),
        "-vframes",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    need = w * h * 3
    if len(raw) < need:
        raise RuntimeError(f"ffmpeg 输出过短: got {len(raw)} need {need}")
    return np.frombuffer(raw[:need], dtype=np.uint8).reshape(h, w, 3)


def _from_video_first_frame(mp4: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = _ffmpeg_first_frame_rgb(mp4)
    gray = rgb.mean(axis=2).astype(np.float64) / 255.0
    sla = _box_blur_sep(gray, radius=3)
    sla = (sla - float(sla.mean())) * 0.55
    lap = _laplacian_2d(sla)
    gy = _sobel_y(sla)
    return sla, lap, np.abs(gy)


def main() -> None:
    p = argparse.ArgumentParser(description="生成与 eddy_demo.mp4 配套的演示物理场 NPZ")
    p.add_argument(
        "--mp4",
        type=str,
        default="submission/figures/eddy_demo/eddy_demo.mp4",
        help="演示视频路径（取首帧推导伪物理场；缺失则退化为合成场）",
    )
    p.add_argument(
        "--out-npz",
        type=str,
        default="submission/figures/eddy_demo/eddy_demo_physics.npz",
        help="输出 NPZ 路径",
    )
    p.add_argument(
        "--out-meta",
        type=str,
        default="submission/figures/eddy_demo/demo_flow_meta.json",
        help="输出配对说明 JSON",
    )
    args = p.parse_args()

    mp4 = REPO_ROOT / args.mp4
    out_npz = REPO_ROOT / args.out_npz
    out_meta = REPO_ROOT / args.out_meta
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    sla: np.ndarray
    vorticity: np.ndarray
    dtdy: np.ndarray
    source_note: str
    if mp4.is_file():
        try:
            sla, vorticity, dtdy = _from_video_first_frame(mp4)
            source_note = f"first_frame_of:{mp4.relative_to(REPO_ROOT).as_posix()}"
        except Exception as e:
            try:
                w, h = _ffprobe_wh(mp4)
            except Exception:
                w, h = 320, 160
            print(f"[warn] 首帧推导失败（{e}），改用与视频分辨率一致的合成场 {w}x{h}")
            sla, vorticity, dtdy = _synthetic_basin(h, w)
            source_note = f"synthetic_fallback_same_resolution_as:{mp4.relative_to(REPO_ROOT).as_posix()}"
    else:
        sla, vorticity, dtdy = _synthetic_basin(160, 320)
        source_note = "synthetic_basin(mp4_missing)"

    # 与 InferenceService mock 时间轴长度 6 对齐，便于日后把序列写入 session → 风浪预警页
    demo_wind_observed = np.array([8.2, 9.1, 11.5, 14.2, 12.0, 10.1], dtype=np.float32)
    demo_wind_predicted = np.array([7.8, 8.0, 8.5, 9.0, 9.2, 9.0], dtype=np.float32)
    demo_wave_observed = np.array([1.2, 1.4, 1.9, 2.6, 2.2, 1.8], dtype=np.float32)
    demo_wave_predicted = np.array([1.1, 1.15, 1.2, 1.25, 1.3, 1.28], dtype=np.float32)

    np.savez_compressed(
        out_npz,
        sla=sla.astype(np.float32),
        vorticity=vorticity.astype(np.float32),
        dtdy=dtdy.astype(np.float32),
        demo_wind_observed=demo_wind_observed,
        demo_wind_predicted=demo_wind_predicted,
        demo_wave_observed=demo_wave_observed,
        demo_wave_predicted=demo_wave_predicted,
    )

    meta = {
        "paired_video": str(mp4.relative_to(REPO_ROOT)) if mp4.is_file() else None,
        "npz": str(out_npz.relative_to(REPO_ROOT)),
        "physics_source": source_note,
        "array_shapes": {"sla": list(sla.shape), "vorticity": list(vorticity.shape), "dtdy": list(dtdy.shape)},
        "demo_flow_steps": [
            "1) 涡旋识别页上传 eddy_demo.mp4",
            "2) 同页运行视频推理；可选上传本 NPZ 点击「对 NPZ 融合场运行单帧检测」",
            "3) 风浪预警页查看联动（无 demo_wind_* 时可能仍以 peak_score 代理）",
        ],
        "realtime_note": "实时系统当前以视频流为主；NC 全链路见离线系统与开发方向文档。",
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_npz}")
    print(f"wrote {out_meta}")
    print(f"shapes HxW={sla.shape}")


if __name__ == "__main__":
    main()
