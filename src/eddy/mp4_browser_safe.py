"""将 BGR 帧序列编码为浏览器可播的 MP4（H.264 + yuv420p + faststart）。

OpenCV 默认 mp4v 在 Chrome/Edge 中常无法内嵌播放；本模块在检测到 ffmpeg 时优先走 libx264。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def ffmpeg_available() -> str | None:
    return shutil.which("ffmpeg")


def encode_bgr_frames_to_browser_mp4(
    frames: list[np.ndarray],
    *,
    fps: float,
    out_path: Path,
    crf: str = "23",
) -> tuple[bool, str]:
    """
    使用 ffmpeg rawvideo → libx264。帧须为 uint8 H×W×3 BGR，形状一致。

    返回 (成功, 说明或错误摘要)。
    """
    exe = ffmpeg_available()
    if not exe:
        return False, "未找到 ffmpeg（未在 PATH 中），无法输出 H.264。"

    if not frames:
        return False, "无帧"

    h0, w0 = int(frames[0].shape[0]), int(frames[0].shape[1])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str | Any] = [
        exe,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{w0}x{h0}",
        "-r",
        str(max(0.5, float(fps))),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.stdin is None:
            return False, "无法打开 ffmpeg stdin"
        for fr in frames:
            arr = np.asarray(fr, dtype=np.uint8)
            if int(arr.shape[0]) != h0 or int(arr.shape[1]) != w0:
                proc.kill()
                return False, f"帧尺寸不一致: 期望 {h0}x{w0}, 得到 {arr.shape[:2]}"
            proc.stdin.write(np.ascontiguousarray(arr).tobytes())
        proc.stdin.close()
        _, err_b = proc.communicate(timeout=600)
        code = proc.returncode
        if code != 0:
            msg = (err_b or b"").decode("utf-8", errors="replace")[-800:]
            return False, f"ffmpeg 退出码 {code}: {msg}"
        if not out_path.is_file() or out_path.stat().st_size < 32:
            return False, "写出文件缺失或过小"
        return True, "H.264/yuv420p（浏览器可播）"
    except Exception as e:
        return False, str(e)
