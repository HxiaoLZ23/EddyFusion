"""将 BGR 帧序列编码为浏览器可播的 MP4（H.264 + yuv420p + faststart）。

OpenCV 默认 mp4v 在 Chrome/Edge 中常无法内嵌播放。
编码器：``EDDY_MP4_ENCODER`` = ``auto`` | ``cpu`` | ``nvenc``（``cpu`` 等同 libx264）。
``auto``：检测到 ffmpeg ``h264_nvenc`` 时优先硬件编码，失败则回退 libx264。
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

import numpy as np

EncoderMode = Literal["libx264", "h264_nvenc"]


def ffmpeg_available() -> str | None:
    return shutil.which("ffmpeg")


def ffmpeg_has_nvenc(exe: str | None = None) -> bool:
    ff = exe or ffmpeg_available()
    if not ff:
        return False
    try:
        proc = subprocess.run(
            [ff, "-hide_banner", "-encoders"],
            capture_output=True,
            timeout=15,
            check=False,
        )
        out = (proc.stdout or b"") + (proc.stderr or b"")
        text = out.decode("utf-8", errors="replace").lower()
        return "h264_nvenc" in text or "hevc_nvenc" in text
    except (OSError, subprocess.TimeoutExpired):
        return False


def resolve_mp4_encoder(*, prefer: str | None = None) -> EncoderMode:
    """
    解析 ``EDDY_MP4_ENCODER`` / ``prefer``：
    - ``cpu`` / ``libx264`` → libx264
    - ``nvenc`` / ``h264_nvenc`` → h264_nvenc（无 NVENC 时仍返回 nvenc，由调用方回退）
    - ``auto`` 或未设置：有 NVENC 则 nvenc，否则 libx264
    """
    raw = (prefer or os.environ.get("EDDY_MP4_ENCODER", "auto")).strip().lower()
    if raw in ("cpu", "libx264", "x264"):
        return "libx264"
    if raw in ("nvenc", "h264_nvenc", "gpu"):
        return "h264_nvenc"
    if ffmpeg_has_nvenc():
        return "h264_nvenc"
    return "libx264"


def mp4_encoder_status() -> dict[str, Any]:
    ff = ffmpeg_available()
    enc = resolve_mp4_encoder()
    return {
        "ffmpeg": ff,
        "env_EDDY_MP4_ENCODER": os.environ.get("EDDY_MP4_ENCODER", "auto"),
        "resolved_encoder": enc,
        "nvenc_available": bool(ff and ffmpeg_has_nvenc(ff)),
    }


def encode_bgr_frames_to_browser_mp4(
    frames: list[np.ndarray],
    *,
    fps: float,
    out_path: Path,
    crf: str = "23",
    encoder: str | None = None,
    allow_nvenc_fallback: bool = True,
) -> tuple[bool, str]:
    """
    使用 ffmpeg rawvideo → H.264。帧须为 uint8 H×W×3 BGR，形状一致。

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

    mode = resolve_mp4_encoder(prefer=encoder)
    ok, msg = _encode_with_mode(
        exe, frames, fps=fps, out_path=out_path, crf=crf, mode=mode, h0=h0, w0=w0
    )
    if ok:
        return True, msg
    if mode == "h264_nvenc" and allow_nvenc_fallback:
        ok2, msg2 = _encode_with_mode(
            exe,
            frames,
            fps=fps,
            out_path=out_path,
            crf=crf,
            mode="libx264",
            h0=h0,
            w0=w0,
        )
        if ok2:
            return True, f"{msg2}（NVENC 失败已回退 CPU: {msg[:200]}）"
        return False, f"NVENC 失败: {msg}; 回退 libx264 仍失败: {msg2}"
    return False, msg


def _encode_with_mode(
    exe: str,
    frames: list[np.ndarray],
    *,
    fps: float,
    out_path: Path,
    crf: str,
    mode: EncoderMode,
    h0: int,
    w0: int,
) -> tuple[bool, str]:
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
    ]
    if mode == "h264_nvenc":
        cmd.extend(
            [
                "-c:v",
                "h264_nvenc",
                "-preset",
                os.environ.get("EDDY_MP4_NVENC_PRESET", "p4"),
                "-rc",
                "vbr",
                "-cq",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ]
        )
        label = "H.264 NVENC/yuv420p（GPU 硬件编码）"
    else:
        cmd.extend(
            [
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                str(crf),
                "-movflags",
                "+faststart",
            ]
        )
        label = "H.264 libx264/yuv420p（CPU 编码）"

    cmd.append(str(out_path))

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
            return False, f"ffmpeg({mode}) 退出码 {code}: {msg}"
        if not out_path.is_file() or out_path.stat().st_size < 32:
            return False, "写出文件缺失或过小"
        return True, label
    except Exception as e:
        return False, str(e)
