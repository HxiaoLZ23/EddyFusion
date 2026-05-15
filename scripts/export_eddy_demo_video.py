from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# 允许从 scripts 目录直接运行时导入 src/*
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_yaml, resolve_path
from src.preprocess.netcdf_io import open_netcdf_dataset


def _pick_dataarray(ds, names: tuple[str, ...]):
    lower = {str(k).lower(): k for k in ds.data_vars}
    for n in names:
        if n.lower() in lower:
            return ds[lower[n.lower()]]
    raise KeyError(f"变量缺失，候选={names}，实际={list(ds.data_vars)}")


def _norm_to_u8(x: np.ndarray, p_lo: float = 2.0, p_hi: float = 98.0) -> np.ndarray:
    xf = x[np.isfinite(x)]
    if xf.size == 0:
        return np.zeros_like(x, dtype=np.uint8)
    lo, hi = np.percentile(xf, (p_lo, p_hi))
    if hi <= lo:
        hi = lo + 1e-9
    y = np.clip((x - lo) / (hi - lo), 0, 1)
    y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
    return (y * 255).astype(np.uint8)


def _iter_processed_images(images_dir: Path, max_frames: int) -> list[Path]:
    files = []
    for pat in ("*.png", "*.jpg", "*.jpeg"):
        files.extend(sorted(images_dir.glob(pat)))
    return files[:max(1, int(max_frames))]


def _write_video_from_images(image_paths: list[Path], out_mp4: Path, fps: int) -> int:
    if not image_paths:
        raise ValueError("未找到可写入视频的图片。")
    first = cv2.imread(str(image_paths[0]))
    if first is None:
        raise ValueError(f"无法读取图片: {image_paths[0]}")
    h, w = first.shape[:2]
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    written = 0
    for p in image_paths:
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        if frame.shape[0] != h or frame.shape[1] != w:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        written += 1
    writer.release()
    return written


def _transcode_web_compatible(mp4_path: Path) -> tuple[bool, str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False, "未检测到 ffmpeg，跳过网页兼容转码（可自行安装 ffmpeg 后重试）。"
    tmp_out = mp4_path.with_name(f"{mp4_path.stem}.websafe.mp4")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(mp4_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(tmp_out),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        tmp_out.replace(mp4_path)
        return True, "已转码为网页兼容 mp4（H.264/yuv420p）。"
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", errors="ignore")
        return False, f"ffmpeg 转码失败，保留原始 mp4。错误摘要: {err[:400]}"


def _first_eddy_nc(data_config: str, explicit_nc: str | None) -> Path:
    if explicit_nc:
        p = resolve_path(explicit_nc)
        if not p.is_file():
            raise FileNotFoundError(f"指定 nc 不存在: {p}")
        return p
    cfg = load_yaml(data_config)
    raw_root = resolve_path(cfg["paths"]["raw_root"])
    sub = cfg["paths"].get("eddy_subdir", "中尺度涡识别")
    eddy_dir = raw_root / sub
    if not eddy_dir.is_dir():
        raise FileNotFoundError(f"涡旋目录不存在: {eddy_dir}")
    ncs: list[Path] = []
    for pat in ("*.nc", "*.nc4", "*.cdf"):
        ncs.extend(sorted(eddy_dir.rglob(pat)))
    if not ncs:
        raise FileNotFoundError(
            "目录下未找到可用 NetCDF。\n"
            f"- 搜索目录: {eddy_dir}\n"
            f"- data_config: {resolve_path(data_config)}\n"
            "请确认命题方涡旋数据已放置，或通过 --nc-path 指定具体文件。"
        )
    return ncs[0]


def _write_video_from_nc(nc_path: Path, out_mp4: Path, fps: int, max_frames: int, time_stride: int) -> int:
    ds, tmp_copy = open_netcdf_dataset(nc_path)
    try:
        adt = _pick_dataarray(ds, ("adt", "ADT"))
        ug = _pick_dataarray(ds, ("ugos", "UGOS"))
        vg = _pick_dataarray(ds, ("vgos", "VGOS"))
        sp = {"latitude", "longitude", "lat", "lon"}
        tdim = [d for d in adt.dims if d not in sp]
        if not tdim:
            raise RuntimeError("未找到时间维")
        tname = tdim[0]
        T = int(adt.sizes[tname])
        idxs = list(range(0, T, max(1, int(time_stride))))[:max(1, int(max_frames))]
        if not idxs:
            raise RuntimeError("未选中任何时间帧")

        # 先构造首帧用于确定视频大小
        i0 = idxs[0]
        a0 = np.asarray(adt.isel({tname: i0}).values, dtype=np.float64)
        u0 = np.asarray(ug.isel({tname: i0}).values, dtype=np.float64)
        v0 = np.asarray(vg.isel({tname: i0}).values, dtype=np.float64)
        rgb0 = np.stack([_norm_to_u8(a0), _norm_to_u8(u0), _norm_to_u8(v0)], axis=-1)
        frame0 = cv2.cvtColor(rgb0, cv2.COLOR_RGB2BGR)
        h, w = frame0.shape[:2]

        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
        written = 0
        for i in idxs:
            a = np.asarray(adt.isel({tname: i}).values, dtype=np.float64)
            u = np.asarray(ug.isel({tname: i}).values, dtype=np.float64)
            v = np.asarray(vg.isel({tname: i}).values, dtype=np.float64)
            rgb = np.stack([_norm_to_u8(a), _norm_to_u8(u), _norm_to_u8(v)], axis=-1)
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if frame.shape[0] != h or frame.shape[1] != w:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            written += 1
        writer.release()
        return written
    finally:
        ds.close()
        if tmp_copy is not None:
            try:
                tmp_copy.unlink(missing_ok=True)  # type: ignore[arg-type]
            except OSError:
                pass


def main() -> None:
    p = argparse.ArgumentParser(description="导出涡旋演示 mp4（优先 processed 图片，或从原始 nc 渲染）")
    p.add_argument("--mode", choices=("auto", "processed", "nc"), default="auto")
    p.add_argument("--images-dir", type=str, default="data/processed/eddy/images/val")
    p.add_argument("--data-config", type=str, default="config/data.yaml")
    p.add_argument("--nc-path", type=str, default="")
    p.add_argument("--out", type=str, default="submission/figures/eddy_demo/eddy_demo.mp4")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--max-frames", type=int, default=180)
    p.add_argument("--time-stride", type=int, default=1)
    web = p.add_mutually_exclusive_group()
    web.add_argument(
        "--web-compatible",
        dest="web_compatible",
        action="store_true",
        help="导出后用 ffmpeg 转码为 H.264/yuv420p（推荐，Streamlit/浏览器可播）",
    )
    web.add_argument(
        "--no-web-compatible",
        dest="web_compatible",
        action="store_false",
        help="跳过转码，保留 OpenCV 写入的 mpeg4/mp4v（部分浏览器无法预览）",
    )
    p.set_defaults(web_compatible=True)
    args = p.parse_args()

    out_mp4 = resolve_path(args.out)
    mode = args.mode
    if mode in ("auto", "processed"):
        images_dir = resolve_path(args.images_dir)
        imgs = _iter_processed_images(images_dir=images_dir, max_frames=int(args.max_frames))
        if imgs:
            written = _write_video_from_images(image_paths=imgs, out_mp4=out_mp4, fps=int(args.fps))
            if args.web_compatible:
                ok, msg = _transcode_web_compatible(out_mp4)
                print(msg)
            print(f"done(processed): {out_mp4} frames={written} source={images_dir}")
            return
        if mode == "processed":
            raise FileNotFoundError(f"processed 图片目录为空: {images_dir}")

    try:
        nc = _first_eddy_nc(data_config=args.data_config, explicit_nc=args.nc_path or None)
        written = _write_video_from_nc(
            nc_path=nc,
            out_mp4=out_mp4,
            fps=int(args.fps),
            max_frames=int(args.max_frames),
            time_stride=int(args.time_stride),
        )
        if args.web_compatible:
            ok, msg = _transcode_web_compatible(out_mp4)
            print(msg)
        print(f"done(nc): {out_mp4} frames={written} source={nc}")
    except Exception as e:
        raise SystemExit(
            "从原始 nc 导出失败。\n"
            f"原因: {e}\n"
            "可选处理：\n"
            "1) 先生成 processed 图像后再用 --mode processed；\n"
            "2) 用 --nc-path 显式指定一个可读 nc 文件；\n"
            "3) 检查 config/data.yaml 的 paths.raw_root 与 paths.eddy_subdir。"
        ) from e


if __name__ == "__main__":
    main()
