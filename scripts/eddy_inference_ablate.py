#!/usr/bin/env python3
"""
单图/目录涡旋推理消融：对比 baseline / 频域增强 / 多尺度 TTA 提示 / 组合。

用于模块A「多尺度/频域」可见证据与简易表；正式 mAP 仍以 `python -m src.eddy.eval` 为准。

示例：
  python scripts/eddy_inference_ablate.py --images data/processed/eddy/images/val --ckpt outputs/eddy/best.pt
  # 8 通道权重 + 导出带 .npy 时：
  python scripts/eddy_inference_ablate.py --images .../images/val --ckpt AutoDL/outputs/eddy_enh/best.pt --use-npy
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eddy.frequency_enhance import enhance_bgr_frequency
from src.eddy.multiscale_tta import tta_any_detection


def _load_image_or_npy(p: Path, use_npy: bool) -> np.ndarray | None:
    if use_npy:
        fn = p.with_suffix(".npy")
        if not fn.is_file():
            # 传入的已是 .npy 路径
            fn = p if p.suffix.lower() == ".npy" else fn
        if fn.is_file():
            return np.load(str(fn)).astype(np.float32)
    im = cv2.imread(str(p))
    return im


def _apply_freq_to_mc_stack(stack_hw8: np.ndarray, freq: str, amount: float) -> np.ndarray:
    """前 3 通道为 BGR（与 Ultralytics PNG/npy 约定一致），频域后直接写回。"""
    bgr = np.clip(stack_hw8[:, :, :3] * 255.0, 0, 255).astype(np.uint8)
    if freq != "none":
        bgr = enhance_bgr_frequency(bgr, mode=freq, amount=amount)  # type: ignore[arg-type]
    out = stack_hw8.copy()
    out[:, :, :3] = bgr.astype(np.float32) / 255.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=str, required=True, help="单张图或含 jpg/png 的目录")
    ap.add_argument("--ckpt", type=str, default="outputs/eddy/best.pt")
    ap.add_argument("--out-csv", type=str, default="outputs/eddy/inference_ablation.csv")
    ap.add_argument(
        "--use-npy",
        action="store_true",
        help="与 PNG 同 stem 的 HWC float npy（8ch 权重）；仅在数据已导出 stacked npy 时可用",
    )
    args = ap.parse_args()

    from ultralytics import YOLO

    from src.utils.config import resolve_path

    ckpt = resolve_path(args.ckpt)
    if not ckpt.is_file():
        raise SystemExit(f"权重不存在: {ckpt}")
    model = YOLO(str(ckpt))

    root = Path(args.images)
    paths: list[Path] = []
    if root.is_file():
        paths = [root]
    else:
        if args.use_npy:
            paths = sorted(root.rglob("*.npy"))
        if not paths:
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                paths.extend(sorted(root.rglob(ext)))

    if not paths:
        raise SystemExit("未找到图像")

    rows = []
    for p in paths:
        im = _load_image_or_npy(p, args.use_npy)
        if im is None:
            continue

        def run(tag: str, frame: np.ndarray, freq: str, tta: bool) -> dict:
            fr = frame
            if isinstance(fr, np.ndarray) and fr.ndim == 3 and fr.shape[2] > 3:
                fr = _apply_freq_to_mc_stack(fr, freq, amount=0.7) if freq != "none" else fr
            else:
                if freq != "none":
                    fr = enhance_bgr_frequency(fr, mode=freq, amount=0.7)  # type: ignore[arg-type]
            tta_hit = tta_any_detection(model, fr) if tta else False
            pred = model.predict(fr, conf=0.25, iou=0.45, imgsz=640, verbose=False)[0]
            n = int(len(pred.boxes)) if pred.boxes is not None else 0
            mc = 0.0
            if n > 0:
                mc = float(np.mean(pred.boxes.conf.detach().cpu().numpy()))
            return {
                "path": str(p),
                "mode": tag,
                "n_det": n,
                "mean_conf": round(mc, 6),
                "tta_any": tta_hit,
            }

        for tag, freq, tta in [
            ("baseline", "none", False),
            ("freq_unsharp", "unsharp", False),
            ("tta_only", "none", True),
            ("freq+tta", "unsharp", True),
        ]:
            rows.append(run(tag, im, freq, tta))

    outp = resolve_path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "mode", "n_det", "mean_conf", "tta_any"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {outp} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
