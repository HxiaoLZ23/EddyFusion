#!/usr/bin/env python3
"""同一区域：GT 多边形 vs YOLO 实例分割（半透明填充或轮廓线）。

用于论文中说明「分割头输出」与 OW 伪标签多边形的几何对照；与 Faster R-CNN bbox 对照图互补。

示例::

  python scripts/eddy_plot_gt_vs_yolo_seg.py \\
    --dataset-yaml data/processed/eddy/dataset.yaml \\
    --split val --indices 0,1,2 \\
    --yolo-ckpt "" \\
    --seg-style fill \\
    --out outputs/eddy/figures/gt_vs_yolo_seg.png

``--seg-style contour``：YOLO 侧仅画轮廓线（不着色）。
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_compare_helpers():
    """复用 ``eddy_plot_method_compare_figure`` 中的路径/GT 解析（避免重复维护）。"""
    p = REPO / "scripts" / "eddy_plot_method_compare_figure.py"
    spec = importlib.util.spec_from_file_location("_eddy_compare_helpers", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载: {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _draw_gt_polygons_only(img: Image.Image, objs, *, cmp_mod) -> Image.Image:
    """仅 OW 伪标签多边形（半透明 + 细轮廓），不画外包矩形。"""
    im = img.convert("RGBA")
    draw = ImageDraw.Draw(im, "RGBA")
    w, h = im.size
    for cls, pn, _bbox in objs:
        rgb = cmp_mod._color_for_class(cls)
        fill = (*rgb, 75)
        outline = (*rgb, 255)
        pts = [(float(pn[i, 0] * w), float(pn[i, 1] * h)) for i in range(len(pn))]
        if len(pts) >= 3:
            draw.polygon(pts, outline=outline[:3], width=2, fill=fill)
    return im.convert("RGB")


def _draw_yolo_seg_overlay(
    img: Image.Image,
    res,
    *,
    score_thr: float,
    seg_style: str,
    cmp_mod,
) -> Image.Image:
    """YOLO-seg：按 ``masks.xy`` 绘制；``seg_style`` 为 ``fill`` 或 ``contour``。"""
    im = img.convert("RGBA")
    draw = ImageDraw.Draw(im, "RGBA")
    if getattr(res, "boxes", None) is None or len(res.boxes) == 0:
        return img.copy()

    boxes = res.boxes
    mask_xy_list = None
    if getattr(res, "masks", None) is not None and res.masks is not None:
        mask_xy_list = res.masks.xy

    for i in range(len(boxes)):
        conf = float(boxes.conf[i])
        if conf < score_thr:
            continue
        cls_id = int(boxes.cls[i])
        rgb = cmp_mod._color_for_class(cls_id)
        outline_c = (*rgb, 255)
        fill_rgba = (*rgb, 75) if seg_style == "fill" else None

        if mask_xy_list is None or i >= len(mask_xy_list):
            xyxy = boxes.xyxy[i].cpu().numpy().reshape(4).tolist()
            draw.rectangle(xyxy, outline=outline_c[:3], width=2)
            draw.text((xyxy[0] + 2, max(0.0, xyxy[1] - 11)), f"{conf:.2f}", fill=outline_c[:3])
            continue

        seg = np.asarray(mask_xy_list[i], dtype=np.float64)
        if seg.ndim != 2 or seg.shape[0] < 3:
            continue
        poly_pts = [(float(seg[j, 0]), float(seg[j, 1])) for j in range(seg.shape[0])]

        if seg_style == "fill":
            draw.polygon(poly_pts, outline=outline_c[:3], width=2, fill=fill_rgba)
        else:
            draw.polygon(poly_pts, outline=outline_c[:3], width=2)
        x1, y1, _, _ = boxes.xyxy[i].cpu().numpy().tolist()
        draw.text((x1 + 2, max(0.0, y1 - 11)), f"{conf:.2f}", fill=outline_c[:3])

    return im.convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser(description="GT 多边形 vs YOLO 分割（同一区域）")
    ap.add_argument("--dataset-yaml", type=str, default="data/processed/eddy/dataset.yaml")
    ap.add_argument("--split", type=str, default="val", choices=("train", "val", "test"))
    ap.add_argument("--indices", type=str, default="0,1,2")
    ap.add_argument("--yolo-ckpt", type=str, default="", help="留空则自动查找常用权重路径")
    ap.add_argument("--score-thr", type=float, default=0.5)
    ap.add_argument(
        "--seg-style",
        type=str,
        choices=("fill", "contour"),
        default="fill",
        help="YOLO 列：半透明填充 或 仅轮廓",
    )
    ap.add_argument("--out", type=str, default="outputs/eddy/figures/gt_vs_yolo_seg.png")
    args = ap.parse_args()

    cmp_mod = _load_compare_helpers()
    from src.utils.config import resolve_path

    dy = resolve_path(args.dataset_yaml)
    img_dir = cmp_mod._dataset_images_dir(dy, args.split)
    if not img_dir.is_dir():
        raise SystemExit(f"无图像目录: {img_dir}")

    imgs = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
    idxs = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
    picks = [imgs[i] for i in idxs if 0 <= i < len(imgs)]
    if not picks:
        raise SystemExit("indices 越界或无图像")

    yolo_path, tried = cmp_mod.resolve_yolo_checkpoint(args.yolo_ckpt)
    if yolo_path is None:
        raise SystemExit("未找到 YOLO 权重，尝试过:\n" + "\n".join(tried[:8]))

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("请 pip install ultralytics") from e

    model = YOLO(str(yolo_path))

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    nrows = len(picks)
    fig, axes = plt.subplots(nrows, 3, figsize=(4.6 * 3, 3.8 * nrows), squeeze=False)

    titles = (
        "输入 RGB",
        "GT（OW 伪标签｜多边形）",
        f"YOLO-seg（{args.seg_style}，thr={args.score_thr}）",
    )

    for r, ipath in enumerate(picks):
        img = Image.open(ipath).convert("RGB")
        w, h = img.size
        stem = ipath.stem
        base_root = ipath.parent.parent.parent
        lbl = base_root / "labels" / args.split / f"{stem}.txt"
        objs = cmp_mod._parse_seg_label(lbl, w, h)

        row = axes[r]
        row[0].imshow(np.asarray(img))
        row[0].set_title(titles[0], fontsize=10)
        row[0].axis("off")

        gt_vis = _draw_gt_polygons_only(img, objs, cmp_mod=cmp_mod)
        row[1].imshow(np.asarray(gt_vis))
        row[1].set_title(titles[1], fontsize=10)
        row[1].axis("off")

        res = model.predict(str(ipath), verbose=False)[0]
        yo_vis = _draw_yolo_seg_overlay(
            img, res, score_thr=args.score_thr, seg_style=args.seg_style, cmp_mod=cmp_mod
        )
        row[2].imshow(np.asarray(yo_vis))
        row[2].set_title(titles[2], fontsize=10)
        row[2].axis("off")

    fig.tight_layout()
    out_p = resolve_path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(out_p)


if __name__ == "__main__":
    main()
