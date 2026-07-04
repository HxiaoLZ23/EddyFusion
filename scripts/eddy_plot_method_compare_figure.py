#!/usr/bin/env python3
"""涡旋方法对比图：原图 | GT（YOLO-seg 伪标签）| Faster R-CNN（bbox）| YOLOv8-seg（可选）。

示例::

  python scripts/eddy_plot_method_compare_figure.py \\
    --dataset-yaml data/processed/eddy/dataset.yaml \\
    --split val \\
    --yolo-ckpt ""   \\
    --frcnn-ckpt outputs/eddy_detector_faster_rcnn/best.pt \\
    --out outputs/eddy/figures/method_compare.png

若无 YOLO 权重，第四列为占位说明。第四列默认只画 YOLO 预测框（与第三列画法一致）；选用 YOLO 的论证见论文正文与 ``mask_map50`` 等指标表，不应单靠本 bbox 对照图得出结论。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _default_yolo_ckpt_candidates() -> list[str]:
    """本地/同步的云产物常见路径（与 config/eddy.yaml、config/eddy_enh.yaml 对齐）。"""
    return [
        "outputs/eddy/best.pt",
        "AutoDL/outputs/eddy_enh/best.pt",
        "AutoDL/outputs/eddy_enh/train/weights/best.pt",
        "AutoDL/outputs/eddy/best.pt",
        "AutoDL/outputs/eddy/train/weights/best.pt",
    ]


def resolve_yolo_checkpoint(cli_path: str) -> tuple[Path | None, list[str]]:
    """返回 (首个存在的权重路径, 尝试过的候选列表)。``cli_path`` 为空则按候选表自动查找。"""
    from src.utils.config import resolve_path

    tried: list[str] = []
    if cli_path.strip():
        p = resolve_path(cli_path.strip())
        tried.append(str(p))
        return (p if p.is_file() else None, tried)
    for rel in _default_yolo_ckpt_candidates():
        p = resolve_path(rel)
        tried.append(str(p))
        if p.is_file():
            return p, tried
    return None, tried


def _dataset_images_dir(dataset_yaml: Path, split: str) -> Path:
    import yaml

    from src.utils.config import resolve_path

    ds = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8")) or {}
    base = resolve_path(ds.get("path", dataset_yaml.parent)) if isinstance(ds, dict) else dataset_yaml.parent
    rel = ds.get(split) if isinstance(ds, dict) else None
    if not rel:
        raise ValueError(f"dataset.yaml 中无 split={split!r}")
    return base / str(rel)


def _parse_seg_label(path: Path, w: int, h: int) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """返回 (cls, poly_norm[N,2], bbox_xyxy 像素)。"""
    out: list[tuple[int, np.ndarray, np.ndarray]] = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls = int(float(parts[0]))
        coords = [float(x) for x in parts[1:]]
        if len(coords) < 6:
            continue
        poly = np.array(coords, dtype=np.float64).reshape(-1, 2)
        xs = poly[:, 0] * w
        ys = poly[:, 1] * h
        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())
        out.append((cls, poly, np.array([x1, y1, x2, y2], dtype=np.float64)))
    return out


def _color_for_class(cls: int) -> tuple[int, int, int]:
    # 与常见论文：气旋/反气旋分两色
    return (60, 120, 220) if cls == 0 else (220, 100, 60)


def _draw_gt(img: Image.Image, objs: list[tuple[int, np.ndarray, np.ndarray]], *, poly: bool) -> Image.Image:
    im = img.copy()
    draw = ImageDraw.Draw(im, "RGBA")
    w, h = im.size
    for cls, pn, bbox in objs:
        rgb = _color_for_class(cls)
        fill = (*rgb, 55)
        outline = (*rgb, 255)
        if poly:
            pts = [(float(pn[i, 0] * w), float(pn[i, 1] * h)) for i in range(len(pn))]
            if len(pts) >= 3:
                draw.polygon(pts, outline=outline[:3], width=2, fill=fill)
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=outline[:3], width=2)
    return im


def _draw_yolo_boxes_only(img: Image.Image, res, *, score_thr: float) -> Image.Image:
    """YOLO(-seg) 推理结果仅画预测框（与 Faster R-CNN 列同一套画法，便于对照）。"""
    if getattr(res, "boxes", None) is None or len(res.boxes) == 0:
        return img.copy()
    boxes = res.boxes.xyxy.cpu().numpy()
    scores = res.boxes.conf.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy().astype(np.int64)
    m = scores >= score_thr
    if not np.any(m):
        return img.copy()
    return _draw_boxes_pil(img, boxes[m], clss[m], scores[m], coco_style_cls=False)


def _build_frcnn(ckpt: Path, device: str):
    import torch
    import torchvision
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    num_classes = 3
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    m = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_features = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    blob = torch.load(ckpt, map_location=device)
    state = blob["model_state"] if isinstance(blob, dict) and "model_state" in blob else blob
    m.load_state_dict(state)
    m.eval()
    m.to(device)
    return m


def _frcnn_forward(model, image_path: Path, *, device: str, score_thr: float):
    from torchvision.transforms import functional as F

    import torch

    im = Image.open(image_path).convert("RGB")
    t = F.to_tensor(im).to(device)
    with torch.no_grad():
        out = model([t])[0]
    boxes = out["boxes"].cpu().numpy()
    scores = out["scores"].cpu().numpy()
    labels = out["labels"].cpu().numpy()
    msk = scores >= score_thr
    return im.copy(), boxes[msk], scores[msk], labels[msk]


def _draw_boxes_pil(
    img: Image.Image,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray | None,
    *,
    coco_style_cls: bool = True,
) -> Image.Image:
    im = img.copy()
    draw = ImageDraw.Draw(im, "RGBA")
    for i in range(len(boxes)):
        lab = int(labels[i])
        if coco_style_cls:
            cls = lab - 1 if lab >= 1 else 0  # COCO id 1,2 -> 0,1
        else:
            cls = lab  # Ultralytics YOLO：已为 0/1
        cls = max(0, min(1, cls))
        rgb = _color_for_class(cls)
        x1, y1, x2, y2 = boxes[i].tolist()
        draw.rectangle([x1, y1, x2, y2], outline=rgb, width=2)
        if scores is not None and i < len(scores):
            draw.text((x1 + 2, max(0, y1 - 12)), f"{float(scores[i]):.2f}", fill=rgb)
    return im


def main() -> None:
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description="涡旋 YOLO-seg / Faster R-CNN 对比拼图")
    ap.add_argument("--dataset-yaml", type=str, default="data/processed/eddy/dataset.yaml")
    ap.add_argument("--split", type=str, default="val", choices=("train", "val", "test"))
    ap.add_argument("--indices", type=str, default="0,1,2", help="排序后图像索引，逗号分隔")
    ap.add_argument("--frcnn-ckpt", type=str, default="outputs/eddy_detector_faster_rcnn/best.pt")
    ap.add_argument(
        "--yolo-ckpt",
        type=str,
        default="",
        help="留空则依次尝试 outputs/eddy/best.pt、AutoDL/outputs/eddy_enh/best.pt 等（见脚本内 _default_yolo_ckpt_candidates）",
    )
    ap.add_argument("--score-thr", type=float, default=0.5, help="Faster R-CNN 置信度阈值")
    ap.add_argument(
        "--yolo-score-thr",
        type=float,
        default=None,
        help="YOLO 列置信度阈值；默认与 --score-thr 相同",
    )
    ap.add_argument("--out", type=str, default="outputs/eddy/figures/method_compare.png")
    ap.add_argument("--device", type=str, default="")
    args = ap.parse_args()
    yolo_thr = float(args.score_thr) if args.yolo_score_thr is None else float(args.yolo_score_thr)

    from src.utils.config import resolve_path

    dy = resolve_path(args.dataset_yaml)
    img_dir = _dataset_images_dir(dy, args.split)
    if not img_dir.is_dir():
        raise SystemExit(f"无图像目录: {img_dir}")

    imgs = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
    if not imgs:
        raise SystemExit(f"{img_dir} 下无 png/jpg")

    idxs = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
    picks = [imgs[i] for i in idxs if 0 <= i < len(imgs)]
    if not picks:
        raise SystemExit("indices 全部越界或无图像")

    frcnn_path = resolve_path(args.frcnn_ckpt)
    if not frcnn_path.is_file():
        raise SystemExit(f"未找到 Faster R-CNN 权重: {frcnn_path}")

    yolo_path, yolo_tried = resolve_yolo_checkpoint(args.yolo_ckpt)
    have_yolo = yolo_path is not None

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    nrows = len(picks)
    ncols = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), squeeze=False)

    frcnn_m = _build_frcnn(frcnn_path, device)

    titles = (
        "输入 RGB",
        "GT（OW 伪标签｜多边形+外包框）",
        f"Faster R-CNN（bbox, thr={args.score_thr}）",
        f"YOLOv8-seg（仅画预测 bbox 对照, thr={yolo_thr}）",
    )
    yolo_model = None
    yolo_load_err: str | None = None
    if have_yolo:
        try:
            from ultralytics import YOLO

            yolo_model = YOLO(str(yolo_path))
        except Exception as e:  # noqa: BLE001 — 拼图需展示：缺依赖、权重损坏、通道不匹配等
            yolo_load_err = f"{type(e).__name__}: {e}"

    for r, ipath in enumerate(picks):
        img = Image.open(ipath).convert("RGB")
        w, h = img.size
        stem = ipath.stem
        base_root = ipath.parent.parent.parent
        lbl = base_root / "labels" / args.split / f"{stem}.txt"
        objs = _parse_seg_label(lbl, w, h)

        row = axes[r]
        row[0].imshow(np.asarray(img))
        row[0].set_title(titles[0], fontsize=10)
        row[0].axis("off")

        gt_vis = _draw_gt(img, objs, poly=True)
        row[1].imshow(np.asarray(gt_vis))
        row[1].set_title(titles[1], fontsize=10)
        row[1].axis("off")

        pil_b, bx, sc, lab = _frcnn_forward(frcnn_m, ipath, device=device, score_thr=args.score_thr)
        fr = _draw_boxes_pil(pil_b, bx, lab, sc)
        row[2].imshow(np.asarray(fr))
        row[2].set_title(titles[2], fontsize=10)
        row[2].axis("off")

        if yolo_model is not None:
            try:
                res = yolo_model.predict(str(ipath), verbose=False)[0]
                yo = _draw_yolo_boxes_only(img, res, score_thr=yolo_thr)
                row[3].imshow(np.asarray(yo))
                row[3].set_title(titles[3], fontsize=10)
            except Exception as e2:  # noqa: BLE001
                row[3].imshow(np.ones((h, w, 3), dtype=np.float32) * 0.92)
                err2 = f"{type(e2).__name__}: {e2}"
                if len(err2) > 280:
                    err2 = err2[:280] + "…"
                row[3].text(
                    0.5,
                    0.5,
                    "YOLO 推理失败\n（常见于 8ch 权重 vs 3ch 图：请换 --dataset-yaml 到 eddy_enh 或换 3ch 权重）\n"
                    + err2,
                    ha="center",
                    va="center",
                    fontsize=8,
                    transform=row[3].transAxes,
                )
                row[3].set_title("YOLOv8-seg（推理失败）", fontsize=10)
        elif have_yolo and yolo_load_err:
            row[3].imshow(np.ones((h, w, 3), dtype=np.float32) * 0.92)
            hint = yolo_load_err
            if "No module" in yolo_load_err or "ultralytics" in yolo_load_err.lower():
                hint += "\n\n可执行: pip install ultralytics"
            if len(hint) > 320:
                hint = hint[:320] + "…"
            row[3].text(0.5, 0.5, hint, ha="center", va="center", fontsize=8, transform=row[3].transAxes)
            row[3].set_title("YOLOv8-seg（未加载）", fontsize=10)
        else:
            row[3].imshow(np.ones((h, w, 3), dtype=np.float32) * 0.92)
            txt = "未找到 YOLO 权重。\n尝试过的路径（节选）:\n" + "\n".join(yolo_tried[:5])
            if len(yolo_tried) > 5:
                txt += "\n..."
            row[3].text(0.5, 0.5, txt, ha="center", va="center", fontsize=7, transform=row[3].transAxes)
            row[3].set_title("YOLOv8-seg（占位）", fontsize=10)
        row[3].axis("off")

    fig.tight_layout()
    out_p = resolve_path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(out_p)


if __name__ == "__main__":
    main()
