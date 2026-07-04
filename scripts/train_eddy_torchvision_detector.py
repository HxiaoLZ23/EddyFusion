#!/usr/bin/env python3
"""涡旋检测对比：torchvision Faster R-CNN（bbox）训练与验证。

前置：YOLO-seg 数据 → COCO bbox JSON：
  python -m src.eddy.coco_bbox_export --dataset-yaml data/processed/eddy/dataset.yaml --out outputs/eddy_coco_bbox

训练（GPU）：
  python scripts/train_eddy_torchvision_detector.py \\
    --coco-root <dataset.yaml 中 path 指向的数据集根> \\
    --train-json outputs/eddy_coco_bbox/train.json \\
    --val-json outputs/eddy_coco_bbox/val.json \\
    --epochs 12 --batch-size 4 --model faster_rcnn

Cascade R-CNN（torchvision 无）：请安装 MMDetection，见 configs/mmdet/README.md。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class EddyCocoBoxDataset(torch.utils.data.Dataset):
    """COCO export（仅 bbox），images file_name 相对 coco_root。"""

    def __init__(
        self,
        coco_root: Path,
        json_file: Path,
        *,
        transforms_horizontal_flip: float = 0.5,
        train: bool,
    ) -> None:
        self.root = coco_root
        self.train = train
        self.flip_p = transforms_horizontal_flip if train else 0.0
        meta = json.loads(Path(json_file).read_text(encoding="utf-8"))
        self.images = {im["id"]: im for im in meta["images"]}
        self.cat_ids = sorted({c["id"] for c in meta["categories"]})
        self.id_to_anns: dict[int, list[dict]] = {}
        for ann in meta["annotations"]:
            self.id_to_anns.setdefault(int(ann["image_id"]), []).append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        im_id = sorted(self.images.keys())[idx]
        info = self.images[im_id]
        path = self.root / info["file_name"]
        img = Image.open(path).convert("RGB")
        w, h = img.size
        boxes: list[list[float]] = []
        labels: list[int] = []
        for ann in self.id_to_anns.get(im_id, []):
            x, y, bw, bh = ann["bbox"]
            if bw <= 1 or bh <= 1:
                continue
            boxes.append([x, y, x + bw, y + bh])
            labels.append(int(ann["category_id"]))

        if self.train and random.random() < self.flip_p:
            img = F.hflip(img)
            boxes = [[w - x2, y1, w - x1, y2] for x1, y1, x2, y2 in boxes]

        # torchvision 检测参考实现仅 ToTensor，不对 ImageNet 归一化
        img_t = F.to_tensor(img)

        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([im_id])}
        return img_t, target


def collate_fn(batch):
    return tuple(zip(*batch))


def _build_model(num_classes: int, *, model_name: str) -> nn.Module:
    """
    num_classes：含背景（涡旋 2 类 → 3）
    """
    if model_name == "faster_rcnn":
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        m = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        in_features = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return m
    raise ValueError(f"未知 model={model_name}（当前仅 faster_rcnn）")


@torch.no_grad()
def _eval_map50(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    iou_thresh: float = 0.5,
    score_thresh: float = 0.05,
) -> dict[str, float]:
    """简易 bbox mAP@0.5：每类在全验证集上算 VOC 风格 AP 再平均；与 COCO 官方略有差异，仅作对照。"""
    model.eval()
    max_label = 2
    all_ap: list[float] = []

    for cat in range(1, max_label + 1):
        preds: list[tuple[int, np.ndarray, float]] = []
        gt_by_img: dict[int, list[np.ndarray]] = {}
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for out, tgt in zip(outputs, targets):
                im_id = int(tgt["image_id"].item())
                gt_boxes = tgt["boxes"].cpu().numpy()
                gt_lab = tgt["labels"].cpu().numpy()
                for box, lab in zip(gt_boxes, gt_lab):
                    if int(lab) == cat:
                        gt_by_img.setdefault(im_id, []).append(box.astype(np.float64))

                pred_boxes = out["boxes"].cpu().numpy()
                pred_scores = out["scores"].cpu().numpy()
                pred_lab = out["labels"].cpu().numpy()
                m = (pred_lab == cat) & (pred_scores >= score_thresh)
                for b, s in zip(pred_boxes[m], pred_scores[m]):
                    preds.append((im_id, b.astype(np.float64), float(s)))

        ap = _voc_ap_for_image_level(preds, gt_by_img, iou_thresh)
        all_ap.append(ap)
    map50 = float(np.mean(all_ap)) if all_ap else 0.0
    return {"bbox_map50": map50, "ap50_cls1": all_ap[0], "ap50_cls2": all_ap[1]}


def _voc_ap_for_image_level(
    preds: list[tuple[int, np.ndarray, float]],
    gt_by_img: dict[int, list[np.ndarray]],
    thr: float,
) -> float:
    """ preds: (image_id, box_xyxy, score)；gt 按图存储。无 GT 且无预测 → AP=1；无 GT 有预测 → 0。"""
    total_gt = sum(len(v) for v in gt_by_img.values())
    if total_gt == 0:
        return 0.0 if preds else 1.0
    if not preds:
        return 0.0

    preds_sorted = sorted(preds, key=lambda x: -x[2])
    matched: dict[int, set[int]] = {}
    tp = np.zeros(len(preds_sorted), dtype=np.float64)
    fp = np.zeros(len(preds_sorted), dtype=np.float64)

    for i, (im_id, box, _) in enumerate(preds_sorted):
        gts = gt_by_img.get(im_id)
        if not gts:
            fp[i] = 1
            continue
        arr = np.stack(gts, axis=0)
        ious = _iou_xyxy_matrix(box.reshape(1, 4), arr)
        j = int(np.argmax(ious))
        used = matched.setdefault(im_id, set())
        if float(ious[j]) >= thr and j not in used:
            tp[i] = 1
            used.add(j)
        else:
            fp[i] = 1

    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    rec = tp_c / total_gt
    prec = tp_c / np.maximum(tp_c + fp_c, 1e-6)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for k in range(len(mpre) - 1, 0, -1):
        mpre[k - 1] = max(mpre[k - 1], mpre[k])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _iou_xyxy_matrix(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    xa1, ya1, xa2, ya2 = box[0]
    xb1, yb1, xb2, yb2 = boxes.T
    inter_x1 = np.maximum(xa1, xb1)
    inter_y1 = np.maximum(ya1, yb1)
    inter_x2 = np.minimum(xa2, xb2)
    inter_y2 = np.minimum(ya2, yb2)
    iw = np.maximum(inter_x2 - inter_x1, 0)
    ih = np.maximum(inter_y2 - inter_y1, 0)
    inter = iw * ih
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter + 1e-6
    return inter / union


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-root", type=str, required=True, help="与 export 的 file_name 根一致（常为 data/processed/eddy）")
    ap.add_argument("--train-json", type=str, required=True)
    ap.add_argument("--val-json", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", type=str, default="faster_rcnn", choices=("faster_rcnn",))
    ap.add_argument("--out", type=str, default="outputs/eddy_detector_faster_rcnn")
    args = ap.parse_args()

    coco_root = Path(args.coco_root).resolve()
    train_json = Path(args.train_json).resolve()
    val_json = Path(args.val_json).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_tr = EddyCocoBoxDataset(coco_root, train_json, train=True)
    ds_va = EddyCocoBoxDataset(coco_root, val_json, train=False)

    nw = args.workers
    tl = torch.utils.data.DataLoader(
        ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=nw, collate_fn=collate_fn
    )
    vl = torch.utils.data.DataLoader(
        ds_va, batch_size=args.batch_size, shuffle=False, num_workers=nw, collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 3  # bg + 2 eddy classes
    model = _build_model(num_classes, model_name=args.model)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    milestones = [int(math.floor(args.epochs * 2 / 3)), int(math.floor(args.epochs * 9 / 10))]
    milestones = sorted({m for m in milestones if m > 0}) or [max(1, args.epochs - 2)]
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.1)

    best_map = -1.0
    log_lines: list[str] = []

    def train_one(ep: int) -> float:
        model.train()
        loss_sum = 0.0
        n_batches = 0
        for images, targets in tl:
            images = list(im.to(device) for im in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            opt.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            opt.step()
            loss_sum += float(losses.item())
            n_batches += 1
        return loss_sum / max(n_batches, 1)

    for ep in range(1, args.epochs + 1):
        tr_loss = train_one(ep)
        sched.step()
        metrics = _eval_map50(model, vl, device)
        line = f"epoch {ep}/{args.epochs} train_loss={tr_loss:.4f} bbox_map50={metrics['bbox_map50']:.4f}"
        print(line, flush=True)
        log_lines.append(line)
        if metrics["bbox_map50"] >= best_map:
            best_map = metrics["bbox_map50"]
            torch.save(
                {"model_state": model.state_dict(), "epoch": ep, "metrics": metrics, "args": vars(args)},
                out_dir / "best.pt",
            )
        torch.save(
            {"model_state": model.state_dict(), "epoch": ep, "metrics": metrics},
            out_dir / "last.pt",
        )

    (out_dir / "train_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    summary = {
        "model": args.model,
        "best_bbox_map50": best_map,
        "note": "bbox mAP@0.5 为简易实现，与 COCO 官方略有差异；与 YOLO mask_map50 不可直接横向比较。",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成。best bbox_map50≈{best_map:.4f}，输出目录: {out_dir}")


if __name__ == "__main__":
    main()
