"""将 Ultralytics/YOLO-seg 数据集（polygon txt）导出为 COCO Detection JSON（仅用轴对齐 bbox，供 R-CNN 类对比实验）。"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from src.utils.config import resolve_path


def _yolo_polygon_line_to_pixels(
    cls: int, coords: list[float], img_w: int, img_h: int
) -> tuple[list[float], int] | None:
    """单行 YOLO-seg：`cls x1 y1 x2 y2 ...` 归一化。"""
    if len(coords) < 6:
        return None
    xs = coords[::2]
    ys = coords[1::2]
    x_pix = [min(max(x * img_w, 0.0), float(img_w - 1)) for x in xs]
    y_pix = [min(max(y * img_h, 0.0), float(img_h - 1)) for y in ys]
    x1, x2 = min(x_pix), max(x_pix)
    y1, y2 = min(y_pix), max(y_pix)
    if x2 <= x1 or y2 <= y1:
        return None
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], int(cls)


def _parse_label_file(txt_path: Path, img_w: int, img_h: int) -> list[tuple[int, list[float]]]:
    objs: list[tuple[int, list[float]]] = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls_id = int(float(parts[0]))
        floats = [float(x) for x in parts[1:]]
        wh = _yolo_polygon_line_to_pixels(cls_id, floats, img_w, img_h)
        if wh is None:
            continue
        bbox, cid = wh
        objs.append((cid + 1, bbox))  # COCO cat id 从 1 开始；导出时两类为 1,2
    return objs


def _coco_template(categories: list[dict[str, Any]]) -> dict[str, Any]:
    return {"info": {}, "licenses": [], "categories": categories, "images": [], "annotations": []}


def build_coco_split(
    image_dir: Path,
    label_dir: Path,
    *,
    image_dir_root: Path,
    category_names: list[str],
    split_name: str,
) -> tuple[dict[str, Any], int, int]:
    categories = [{"id": i + 1, "name": n, "supercategory": "eddy"} for i, n in enumerate(category_names)]
    coco = _coco_template(categories)
    ann_id = 1
    img_id = 1
    n_skip = 0
    imgs = sorted(image_dir.glob("*"))
    for img_path in imgs:
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        lbl = label_dir / f"{img_path.stem}.txt"
        try:
            w, h = Image.open(img_path).convert("RGB").size
        except OSError:
            n_skip += 1
            continue
        img_rel = img_path.relative_to(image_dir_root).as_posix()
        coco["images"].append(
            {
                "id": img_id,
                "file_name": img_rel,
                "width": w,
                "height": h,
                "license": None,
                "split": split_name,
            }
        )
        objs: list[tuple[int, list[float]]] = []
        if lbl.is_file():
            objs = _parse_label_file(lbl, w, h)
        else:
            n_skip += 1
        for cid, bbox_xywh in objs:
            x, y, bw, bh = bbox_xywh
            area = float(bw * bh)
            if area <= 0:
                continue
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cid,
                    "bbox": [x, y, bw, bh],
                    "area": area,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
        img_id += 1
    return coco, len(coco["images"]), ann_id - 1


def export_from_dataset_yaml(
    dataset_yaml: Path | str,
    out_dir: Path | str,
    *,
    copy_images: bool = False,
    category_names: tuple[str, str] = ("eddy_cyclonic", "eddy_anticyclonic"),
) -> dict[str, Path]:
    """
    dataset.yaml 中与 Ultralytics 相同：train/val/test 为相对 path 的子路径。
    输出：out_dir/train.json val.json test.json （可选拷贝 images 至 out_dir/images）
    """
    import yaml as yamllib

    p = resolve_path(dataset_yaml)
    txt = p.read_text(encoding="utf-8")
    ds = yamllib.safe_load(txt) or {}
    if not isinstance(ds, dict):
        raise ValueError(f"无效的 dataset yaml: {p}")
    eddy_root = p.parent
    ds_path = ds.get("path")
    base = resolve_path(ds_path if isinstance(ds_path, str) else eddy_root)
    splits = []
    if "train" in ds:
        splits.append(("train", str(ds["train"])))
    if "val" in ds:
        splits.append(("val", str(ds["val"])))
    if "test" in ds:
        splits.append(("test", str(ds["test"])))

    out = resolve_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    outs: dict[str, Path] = {}
    cats = list(category_names)
    if copy_images:
        (out / "images").mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"dataset_yaml": str(p), "coco_root": str(base), "splits": {}}
    for sp, rel_img in splits:
        image_dir = base / rel_img
        label_rel = rel_img.replace("images", "labels", 1)
        label_dir = base / label_rel
        if not image_dir.is_dir():
            print(f"警告: 无图像目录 {image_dir}，跳过 split={sp}")
            continue
        if not label_dir.is_dir():
            print(f"警告: 无标签目录 {label_dir}，跳过 split={sp}")
            continue
        coco, n_im, n_an = build_coco_split(
            image_dir,
            label_dir,
            image_dir_root=base,
            category_names=cats,
            split_name=sp,
        )
        json_path = out / f"{sp}.json"
        json_path.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
        outs[sp] = json_path
        summary["splits"][sp] = {"images": n_im, "annotations": n_an, "json": str(json_path)}
        if copy_images:
            dst = out / "images" / sp
            dst.mkdir(parents=True, exist_ok=True)
            for im in image_dir.glob("*"):
                if im.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    shutil.copy2(im, dst / im.name)
            # 重写 file_name 为相对 out 的路径
            rel_prefix = f"images/{sp}/"
            for img in coco["images"]:
                name = Path(img["file_name"]).name
                img["file_name"] = rel_prefix + name
            json_path.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")

    (out / "export_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return outs


def main_argv(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="YOLO-seg → COCO bbox JSON（涡旋对比 R-CNN）")
    ap.add_argument("--dataset-yaml", type=str, default="data/processed/eddy/dataset.yaml")
    ap.add_argument("--out", type=str, default="outputs/eddy_coco_bbox")
    ap.add_argument("--copy-images", action="store_true", help="复制图像到 out/images/{split}/，便于 MMDet 根目录")
    args = ap.parse_args(argv)
    export_from_dataset_yaml(args.dataset_yaml, args.out, copy_images=bool(args.copy_images))
    print(f"已写入: {resolve_path(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_argv())
