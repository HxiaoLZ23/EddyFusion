"""冒烟用小数据集：生成最小 YOLO-seg 目录与 dataset.yaml。

当仓库里尚无 ``data/processed/eddy`` 时，可用于跑通 ``coco_bbox_export`` +
``train_eddy_torchvision_detector``。真实实验请换为你的 ``dataset.yaml``。
"""
from __future__ import annotations

from pathlib import Path

import yaml
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]


def poly_rect(cx: float, cy: float, w: float, h: float) -> list[float]:
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def main() -> None:
    root = REPO / "outputs/rcnn_pipeline_smoke/dataset_root"
    for sp in ("train", "val"):
        (root / sp / "images").mkdir(parents=True, exist_ok=True)
        (root / sp / "labels").mkdir(parents=True, exist_ok=True)

    specs = [
        ("train", "dummy_tr001.png", [(0, poly_rect(0.45, 0.5, 0.28, 0.26)), (1, poly_rect(0.72, 0.3, 0.14, 0.14))]),
        ("train", "dummy_tr002.png", [(0, poly_rect(0.52, 0.48, 0.22, 0.34))]),
        ("train", "dummy_tr003.png", [(1, poly_rect(0.5, 0.55, 0.3, 0.2))]),
        ("val", "dummy_va001.png", [(0, poly_rect(0.49, 0.51, 0.26, 0.24))]),
    ]

    for split, name, objs in specs:
        img = Image.new("RGB", (256, 256), color=(245, 245, 240))
        ImageDraw.Draw(img).ellipse([48, 48, 208, 208], outline=(200, 200, 200))
        img.save(root / split / "images" / name)
        lines = []
        for cid, poly in objs:
            flat = " ".join(f"{v:.6f}" for v in poly)
            lines.append(f"{cid} {flat}")
        (root / split / "labels" / (Path(name).stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")

    yaml_path = REPO / "outputs/rcnn_pipeline_smoke/dataset.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "path": "outputs/rcnn_pipeline_smoke/dataset_root",
        "train": "train/images",
        "val": "val/images",
        "names": {0: "eddy_cyclonic", 1: "eddy_anticyclonic"},
    }
    yaml_path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    print(yaml_path)


if __name__ == "__main__":
    main()
