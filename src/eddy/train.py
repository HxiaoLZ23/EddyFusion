from __future__ import annotations

import argparse

from src.utils.config import load_yaml, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="涡旋 YOLOv8-seg 训练")
    parser.add_argument("--config", type=str, default="config/eddy.yaml")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="使用 ultralytics 内置 coco8-seg 做 1～2 epoch 烟测（无需本地数据）",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out = resolve_path(cfg["paths"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        from ultralytics import YOLO

        model = YOLO("yolov8n-seg.pt")
        model.train(
            data="coco8-seg.yaml",
            epochs=2,
            imgsz=640,
            project=str(out),
            name="smoke",
            exist_ok=True,
        )
        print("smoke 训练完成，权重目录:", out / "smoke")
        return

    raise NotImplementedError(
        "请准备 data/processed/eddy/ 下 ultralytics 数据集并配置 dataset_yaml，"
        "或先用 --smoke 验证环境与 GPU"
    )


if __name__ == "__main__":
    main()
