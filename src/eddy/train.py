from __future__ import annotations

import argparse
import shutil

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

    dataset_yaml = resolve_path(cfg["paths"]["dataset_yaml"])
    if not dataset_yaml.is_file():
        raise FileNotFoundError(
            f"未找到 Ultralytics 数据集描述: {dataset_yaml}\n"
            "请先准备 data/processed/eddy/ 下 YOLO-seg 格式数据并写入 dataset.yaml，"
            "或运行: python -m src.preprocess.eddy_dataset --write-template\n"
            "或先用 --smoke 验证环境。"
        )

    from ultralytics import YOLO

    backbone = str(cfg["model"]["backbone"])
    weights = backbone if backbone.endswith(".pt") else f"{backbone}.pt"
    model = YOLO(weights)

    tc = cfg["train"]
    ms = cfg["model"]["input_size"]
    imgsz = int(ms[0]) if isinstance(ms, (list, tuple)) else int(ms)
    device = tc.get("device", "cuda")
    if device == "cuda":
        device = 0

    name = "train"
    model.train(
        data=str(dataset_yaml),
        epochs=int(tc["epochs"]),
        batch=int(tc["batch_size"]),
        imgsz=imgsz,
        device=device,
        project=str(out),
        name=name,
        exist_ok=True,
        workers=int(tc.get("workers", 4)),
        amp=bool(tc.get("amp", True)),
    )

    trained_best = out / name / "weights" / "best.pt"
    trained_last = out / name / "weights" / "last.pt"
    if trained_best.is_file():
        shutil.copy2(trained_best, out / "best.pt")
        print("已复制 best.pt ->", out / "best.pt")
    if trained_last.is_file():
        shutil.copy2(trained_last, out / "last.pt")
        print("已复制 last.pt ->", out / "last.pt")


if __name__ == "__main__":
    main()
