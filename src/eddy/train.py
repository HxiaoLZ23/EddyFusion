from __future__ import annotations

import argparse
import shutil
import sys

from src.utils.config import load_yaml, resolve_path


def _maybe_line_buffer_stdio() -> None:
    """云机 WebShell / 重定向时 stdout 常为全缓冲，导致长时间无输出；尽力改为按行缓冲。"""
    if sys.stdout.isatty() and sys.stderr.isatty():
        return
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(line_buffering=True)
            except Exception:
                pass

# `model.train(...)` 已由本文件显式传入的键，勿从 yaml 重复转发
_TRAIN_RESERVED = frozenset(
    {
        "device",
        "epochs",
        "batch_size",
        "workers",
        "amp",
        "pretrained",
        "run_name",
        "name",
    }
)
# Ultralytics YOLO 训练常见可调项（按需写在 config train 段）
_ULTRA_EXTRA_KEYS = frozenset(
    {
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_epochs",
        "warmup_momentum",
        "warmup_bias_lr",
        "cos_lr",
        "close_mosaic",
        "patience",
        "seed",
        "optimizer",
        "label_smoothing",
        "freeze",
        "rect",
        "single_cls",
        "deterministic",
        "plots",
        "verbose",
        "val",
        "overlap_mask",
        "mask_ratio",
        "dropout",
        "mosaic",
        "mixup",
        "copy_paste",
        "copy_paste_mode",
        "auto_augment",
        "erasing",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "flipud",
        "fliplr",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "bgr",
    }
)


def _ultra_train_extras(tc: dict) -> dict:
    """从 yaml ``train`` 段提取转发给 Ultralytics 的额外参数。"""
    out: dict = {}
    for k, v in tc.items():
        if k in _TRAIN_RESERVED or v is None:
            continue
        if k in _ULTRA_EXTRA_KEYS:
            out[k] = v
    return out


def _dataset_channels(dataset_yaml_path) -> int:
    p = resolve_path(dataset_yaml_path)
    if not p.is_file():
        return 3
    data = load_yaml(p)
    return int(data.get("channels", 3))


def main() -> None:
    _maybe_line_buffer_stdio()
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
            verbose=True,
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

    ch = _dataset_channels(dataset_yaml)
    backbone = str(cfg["model"]["backbone"])
    weights = backbone if backbone.endswith(".pt") else f"{backbone}.pt"

    if ch > 3:
        arch = str(cfg["model"].get("architecture_yaml", "yolov8n-seg.yaml"))
        print(f"[eddy train] 多通道数据 channels={ch}，自 yaml 构建模型（无 COCO 预训练首层）: {arch}")
        model = YOLO(arch)
        use_pretrained = False
    else:
        model = YOLO(weights)
        use_pretrained = bool(cfg["train"].get("pretrained", True))

    tc = cfg["train"]
    ms = cfg["model"]["input_size"]
    imgsz = int(ms[0]) if isinstance(ms, (list, tuple)) else int(ms)
    device = tc.get("device", "cuda")
    if device == "cuda":
        device = 0

    run_name = str(tc.get("run_name") or tc.get("name") or "train")
    extras = _ultra_train_extras(tc)
    train_kw = dict(
        data=str(dataset_yaml),
        epochs=int(tc["epochs"]),
        batch=int(tc["batch_size"]),
        imgsz=imgsz,
        device=device,
        project=str(out),
        name=run_name,
        exist_ok=True,
        workers=int(tc.get("workers", 4)),
        amp=bool(tc.get("amp", True)),
        pretrained=use_pretrained,
    )
    train_kw.update(extras)
    train_kw["verbose"] = bool(tc.get("verbose", True))
    print("[eddy train] 启动 Ultralytics；若干步后应出现 epoch/tqdm。若无输出可试: python -u -m src.eddy.train ...", flush=True)
    model.train(**train_kw)

    trained_best = out / run_name / "weights" / "best.pt"
    trained_last = out / run_name / "weights" / "last.pt"
    if trained_best.is_file():
        shutil.copy2(trained_best, out / "best.pt")
        print("已复制 best.pt ->", out / "best.pt")
    if trained_last.is_file():
        shutil.copy2(trained_last, out / "last.pt")
        print("已复制 last.pt ->", out / "last.pt")


if __name__ == "__main__":
    main()
