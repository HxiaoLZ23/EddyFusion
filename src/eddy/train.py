"""
涡旋检测训练入口：Ultralytics YOLOv8 实例分割。

数据前提
--------
不直接读取 NetCDF。须先用预处理导出 YOLO-seg 数据集，见：
  - ``src/preprocess/eddy_dataset.py``（``--export-yolo``）
  - ``src/preprocess/eddy_yolo_export.py``（核心：NC → images/labels + dataset.yaml）
  - 文档：``docs/架构与方法/NetCDF与三模块数据及训练说明.md`` §3

配置
----
``config/eddy.yaml`` 默认 Fair-B0：``paths.dataset_yaml`` → ``AutoDL/dataset/eddy_v6_b0_fair/dataset.yaml``。
``channels: 3`` 可用 COCO 预训练权重；``channels: 8`` 需 ``architecture_yaml`` 且通常 ``pretrained: false``。

用法
----
  python -m src.eddy.train --config config/eddy.yaml
  python -m src.eddy.train --smoke   # 仅验证 ultralytics 环境
"""

from __future__ import annotations

import argparse
import shutil
import sys

from src.eddy.multichannel_init import (
    build_yolo_multichannel_from_baseline,
    build_yolo_multichannel_from_coco,
)
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


# model.train(...) 已由本文件显式传入的键，勿从 yaml 重复转发
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
    """从 yaml ``train`` 段提取可安全转发给 ``model.train()`` 的超参（学习率、增强等）。"""
    out: dict = {}
    for k, v in tc.items():
        if k in _TRAIN_RESERVED or v is None:
            continue
        if k in _ULTRA_EXTRA_KEYS:
            out[k] = v
    return out


def _dataset_channels(dataset_yaml_path) -> int:
    """读取 dataset.yaml 的 channels 字段，决定 3 通道 RGB 或 8 通道物理栈。"""
    p = resolve_path(dataset_yaml_path)
    if not p.is_file():
        return 3
    data = load_yaml(p)
    return int(data.get("channels", 3))


def main() -> None:
    _maybe_line_buffer_stdio()
    parser = argparse.ArgumentParser(description="涡旋 YOLOv8-seg 训练（输入为 YOLO 数据集，非 NetCDF）")
    parser.add_argument("--config", type=str, default="config/eddy.yaml")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="使用 ultralytics 内置 coco8-seg 做 1～2 epoch 烟测（无需本地数据）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从 paths.output_dir/<run_name>/weights/last.pt 断点续训（Ultralytics resume=True）",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out = resolve_path(cfg["paths"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    # --- 烟测：不依赖 data/processed/eddy ---
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
            "请先运行 NC→YOLO 导出，例如:\n"
            "  python -m src.preprocess.eddy_dataset --export-yolo -- "
            "--data-config config/data.yaml --out data/processed/eddy\n"
            "详见 docs/架构与方法/NetCDF与三模块数据及训练说明.md\n"
            "或: python -m src.preprocess.eddy_dataset --write-template\n"
            "或先用 --smoke 验证环境。"
        )

    from ultralytics import YOLO

    tc = cfg["train"]
    run_name = str(tc.get("run_name") or tc.get("name") or "train")

    if args.resume:
        last_pt = out / run_name / "weights" / "last.pt"
        if not last_pt.is_file():
            raise FileNotFoundError(f"断点权重不存在，无法 resume: {last_pt}")
        print(f"[eddy train] 断点续训: {last_pt}", flush=True)
        model = YOLO(str(last_pt))
        use_pretrained = False
    else:
        ch = _dataset_channels(dataset_yaml)
        backbone = str(cfg["model"]["backbone"])
        weights = backbone if backbone.endswith(".pt") else f"{backbone}.pt"

        if ch > 3:
            arch = str(cfg["model"].get("architecture_yaml", "yolov8n-seg.yaml"))
            init_from_ckpt = cfg["model"].get("init_from_ckpt")
            init_from_baseline = cfg["model"].get("init_from_baseline")
            init_from_coco = bool(cfg["model"].get("init_from_coco", False))
            if init_from_ckpt:
                ckpt = resolve_path(str(init_from_ckpt))
                if not ckpt.is_file():
                    raise FileNotFoundError(f"init_from_ckpt 不存在: {ckpt}")
                print(f"[eddy train] 多通道 channels={ch}，从 checkpoint 热启动: {ckpt}", flush=True)
                model = YOLO(str(ckpt))
            elif init_from_baseline:
                baseline_pt = resolve_path(str(init_from_baseline))
                init_pt = out / f"init_from3ch_ch{ch}.pt"
                print(
                    f"[eddy train] 多通道 channels={ch}，3ch 基线首层扩展: {baseline_pt} -> {init_pt.name}",
                    flush=True,
                )
                model = build_yolo_multichannel_from_baseline(
                    channels=ch,
                    baseline_pt=baseline_pt,
                    save_path=init_pt,
                )
            elif init_from_coco:
                backbone = str(cfg["model"]["backbone"])
                weights = backbone if backbone.endswith(".pt") else f"{backbone}.pt"
                init_pt = out / f"init_ch{ch}.pt"
                print(
                    f"[eddy train] 多通道 channels={ch}，COCO 首层扩展初始化: {weights} -> {init_pt.name}",
                    flush=True,
                )
                model = build_yolo_multichannel_from_coco(
                    channels=ch,
                    backbone_pt=weights,
                    save_path=init_pt,
                )
            else:
                print(
                    f"[eddy train] 多通道数据 channels={ch}，自 yaml 随机初始化: {arch}",
                    flush=True,
                )
                model = YOLO(arch)
            use_pretrained = False
        else:
            model = YOLO(weights)
            use_pretrained = bool(tc.get("pretrained", True))
    ms = cfg["model"]["input_size"]
    imgsz = int(ms[0]) if isinstance(ms, (list, tuple)) else int(ms)
    device = tc.get("device", "cuda")
    if device == "cuda":
        device = 0

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
        resume=bool(args.resume),
    )
    train_kw.update(extras)
    train_kw["verbose"] = bool(tc.get("verbose", True))
    print(
        "[eddy train] 启动 Ultralytics；若干步后应出现 epoch/tqdm。"
        "若无输出可试: python -u -m src.eddy.train ...",
        flush=True,
    )
    model.train(**train_kw)

    # 便于 eval 脚本固定路径：复制 run 内 best/last 到 output_dir 根
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
