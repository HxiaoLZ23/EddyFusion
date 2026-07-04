"""多通道 YOLO-seg：从 COCO 3 通道预训练扩展首层卷积到 N 通道。"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def _first_conv_module(model: nn.Module) -> nn.Conv2d:
    """Ultralytics YOLOv8 首层 Conv 包装内的 Conv2d。"""
    layer0 = model.model[0]
    conv = getattr(layer0, "conv", None)
    if conv is None or not isinstance(conv, nn.Conv2d):
        raise TypeError(f"无法定位首层 Conv2d，got {type(layer0)}")
    return conv


def expand_first_conv_in_channels(model: nn.Module, target_ch: int) -> None:
    """将 model 首层 in_channels 从 3 扩到 target_ch；新增通道用 RGB 均值初始化。"""
    if target_ch <= 3:
        return
    old_conv = _first_conv_module(model)
    if old_conv.in_channels == target_ch:
        return
    if old_conv.in_channels != 3:
        raise ValueError(f"仅支持从 3ch 扩展，当前 in_channels={old_conv.in_channels}")

    out_ch = old_conv.out_channels
    k = old_conv.kernel_size[0]
    stride = old_conv.stride
    padding = old_conv.padding
    bias = old_conv.bias is not None

    new_conv = nn.Conv2d(
        target_ch,
        out_ch,
        kernel_size=k,
        stride=stride,
        padding=padding,
        bias=bias,
    )
    with torch.no_grad():
        new_conv.weight[:, :3].copy_(old_conv.weight)
        mean_rgb = old_conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight[:, 3:].copy_(mean_rgb.expand(-1, target_ch - 3, -1, -1))
        if bias and old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    layer0 = model.model[0]
    layer0.conv = new_conv


def build_yolo_multichannel_from_coco(
    *,
    channels: int,
    backbone_pt: str = "yolov8n-seg.pt",
    save_path: Path | None = None,
):
    """加载 COCO 预训练 YOLO-seg，扩展首层到 ``channels`` 并可选落盘。"""
    from ultralytics import YOLO

    if channels <= 3:
        return YOLO(backbone_pt)

    yolo = YOLO(backbone_pt)
    expand_first_conv_in_channels(yolo.model, channels)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        yolo.save(str(save_path))

    return yolo


def build_yolo_multichannel_from_baseline(
    *,
    channels: int,
    baseline_pt: str | Path,
    save_path: Path | None = None,
):
    """从已训 3ch 涡旋权重扩展首层到 ``channels``（保留 backbone/head 已学特征）。"""
    from ultralytics import YOLO

    baseline_pt = Path(baseline_pt)
    if not baseline_pt.is_file():
        raise FileNotFoundError(f"3ch 基线权重不存在: {baseline_pt}")

    yolo = YOLO(str(baseline_pt))
    if channels <= 3:
        return yolo

    expand_first_conv_in_channels(yolo.model, channels)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        yolo.save(str(save_path))

    return yolo
