"""YOLOv8-seg：训练/评估由 ultralytics 与 `src/eddy/train.py` 驱动。

- 3 通道：标准 ``YOLO(weights)`` + COCO 预训练；
- ≥4 通道：``YOLO(config/model.architecture_yaml)``，数据集 ``channels`` 与 ``*.npy`` 对齐，见 ``src/eddy/stacked_physics.py``。
"""

from __future__ import annotations


def build_model(config: dict):  # noqa: ARG001
    """若需程序化构建，请使用 ``from ultralytics import YOLO``。"""
    raise NotImplementedError(
        "请直接使用 ultralytics.YOLO；3ch 见 config/eddy.yaml，8ch 物理堆叠见 config/eddy_enh.yaml"
    )
