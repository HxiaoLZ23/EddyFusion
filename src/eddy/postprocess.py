"""涡旋实例后处理（几何属性等）。"""

from __future__ import annotations

from typing import Any

from src.eddy.geometry import geometries_from_ultralytics_result


def extract_instance_geometries(pred: Any, image_shape_hw: tuple[int, int]) -> list[dict[str, Any]]:
    """
    pred: ultralytics Results（单张）。
    image_shape_hw: (H, W) 与输入 BGR 一致。
    """
    return geometries_from_ultralytics_result(pred, image_shape_hw)


def postprocess_masks(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("postprocess_masks：请使用 extract_instance_geometries")
