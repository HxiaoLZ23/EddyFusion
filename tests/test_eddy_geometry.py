"""涡旋实例几何属性（面积/周长/类型/轮廓）单元测试。"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.eddy.geometry import (
    EDDY_TYPE_ZH,
    _mask_to_geometry,
    eddy_type_label,
    geometry_to_stats_row,
)


def test_eddy_type_label() -> None:
    assert eddy_type_label(0, zh=True) == EDDY_TYPE_ZH[0]
    assert eddy_type_label(1, zh=True) == EDDY_TYPE_ZH[1]
    assert eddy_type_label(None, zh=True) is None


def test_mask_geometry_disk_has_area_perimeter_contour() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    cv2.circle(mask, (32, 32), 12, 255, thickness=-1)
    g = _mask_to_geometry(mask, conf=0.9, cls_id=1)
    assert g["area_pixels"] > 400
    assert g["perimeter_px"] > 50
    assert g["eddy_type"] == EDDY_TYPE_ZH[1]
    assert len(g["contour_xy"]) >= 3
    assert g["centroid_xy"] is not None
    cx, cy = g["centroid_xy"]
    assert abs(cx - 32) < 2 and abs(cy - 32) < 2


def test_geometry_to_stats_row_roundtrip() -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)
    cv2.rectangle(mask, (4, 4), (20, 20), 255, thickness=-1)
    g = _mask_to_geometry(mask, conf=0.75, cls_id=0)
    row = geometry_to_stats_row(g, 1)
    assert row["id"] == 1
    assert row["area_px"] > 0
    assert row["perimeter_px"] > 0
    assert row["eddy_type"] == EDDY_TYPE_ZH[0]
    assert isinstance(row["contour_xy"], list)
    assert row["confidence"] == pytest.approx(0.75)
