"""频域/空域高频增强（推理前预处理，用于可视化与轻量消融，非训练内置支路）。"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

Mode = Literal["none", "laplacian", "unsharp"]


def enhance_bgr_frequency(image_bgr: np.ndarray, *, mode: Mode = "unsharp", amount: float = 0.7) -> np.ndarray:
    """
    mode:
    - laplacian: 将拉普拉斯边缘按系数叠回亮度，突出边界；
    - unsharp: Unsharp masking（与高斯模糊的差分）；
    - none: 原样返回副本。
    """
    if mode == "none" or amount <= 0:
        return image_bgr.copy()
    img = image_bgr.astype(np.float32)
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    if mode == "laplacian":
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        sharp = gray + amount * lap
        sharp = np.clip(sharp, 0, 255)
        merged = cv2.merge([sharp, sharp, sharp])
        out = 0.5 * img + 0.5 * merged
    else:
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2)
        high = gray - blur
        sharp = gray + amount * high
        sharp = np.clip(sharp, 0, 255)
        merged = cv2.merge([sharp, sharp, sharp])
        out = 0.6 * img + 0.4 * merged
    return np.clip(out, 0, 255).astype(np.uint8)
