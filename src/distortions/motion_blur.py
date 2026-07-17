"""Motion blur with line-shaped PSF."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def motion_kernel(length: int, angle: float) -> np.ndarray:
    length = max(1, int(length))
    if length == 1:
        return np.array([[1.0]], dtype=np.float32)
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    center = (length / 2.0 - 0.5, length / 2.0 - 0.5)
    rot = cv2.getRotationMatrix2D(center, float(angle), 1.0)
    kernel = cv2.warpAffine(kernel, rot, (length, length))
    s = float(kernel.sum())
    if s > 0:
        kernel /= s
    return kernel


def apply_motion_blur(
    image: np.ndarray,
    kernel_length: int = 15,
    angle: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    kernel = motion_kernel(kernel_length, angle)
    out = cv2.filter2D(image, -1, kernel)
    meta = {
        "distortion": "motion_blur",
        "kernel_length": int(kernel_length),
        "angle": float(angle),
        "psf": kernel.tolist(),
    }
    return out, meta
