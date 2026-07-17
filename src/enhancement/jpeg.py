"""JPEG artifact reduction via bilateral filtering."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def enhance_jpeg(
    image: np.ndarray,
    bilateral_d: int = 9,
    bilateral_sigma_color: float = 75.0,
    bilateral_sigma_space: float = 75.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    out = cv2.bilateralFilter(
        image,
        d=int(bilateral_d),
        sigmaColor=float(bilateral_sigma_color),
        sigmaSpace=float(bilateral_sigma_space),
    )
    meta = {
        "enhancement": "bilateral",
        "method_used": "bilateral",
        "bilateral_d": int(bilateral_d),
        "bilateral_sigma_color": float(bilateral_sigma_color),
        "bilateral_sigma_space": float(bilateral_sigma_space),
    }
    return out, meta
