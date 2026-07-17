"""Motion-blur enhancement using stable unsharp masking."""
from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def unsharp_mask(
    image: np.ndarray,
    amount: float = 0.6,
    sigma: float = 1.2,
) -> np.ndarray:
    blur = cv2.GaussianBlur(image, (0, 0), sigma)
    sharp = cv2.addWeighted(image, 1.0 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def enhance_motion_blur(
    image: np.ndarray,
    kernel_length: int = 15,
    angle: float = 0.0,
    method: str = "unsharp",
    unsharp_amount: float = 0.6,
    unsharp_sigma: float = 1.2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Enhance blurred images; stable unsharp masking is the project method."""
    if str(method).lower() != "unsharp":
        logger.warning("Unsupported motion enhancement '%s'; using unsharp masking", method)
    out = unsharp_mask(image, amount=unsharp_amount, sigma=unsharp_sigma)

    meta = {
        "enhancement": "motion_deblur",
        "requested_method": method,
        "method_used": "stable_unsharp_mask",
        "kernel_length": int(kernel_length),
        "angle": float(angle),
        "unsharp_amount": float(unsharp_amount),
        "unsharp_sigma": float(unsharp_sigma),
    }
    return out, meta
