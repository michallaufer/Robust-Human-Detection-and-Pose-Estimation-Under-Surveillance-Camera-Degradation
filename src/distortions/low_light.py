"""Low-light gamma darkening."""
from __future__ import annotations

from typing import Any

import numpy as np


def apply_low_light(
    image: np.ndarray,
    gamma: float = 2.2,
    brightness_scale: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Darken with I_out = (I/255)^gamma * brightness_scale * 255.

    Returns (uint8 BGR image, metadata).
    """
    if image.ndim != 3:
        raise ValueError("Expected HxWxC image")
    img = image.astype(np.float32) / 255.0
    dark = np.power(np.clip(img, 0.0, 1.0), float(gamma))
    dark = np.clip(dark * float(brightness_scale), 0.0, 1.0)
    out = (dark * 255.0).astype(np.uint8)
    meta = {
        "distortion": "low_light",
        "gamma": float(gamma),
        "brightness_scale": float(brightness_scale),
    }
    return out, meta
