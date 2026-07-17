"""Low-light enhancement: CLAHE on luminance + gamma brightening."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def enhance_low_light(
    image: np.ndarray,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: int = 8,
    brighten_gamma: float = 0.6,
) -> tuple[np.ndarray, dict[str, Any]]:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=float(clahe_clip_limit),
        tileGridSize=(int(clahe_tile_size), int(clahe_tile_size)),
    )
    l2 = clahe.apply(l)
    merged = cv2.merge([l2, a, b])
    out = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    img = out.astype(np.float32) / 255.0
    bright = np.power(np.clip(img, 0.0, 1.0), float(brighten_gamma))
    out = (bright * 255.0).astype(np.uint8)
    meta = {
        "enhancement": "clahe_gamma",
        "clahe_clip_limit": float(clahe_clip_limit),
        "clahe_tile_size": int(clahe_tile_size),
        "brighten_gamma": float(brighten_gamma),
        "method_used": "clahe_gamma",
    }
    return out, meta
