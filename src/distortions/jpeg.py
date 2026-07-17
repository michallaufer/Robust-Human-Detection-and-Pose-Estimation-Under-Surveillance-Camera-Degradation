"""In-memory JPEG compression."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def apply_jpeg(image: np.ndarray, quality: int = 20) -> tuple[np.ndarray, dict[str, Any]]:
    quality = int(np.clip(quality, 1, 100))
    ok, enc = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if out is None:
        raise RuntimeError("JPEG decode failed")
    meta = {"distortion": "jpeg", "quality": quality}
    return out, meta
