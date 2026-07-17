"""Enhancement registry matched to distortion families."""
from __future__ import annotations

from typing import Any

import numpy as np

from src.common.io import load_yaml, project_path
from src.enhancement.deblur import enhance_motion_blur
from src.enhancement.jpeg import enhance_jpeg
from src.enhancement.low_light import enhance_low_light


def default_cfg() -> dict[str, Any]:
    return load_yaml(project_path("configs", "distortions.yaml"))


def apply_enhancement(
    image: np.ndarray,
    distortion_name: str,
    distortion_meta: dict[str, Any] | None = None,
    cfg: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    full = cfg or default_cfg()
    enh = full["enhancement"]
    distortion_meta = distortion_meta or {}

    if distortion_name == "low_light":
        c = enh["low_light"]
        return enhance_low_light(
            image,
            clahe_clip_limit=float(c["clahe_clip_limit"]),
            clahe_tile_size=int(c["clahe_tile_size"]),
            brighten_gamma=float(c["brighten_gamma"]),
        )
    if distortion_name == "motion_blur":
        c = enh["motion_blur"]
        return enhance_motion_blur(
            image,
            kernel_length=int(distortion_meta.get("kernel_length", 15)),
            angle=float(distortion_meta.get("angle", 0.0)),
            method=str(c.get("method", "unsharp")),
            unsharp_amount=float(c.get("unsharp_amount", 0.6)),
            unsharp_sigma=float(c.get("unsharp_sigma", 1.2)),
        )
    if distortion_name == "jpeg":
        c = enh["jpeg"]
        return enhance_jpeg(
            image,
            bilateral_d=int(c["bilateral_d"]),
            bilateral_sigma_color=float(c["bilateral_sigma_color"]),
            bilateral_sigma_space=float(c["bilateral_sigma_space"]),
        )
    raise ValueError(f"No enhancement for distortion: {distortion_name}")
