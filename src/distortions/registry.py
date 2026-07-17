"""Distortion registry."""
from __future__ import annotations

from typing import Any

import numpy as np

from src.common.io import load_yaml, project_path
from src.common.reproducibility import seeded_rng
from src.distortions.jpeg import apply_jpeg
from src.distortions.low_light import apply_low_light
from src.distortions.motion_blur import apply_motion_blur


def default_cfg() -> dict[str, Any]:
    return load_yaml(project_path("configs", "distortions.yaml"))


def compute_mse_psnr(clean: np.ndarray, degraded: np.ndarray) -> tuple[float, float]:
    a = clean.astype(np.float64)
    b = degraded.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return mse, 99.0
    psnr = float(20.0 * np.log10(255.0 / np.sqrt(mse)))
    return mse, psnr


def apply_distortion(
    image: np.ndarray,
    name: str,
    severity: str,
    seed: int = 42,
    image_id: int | str = 0,
    cfg: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Apply named distortion at severity.

    Returns (distorted_image, metadata) including mse/psnr vs clean when attached
    by the caller via attach_quality().
    """
    cfg = cfg or default_cfg()
    if name not in cfg:
        raise ValueError(f"Unknown distortion: {name}")
    if severity not in cfg[name]:
        raise ValueError(f"Unknown severity '{severity}' for {name}")
    params = dict(cfg[name][severity])
    rng = seeded_rng(seed, image_id, name, severity)

    if name == "low_light":
        out, meta = apply_low_light(
            image,
            gamma=float(params["gamma"]),
            brightness_scale=float(params.get("brightness_scale", 1.0)),
        )
    elif name == "motion_blur":
        angle = float(params.get("angle", 0.0))
        if bool(cfg.get("motion_blur", {}).get("train_random_angle", False)):
            # Evaluation uses fixed angle from config; training may override elsewhere
            pass
        # Deterministic angle jitter only if config requests and angle is null
        if params.get("angle") is None:
            angle = float(rng.uniform(0, 180))
        out, meta = apply_motion_blur(
            image,
            kernel_length=int(params["kernel_length"]),
            angle=angle,
        )
    elif name == "jpeg":
        out, meta = apply_jpeg(image, quality=int(params["quality"]))
    else:
        raise ValueError(f"Unsupported distortion: {name}")

    meta["severity"] = severity
    mse, psnr = compute_mse_psnr(image, out)
    meta["mse"] = mse
    meta["psnr"] = psnr
    return out, meta


def list_conditions(include_clean: bool = True) -> list[tuple[str, str | None, str | None]]:
    """Return list of (condition_name, distortion|None, severity|None)."""
    from src.common.constants import DISTORTION_NAMES, SEVERITIES

    rows: list[tuple[str, str | None, str | None]] = []
    if include_clean:
        rows.append(("clean", None, None))
    for d in DISTORTION_NAMES:
        for s in SEVERITIES:
            rows.append((f"{d}_{s}", d, s))
    return rows
