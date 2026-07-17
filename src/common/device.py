"""Device resolution with safe CUDA fallback."""
from __future__ import annotations

import logging
import warnings

logger = logging.getLogger(__name__)


def resolve_device(requested: str | int | None = "auto") -> str:
    """
    Resolve a user-requested device string for Ultralytics / PyTorch.

    Supported: ``auto``, ``cpu``, ``0`` (GPU index 0).

    If CUDA GPU ``0`` is requested but unavailable, warn and fall back to ``cpu``.
    """
    if requested is None:
        requested = "auto"
    req = str(requested).strip().lower()

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyTorch is required for device resolution") from exc

    cuda_ok = bool(torch.cuda.is_available())

    if req == "auto":
        resolved = "0" if cuda_ok else "cpu"
        logger.info("resolve_device(auto) -> %s (cuda_available=%s)", resolved, cuda_ok)
        return resolved

    if req == "cpu":
        return "cpu"

    if req == "0":
        if cuda_ok:
            return "0"
        msg = (
            "Requested --device 0 but torch.cuda.is_available() is False. "
            "Falling back to CPU. Install a CUDA build of PyTorch or use Colab for GPU training."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        logger.warning(msg)
        return "cpu"

    raise ValueError(
        f"Unsupported device '{requested}'. Use one of: auto, cpu, 0"
    )


def device_info() -> dict[str, str | bool | int | None]:
    """Collect environment device information for logging / README."""
    import platform
    import sys

    import torch

    info: dict[str, str | bool | int | None] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "gpu_name": None,
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except Exception:  # pragma: no cover
        info["ultralytics"] = None
    return info
