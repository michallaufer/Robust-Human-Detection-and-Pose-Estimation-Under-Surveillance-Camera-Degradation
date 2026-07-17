"""Reproducibility helpers."""
from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def seeded_rng(seed: int, *parts: int | str) -> np.random.Generator:
    """Deterministic RNG from global seed and optional parts (e.g. image_id)."""
    material = f"{seed}|" + "|".join(str(p) for p in parts)
    digest = hashlib.md5(material.encode("utf-8")).hexdigest()
    derived = int(digest[:8], 16)
    return np.random.default_rng(derived)
