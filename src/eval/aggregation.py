"""Aggregate helpers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.eval.condition_runner import aggregate_csv


def load_aggregate(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


__all__ = ["aggregate_csv", "load_aggregate"]
