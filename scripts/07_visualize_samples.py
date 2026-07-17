#!/usr/bin/env python
"""Qualitative sample grids."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.device import resolve_device
from src.common.logging_utils import setup_logging
from src.viz.qualitative import make_sample_grids


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--severity", default="medium")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-examples", type=int, default=1)
    args = p.parse_args()
    device = resolve_device(args.device)
    make_sample_grids(severity=args.severity, device=device, max_examples=args.max_examples)


if __name__ == "__main__":
    main()
