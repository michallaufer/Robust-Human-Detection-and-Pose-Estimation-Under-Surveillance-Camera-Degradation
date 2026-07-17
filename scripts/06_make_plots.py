#!/usr/bin/env python
"""Generate figures from validated coco_person aggregate tables."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.logging_utils import setup_logging
from src.common.io import project_path
from src.viz.plots import make_all_plots


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--aggregate", default=None)
    args = p.parse_args()
    path = Path(args.aggregate) if args.aggregate else project_path("results", "coco_person", "tables", "val_aggregate.csv")
    make_all_plots(path)


if __name__ == "__main__":
    main()
