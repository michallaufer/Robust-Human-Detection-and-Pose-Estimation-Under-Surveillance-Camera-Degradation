#!/usr/bin/env python
"""Export YOLO-Pose labels and build mixed-corruption train set."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.logging_utils import setup_logging
from src.common.reproducibility import set_seed
from src.dataset.yolo_pose_export import export_yolo_pose
from src.train.mixed_corruption_dataset import build_mixed_train_set


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-export", action="store_true")
    args = p.parse_args()
    set_seed(args.seed)
    if not args.skip_export:
        export_yolo_pose("val")
        try:
            export_yolo_pose("train")
        except FileNotFoundError as exc:
            logging.warning("%s", exc)
    build_mixed_train_set(seed=args.seed)


if __name__ == "__main__":
    main()
