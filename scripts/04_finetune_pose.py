#!/usr/bin/env python
"""Fine-tune YOLO-Pose (prefer Colab GPU if local CUDA unavailable)."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.device import resolve_device
from src.common.logging_utils import setup_logging
from src.train.pose_finetune import fine_tune_pose


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--resume-training", action="store_true")
    args = p.parse_args()
    device = resolve_device(args.device)
    logging.info("Resolved device: %s", device)
    best = fine_tune_pose(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        patience=args.patience,
        workers=args.workers,
        resume_training=args.resume_training,
    )
    print(best)


if __name__ == "__main__":
    main()
