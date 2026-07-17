#!/usr/bin/env python
"""Evaluate fine-tuned pose model on clean/distorted/enhanced conditions."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.constants import DISTORTION_NAMES, SEVERITIES
from src.common.device import resolve_device
from src.common.logging_utils import setup_logging
from src.common.reproducibility import set_seed
from src.eval.condition_runner import run_evaluation


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--split", default="val")
    args = p.parse_args()

    if not Path(args.weights).exists():
        raise SystemExit(f"Weights not found: {args.weights}")

    set_seed(args.seed)
    device = resolve_device(args.device)
    logging.info("Resolved device: %s", device)

    conditions = ["clean"]
    for d in DISTORTION_NAMES:
        for s in SEVERITIES:
            conditions.append(f"{d}_{s}")
            conditions.append(f"enhanced_{d}_{s}")

    path = run_evaluation(
        split=args.split,
        tasks=["pose"],
        conditions=conditions,
        max_images=args.max_images,
        device=device,
        imgsz=args.imgsz,
        seed=args.seed,
        resume=args.resume,
        output_dir=args.output_dir,
        pose_weights=args.weights,
        pose_only_finetuned=True,
        include_enhanced=True,
    )
    logging.info("Finetuned eval: %s", path)


if __name__ == "__main__":
    main()
