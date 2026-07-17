#!/usr/bin/env python
"""Distorted + enhanced evaluation grid."""
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
    log = logging.getLogger("distort")
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--tasks", default="boundary,detection,pose")
    p.add_argument("--conditions", default=None, help="Comma list or omit for full matrix")
    p.add_argument("--split", default="val")
    p.add_argument("--no-enhance", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    log.info("Resolved device: %s", device)

    if args.conditions:
        conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    else:
        conditions = ["clean"]
        for d in DISTORTION_NAMES:
            for s in SEVERITIES:
                conditions.append(f"{d}_{s}")
                if not args.no_enhance:
                    conditions.append(f"enhanced_{d}_{s}")

    path = run_evaluation(
        split=args.split,
        tasks=[t.strip() for t in args.tasks.split(",") if t.strip()],
        conditions=conditions,
        max_images=args.max_images,
        device=device,
        imgsz=args.imgsz,
        seed=args.seed,
        resume=args.resume,
        output_dir=args.output_dir,
        include_enhanced=not args.no_enhance,
    )
    log.info("Results: %s", path)


if __name__ == "__main__":
    main()
