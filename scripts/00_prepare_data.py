#!/usr/bin/env python
"""Prepare COCO person subset via selective image download (never full train zip)."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.device import device_info
from src.common.io import load_yaml, project_path, save_json
from src.common.logging_utils import setup_logging
from src.common.reproducibility import set_seed
from src.dataset.selective_download import download_selected_images, ensure_annotations
from src.dataset.splits import select_person_images, write_split_artifacts
from src.dataset.yolo_pose_export import export_yolo_pose


def main() -> None:
    setup_logging()
    log = logging.getLogger("prepare")
    p = argparse.ArgumentParser(description="Selective COCO person subset preparation")
    p.add_argument("--train-n", type=int, default=None)
    p.add_argument("--val-n", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-download", action="store_true", help="Skip annotation zip download")
    p.add_argument("--skip-train-images", action="store_true")
    p.add_argument("--skip-val-images", action="store_true")
    p.add_argument("--download-selected-train", action="store_true")
    p.add_argument("--download-selected-val", action="store_true")
    p.add_argument("--export-yolo", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--verify-files", action="store_true", default=True)
    args = p.parse_args()

    cfg = load_yaml(project_path("configs", "dataset.yaml"))
    set_seed(args.seed)
    train_n = args.train_n if args.train_n is not None else int(cfg["train_images"])
    val_n = args.val_n if args.val_n is not None else int(cfg["val_images"])

    log.info("Environment: %s", device_info())
    log.info("NOTE: This script never downloads full train2017.zip / val2017.zip archives.")

    if not args.skip_download:
        ensure_annotations(force=args.force)
    else:
        # Still repair demo-overwritten tiny annotation JSONs from existing zip
        try:
            ensure_annotations(force=False)
        except Exception as exc:
            log.warning("Annotation check skipped/failed under --skip-download: %s", exc)

    # Always build val split metadata if val_n > 0
    if val_n > 0:
        val_rows = select_person_images("val", n=val_n, seed=args.seed, require_keypoints=True)
        write_split_artifacts("val", val_rows)
        if args.download_selected_val and not args.skip_val_images:
            dest = project_path(cfg["selected_root"], "val2017")
            stats = download_selected_images(
                [r["file_name"] for r in val_rows],
                split="val",
                dest_dir=dest,
                force=args.force,
                verify=args.verify_files,
            )
            log.info("Val download: %s", stats)
            if stats["failed"]:
                raise SystemExit(f"Failed to download {len(stats['failed'])} val images")

    if train_n > 0 and not args.skip_train_images:
        train_rows = select_person_images("train", n=train_n, seed=args.seed, require_keypoints=True)
        write_split_artifacts("train", train_rows)
        if args.download_selected_train:
            dest = project_path(cfg["selected_root"], "train2017")
            stats = download_selected_images(
                [r["file_name"] for r in train_rows],
                split="train",
                dest_dir=dest,
                force=args.force,
                verify=args.verify_files,
            )
            log.info("Train download: %s", stats)
            if stats["failed"]:
                raise SystemExit(f"Failed to download {len(stats['failed'])} train images")
    elif args.skip_train_images:
        log.info("Skipping train image selection/download")

    if args.export_yolo:
        if (project_path(cfg["splits_dir"]) / "val_ids.json").exists():
            export_yolo_pose("val")
        if (project_path(cfg["splits_dir"]) / "train_ids.json").exists():
            export_yolo_pose("train")

    save_json(
        project_path("results", "coco_person", "logs", "prepare_meta.json"),
        {"train_n": train_n, "val_n": val_n, "seed": args.seed},
    )
    log.info("Done.")


if __name__ == "__main__":
    main()
