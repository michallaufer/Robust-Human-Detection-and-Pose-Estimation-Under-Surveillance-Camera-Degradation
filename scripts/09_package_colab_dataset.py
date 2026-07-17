#!/usr/bin/env python
"""Package YOLO-pose training subset for Colab (no full COCO archives)."""
from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.io import ensure_dir, project_path
from src.common.logging_utils import setup_logging


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(project_path("data", "processed", "colab_yolo_pose.zip")))
    args = p.parse_args()

    yolo = project_path("data", "yolo_pose")
    if not yolo.exists():
        raise SystemExit(f"Missing {yolo}. Export training data first.")

    out = Path(args.out)
    ensure_dir(out.parent)
    include = [
        yolo,
        project_path("configs", "train_pose.yaml"),
        project_path("configs", "distortions.yaml"),
        project_path("requirements.txt"),
    ]
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in include:
            item = Path(item)
            if item.is_file():
                zf.write(item, arcname=item.relative_to(ROOT).as_posix())
            elif item.is_dir():
                for f in item.rglob("*"):
                    if f.is_file():
                        # skip huge raw if any
                        if f.suffix.lower() in {".zip"}:
                            continue
                        zf.write(f, arcname=f.relative_to(ROOT).as_posix())
    logging.info("Wrote %s (%.1f MB)", out, out.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
