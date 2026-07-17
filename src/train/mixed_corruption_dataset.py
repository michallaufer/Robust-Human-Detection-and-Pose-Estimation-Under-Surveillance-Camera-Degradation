"""Pre-generate mixed-corruption YOLO training images (labels unchanged)."""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm

from src.common.constants import DISTORTION_NAMES, SEVERITIES
from src.common.io import ensure_dir, load_yaml, project_path
from src.common.reproducibility import seeded_rng
from src.distortions.registry import apply_distortion

logger = logging.getLogger(__name__)


def assign_condition(image_id: int, seed: int) -> tuple[str, str | None, str | None]:
    """Deterministic 25/25/25/25 mixture assignment."""
    rng = seeded_rng(seed, image_id, "mix")
    choice = str(rng.choice(["clean", *DISTORTION_NAMES], p=[0.25, 0.25, 0.25, 0.25]))
    if choice == "clean":
        return "clean", None, None
    sev = str(rng.choice(list(SEVERITIES)))
    return f"{choice}_{sev}", choice, sev


def build_mixed_train_set(seed: int = 42) -> Path:
    """
    Copy/corrupt train images into data/yolo_pose/images/train_mixed and write manifest.

    Labels are reused from data/yolo_pose/labels/train (geometry unchanged).
    """
    cfg = load_yaml(project_path("configs", "dataset.yaml"))
    yolo_root = project_path(cfg["yolo_pose_dir"])
    src_img = yolo_root / "images" / "train"
    src_lbl = yolo_root / "labels" / "train"
    if not src_img.exists():
        raise FileNotFoundError(f"Missing {src_img}. Run YOLO export first.")

    out_img = ensure_dir(yolo_root / "images" / "train_mixed")
    out_lbl = ensure_dir(yolo_root / "labels" / "train_mixed")
    manifest_path = yolo_root / "train_manifest.csv"

    rows: list[dict[str, Any]] = []
    images = sorted([p for p in src_img.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    dist_cfg = load_yaml(project_path("configs", "distortions.yaml"))

    for im_path in tqdm(images, desc="mix-train"):
        # Try parse COCO-style id from stem if numeric
        try:
            image_id = int(im_path.stem)
        except ValueError:
            image_id = abs(hash(im_path.stem)) % (10**9)
        cond, distortion, severity = assign_condition(image_id, seed)
        image = cv2.imread(str(im_path))
        if image is None:
            continue
        meta: dict[str, Any] = {"condition": cond}
        if distortion is None:
            out = image
        else:
            # For motion blur training, optional random angle
            if distortion == "motion_blur" and dist_cfg.get("motion_blur", {}).get("train_random_angle", True):
                rng = seeded_rng(seed, image_id, "angle")
                angle = float(rng.uniform(0, 180))
                # temporarily override
                tmp_cfg = load_yaml(project_path("configs", "distortions.yaml"))
                tmp_cfg["motion_blur"][severity]["angle"] = angle
                out, meta = apply_distortion(image, distortion, severity, seed=seed, image_id=image_id, cfg=tmp_cfg)
            else:
                out, meta = apply_distortion(image, distortion, severity, seed=seed, image_id=image_id, cfg=dist_cfg)
        out_name = im_path.name
        cv2.imwrite(str(out_img / out_name), out)
        # copy labels
        lbl = src_lbl / f"{im_path.stem}.txt"
        dst_lbl = out_lbl / f"{im_path.stem}.txt"
        if lbl.exists():
            dst_lbl.write_text(lbl.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            dst_lbl.write_text("", encoding="utf-8")
        rows.append(
            {
                "original_image_id": image_id,
                "source_file": str(im_path.as_posix()),
                "output_file": str((out_img / out_name).as_posix()),
                "assigned_condition": cond,
                "distortion": distortion or "",
                "severity": severity or "",
                "parameters": str({k: v for k, v in meta.items() if k != "psf"}),
                "seed": seed,
            }
        )

    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    # Point dataset yaml train to mixed
    from src.common.io import save_yaml

    save_yaml(
        yolo_root / "dataset.yaml",
        {
            "path": str(yolo_root.resolve().as_posix()),
            "train": "images/train_mixed",
            "val": "images/val",
            "kpt_shape": [17, 3],
            "flip_idx": [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
            "names": {0: "person"},
        },
    )
    # Also keep labels path: Ultralytics maps images/train_mixed -> labels/train_mixed automatically
    logger.info("Wrote mixed train set (%s images) and %s", len(rows), manifest_path)

    # Validate mixture proportions approximately
    if rows:
        from collections import Counter

        fam = Counter()
        for r in rows:
            if not r["distortion"]:
                fam["clean"] += 1
            else:
                fam[r["distortion"]] += 1
        logger.info("Mix counts: %s", dict(fam))
    return manifest_path
