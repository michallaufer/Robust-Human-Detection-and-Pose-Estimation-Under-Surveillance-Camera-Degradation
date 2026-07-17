"""Person subset selection and metadata."""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.common.constants import PERSON_CATEGORY_ID
from src.common.io import ensure_dir, load_json, load_yaml, project_path, save_json
from src.common.reproducibility import set_seed

logger = logging.getLogger(__name__)


def person_size_category(area: float) -> str:
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def _load_kp_coco(ann_path: Path):
    from pycocotools.coco import COCO

    return COCO(str(ann_path))


def select_person_images(
    split: str,
    n: int,
    seed: int = 42,
    require_keypoints: bool = True,
) -> list[dict[str, Any]]:
    """
    Select up to n person images with metadata rows.

    Prefers images with more visible keypoints when available.
    """
    cfg = load_yaml(project_path("configs", "dataset.yaml"))
    coco_root = project_path(cfg["coco_root"])
    key = "keypoints_train" if split == "train" else "keypoints_val"
    ann_path = coco_root / cfg["annotations"][key]
    if not ann_path.exists():
        raise FileNotFoundError(
            f"Missing {ann_path}. Run: python scripts/00_prepare_data.py (annotations only)."
        )

    coco = _load_kp_coco(ann_path)
    cat_id = int(cfg.get("person_category_id", PERSON_CATEGORY_ID))
    img_ids = coco.getImgIds(catIds=[cat_id])

    rows: list[dict[str, Any]] = []
    for image_id in img_ids:
        info = coco.loadImgs([image_id])[0]
        anns = coco.loadAnns(
            coco.getAnnIds(imgIds=[image_id], catIds=[cat_id], iscrowd=False)
        )
        if not anns:
            continue
        n_people = len(anns)
        n_with_kp = sum(1 for a in anns if a.get("num_keypoints", 0) > 0)
        if require_keypoints and n_with_kp == 0:
            continue
        vis = 0
        areas = []
        size_counts = {"small": 0, "medium": 0, "large": 0}
        for a in anns:
            areas.append(float(a.get("area", 0.0)))
            sc = person_size_category(float(a.get("area", 0.0)))
            size_counts[sc] += 1
            kps = a.get("keypoints") or []
            if len(kps) >= 51:
                arr = np.array(kps, dtype=np.float32).reshape(17, 3)
                vis += int((arr[:, 2] == 2).sum())
        largest = max(areas) if areas else 0.0
        rows.append(
            {
                "image_id": int(image_id),
                "file_name": info["file_name"],
                "width": int(info["width"]),
                "height": int(info["height"]),
                "number_of_people": n_people,
                "number_of_people_with_keypoints": n_with_kp,
                "total_visible_keypoints": vis,
                "largest_person_area": largest,
                "person_size_category_summary": (
                    f"s{size_counts['small']}_m{size_counts['medium']}_l{size_counts['large']}"
                ),
                "_score": vis * 10 + n_with_kp,
            }
        )

    if not rows:
        raise RuntimeError(f"No eligible person images found for split={split}")

    set_seed(seed)
    rng = np.random.default_rng(seed)
    # Prefer higher keypoint visibility, then random shuffle within tiers
    rows.sort(key=lambda r: (-r["_score"], r["image_id"]))
    # Take a pool larger than n then sample for diversity while keeping preference
    pool = rows[: max(n * 5, n)]
    rng.shuffle(pool)
    selected = sorted(pool[:n], key=lambda r: r["image_id"])
    for r in selected:
        r.pop("_score", None)
    logger.info("Selected %s / %s eligible images for %s", len(selected), len(rows), split)
    return selected


def write_split_artifacts(split: str, rows: list[dict[str, Any]]) -> dict[str, Path]:
    cfg = load_yaml(project_path("configs", "dataset.yaml"))
    splits_dir = ensure_dir(project_path(cfg["splits_dir"]))
    ids_path = splits_dir / f"{split}_person_ids.txt"
    meta_path = splits_dir / f"{split}_person_metadata.csv"
    json_path = splits_dir / f"{split}_ids.json"

    with open(ids_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"{r['image_id']}\n")

    fieldnames = [
        "image_id",
        "file_name",
        "width",
        "height",
        "number_of_people",
        "number_of_people_with_keypoints",
        "total_visible_keypoints",
        "largest_person_area",
        "person_size_category_summary",
    ]
    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})

    selected_root = project_path(cfg["selected_root"]) / f"{split}2017"
    coco_root = project_path(cfg["coco_root"])
    key = "keypoints_train" if split == "train" else "keypoints_val"
    payload = {
        "split": split,
        "image_ids": [r["image_id"] for r in rows],
        "n": len(rows),
        "person_category_id": int(cfg.get("person_category_id", PERSON_CATEGORY_ID)),
        "ann_file": str((coco_root / cfg["annotations"][key]).as_posix()),
        "image_dir": str(selected_root.as_posix()),
        "dataset_name": "coco_person",
        "file_names": [r["file_name"] for r in rows],
    }
    save_json(json_path, payload)
    return {"ids": ids_path, "metadata": meta_path, "json": json_path}


def load_split_manifest(split: str) -> dict[str, Any]:
    cfg = load_yaml(project_path("configs", "dataset.yaml"))
    path = project_path(cfg["splits_dir"]) / f"{split}_ids.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing split manifest: {path}")
    return load_json(path)
