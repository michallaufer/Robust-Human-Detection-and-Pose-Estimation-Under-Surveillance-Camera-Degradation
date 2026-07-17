"""COCO person loading utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
from pycocotools.coco import COCO

from src.dataset.splits import load_split_manifest, person_size_category


def load_coco(ann_file: str | Path) -> COCO:
    return COCO(str(ann_file))


def anns_to_boxes(anns: list[dict[str, Any]]) -> np.ndarray:
    boxes = []
    for a in anns:
        x, y, w, h = a["bbox"]
        boxes.append([x, y, x + w, y + h])
    return np.asarray(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)


def anns_to_keypoints(anns: list[dict[str, Any]]) -> list[np.ndarray]:
    out = []
    for a in anns:
        kps = np.asarray(a.get("keypoints", [0] * 51), dtype=np.float32).reshape(17, 3)
        out.append(kps)
    return out


def anns_to_areas(anns: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([float(a.get("area", 0.0)) for a in anns], dtype=np.float32)


def iter_split(
    split: str,
    max_images: int | None = None,
) -> Iterator[tuple[int, np.ndarray, dict[str, Any], list[dict[str, Any]], COCO]]:
    """Yield (image_id, bgr, info, person_anns, coco)."""
    meta = load_split_manifest(split)
    coco = load_coco(meta["ann_file"])
    img_dir = Path(meta["image_dir"])
    cat = int(meta["person_category_id"])
    ids = list(meta["image_ids"])
    if max_images is not None:
        ids = ids[: int(max_images)]

    for image_id in ids:
        info = coco.loadImgs([image_id])[0]
        path = img_dir / info["file_name"]
        # selected downloads keep original COCO filenames
        if not path.exists():
            # fall back to zero-padded numeric names used by older demo export
            alt = img_dir / f"{int(image_id):012d}.jpg"
            path = alt if alt.exists() else path
        image = cv2.imread(str(path))
        if image is None:
            continue
        ann_ids = coco.getAnnIds(imgIds=[image_id], catIds=[cat], iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        yield int(image_id), image, info, anns, coco


def summarize_person_sizes(anns: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"small": 0, "medium": 0, "large": 0}
    for a in anns:
        counts[person_size_category(float(a.get("area", 0.0)))] += 1
    return counts
