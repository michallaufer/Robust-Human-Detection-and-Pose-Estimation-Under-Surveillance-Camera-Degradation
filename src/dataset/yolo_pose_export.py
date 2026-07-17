"""Export COCO person subset to Ultralytics YOLO-Pose layout."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.common.constants import FLIP_IDX
from src.common.io import ensure_dir, load_yaml, project_path, save_yaml
from src.dataset.coco_person import load_coco
from src.dataset.splits import load_split_manifest

logger = logging.getLogger(__name__)


def _clip01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


def annotation_to_yolo_line(ann: dict[str, Any], width: int, height: int) -> str | None:
    x, y, bw, bh = ann["bbox"]
    if bw <= 1 or bh <= 1:
        return None
    # clip box to image
    x2 = min(width, x + bw)
    y2 = min(height, y + bh)
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw <= 1 or bh <= 1:
        return None
    cx = _clip01((x1 + bw / 2) / width)
    cy = _clip01((y1 + bh / 2) / height)
    nw = _clip01(bw / width)
    nh = _clip01(bh / height)
    kps = np.asarray(ann.get("keypoints", [0] * 51), dtype=np.float32).reshape(17, 3)
    parts = [f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"]
    for kx, ky, kv in kps:
        if kv <= 0:
            parts.extend(["0.000000", "0.000000", "0"])
            continue
        # clip coords; keep visibility (COCO 1/2 -> YOLO 1)
        nx = _clip01(float(kx) / width)
        ny = _clip01(float(ky) / height)
        vis = 1
        parts.extend([f"{nx:.6f}", f"{ny:.6f}", str(vis)])
    return " ".join(parts)


def export_yolo_pose(split: str = "val", link_images: bool = True) -> Path:
    cfg = load_yaml(project_path("configs", "dataset.yaml"))
    meta = load_split_manifest(split)
    coco = load_coco(meta["ann_file"])
    out_root = ensure_dir(project_path(cfg["yolo_pose_dir"]))
    img_out = ensure_dir(out_root / "images" / split)
    lbl_out = ensure_dir(out_root / "labels" / split)
    src_dir = Path(meta["image_dir"])
    cat = int(meta["person_category_id"])

    n_img = 0
    n_lbl = 0
    for image_id in tqdm(meta["image_ids"], desc=f"yolo-export-{split}"):
        info = coco.loadImgs([image_id])[0]
        src = src_dir / info["file_name"]
        if not src.exists():
            logger.warning("Missing image %s", src)
            continue
        stem = Path(info["file_name"]).stem
        dst = img_out / info["file_name"]
        if not dst.exists():
            if link_images:
                try:
                    dst.symlink_to(src.resolve())
                except OSError:
                    shutil.copy2(src, dst)
            else:
                shutil.copy2(src, dst)
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[image_id], catIds=[cat], iscrowd=False))
        lines = []
        for a in anns:
            line = annotation_to_yolo_line(a, info["width"], info["height"])
            if line:
                lines.append(line)
        with open(lbl_out / f"{stem}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        n_img += 1
        n_lbl += 1

    yaml_path = out_root / "dataset.yaml"
    save_yaml(
        yaml_path,
        {
            "path": str(out_root.resolve().as_posix()),
            "train": "images/train",
            "val": "images/val",
            "kpt_shape": [17, 3],
            "flip_idx": FLIP_IDX,
            "names": {0: "person"},
        },
    )
    logger.info("Exported %s images for split=%s -> %s", n_img, split, out_root)
    if n_img != n_lbl:
        raise RuntimeError("Image/label count mismatch")
    return out_root
