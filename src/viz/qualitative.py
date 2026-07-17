"""Qualitative comparison grids."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from src.common.constants import DISTORTION_NAMES
from src.common.io import ensure_dir, load_yaml, project_path, save_json
from src.dataset.coco_masks import silhouette_boundary, union_person_mask
from src.dataset.coco_person import anns_to_boxes, anns_to_keypoints, iter_split
from src.distortions.registry import apply_distortion
from src.enhancement.registry import apply_enhancement
from src.eval.boundary_metrics import canny_edges
from src.tasks.boundary import load_yolo
from src.viz.overlays import draw_boxes, draw_keypoints, overlay_edges

logger = logging.getLogger(__name__)


def make_sample_grids(
    severity: str = "medium",
    device: str = "auto",
    max_examples: int = 1,
    split: str = "val",
) -> list[Path]:
    cfg = load_yaml(project_path("configs", "distortions.yaml"))
    out_dir = ensure_dir(project_path("results", "coco_person", "qualitative"))
    det, _ = load_yolo("yolov8n.pt", device=device)
    pose, _ = load_yolo("yolov8n-pose.pt", device=device)
    paths = []
    meta = []
    for i, (image_id, image, info, anns, coco) in enumerate(iter_split(split, max_images=max_examples)):
        h, w = image.shape[:2]
        mask = union_person_mask(anns, coco, h, w)
        gt_b = silhouette_boundary(mask)
        gt_boxes = anns_to_boxes(anns)
        gt_kpts = anns_to_keypoints(anns)
        panels = []
        for dist in DISTORTION_NAMES:
            deg, dmeta = apply_distortion(image, dist, severity, seed=42, image_id=image_id, cfg=cfg)
            enh, emeta = apply_enhancement(deg, dist, dmeta, cfg=cfg)
            edges = canny_edges(deg)
            # predictions on distorted
            r = pose.predict(deg, verbose=False)[0]
            p_boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None and len(r.boxes) else np.zeros((0, 4))
            p_kpts = []
            if r.keypoints is not None and r.keypoints.data is not None:
                k = r.keypoints.data.cpu().numpy()
                for j in range(k.shape[0]):
                    arr = np.zeros((17, 3), dtype=np.float32)
                    arr[:, :2] = k[j, :, :2]
                    arr[:, 2] = 1
                    p_kpts.append(arr)
            row_imgs = [
                draw_boxes(image, gt_boxes, (255, 0, 0)),
                draw_boxes(deg, p_boxes),
                draw_boxes(enh, p_boxes, (0, 255, 255)),
                overlay_edges(deg, edges),
                overlay_edges(image, gt_b, (0, 0, 255)),
                draw_keypoints(draw_boxes(deg, p_boxes), p_kpts),
            ]
            hh = 220
            resized = []
            for im in row_imgs:
                scale = hh / im.shape[0]
                resized.append(cv2.resize(im, (int(im.shape[1] * scale), hh)))
            max_w = max(im.shape[1] for im in resized)
            padded = []
            for im in resized:
                canvas = np.zeros((hh, max_w, 3), dtype=np.uint8)
                canvas[:, : im.shape[1]] = im
                padded.append(canvas)
            panels.append(np.hstack(padded))
            meta.append({"image_id": image_id, "distortion": dist, "severity": severity, "enh": emeta})
        grid = np.vstack(panels)
        out = out_dir / f"sample_{image_id}_{severity}.png"
        cv2.imwrite(str(out), grid)
        paths.append(out)
        if i + 1 >= max_examples:
            break
    save_json(out_dir / "qualitative_meta.json", meta)
    logger.info("Wrote qualitative grids: %s", paths)
    return paths
