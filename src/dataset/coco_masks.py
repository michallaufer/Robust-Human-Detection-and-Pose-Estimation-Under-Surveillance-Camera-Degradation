"""COCO person mask helpers via pycocotools."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def instance_masks(anns: list[dict[str, Any]], coco, height: int, width: int) -> list[np.ndarray]:
    masks = []
    for ann in anns:
        m = coco.annToMask(ann).astype(np.uint8)
        if m.shape != (height, width):
            m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
        masks.append(m)
    return masks


def union_person_mask(anns: list[dict[str, Any]], coco, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for m in instance_masks(anns, coco, height, width):
        mask = np.maximum(mask, m)
    return mask


def silhouette_boundary(mask: np.ndarray, thickness: int = 1) -> np.ndarray:
    """Binary GT silhouette boundary via morphological gradient."""
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return m
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    boundary = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, k)
    if thickness > 1:
        kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
        boundary = cv2.dilate(boundary, kd)
    return (boundary > 0).astype(np.uint8)


def person_eval_band(mask: np.ndarray, band_px: int = 21) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_px, band_px))
    return cv2.dilate(m, k)
