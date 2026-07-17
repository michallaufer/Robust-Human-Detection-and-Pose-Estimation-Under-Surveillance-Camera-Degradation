"""Boundary metrics (person-region Canny vs silhouette)."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.dataset.coco_masks import person_eval_band, silhouette_boundary, union_person_mask


def canny_edges(
    image: np.ndarray,
    low: int = 50,
    high: int = 150,
    gaussian_ksize: int = 3,
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if gaussian_ksize and gaussian_ksize >= 3:
        k = int(gaussian_ksize) | 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    return (cv2.Canny(gray, int(low), int(high)) > 0).astype(np.uint8)


def boundary_metrics(
    pred_edges: np.ndarray,
    gt_boundary: np.ndarray,
    eval_band: np.ndarray | None = None,
    tolerance_px: int = 3,
) -> dict[str, float]:
    pred = pred_edges.astype(bool)
    gt = gt_boundary.astype(bool)
    if eval_band is not None:
        band = eval_band.astype(bool)
        pred = pred & band
        gt = gt & band

    if tolerance_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * tolerance_px + 1, 2 * tolerance_px + 1)
        )
        gt_dil = cv2.dilate(gt.astype(np.uint8), k).astype(bool)
        pred_dil = cv2.dilate(pred.astype(np.uint8), k).astype(bool)
    else:
        gt_dil, pred_dil = gt, pred

    pred_n = int(pred.sum())
    gt_n = int(gt.sum())
    tp = int(np.logical_and(pred, gt_dil).sum())
    matched_gt = int(np.logical_and(gt, pred_dil).sum())
    precision = float(tp / pred_n) if pred_n else 0.0
    recall = float(matched_gt / gt_n) if gt_n else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    mean_dist = float("nan")
    if gt_n > 0 and pred_n > 0:
        inv = np.where(pred, 0, 255).astype(np.uint8)
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
        ys, xs = np.where(gt)
        mean_dist = float(dist[ys, xs].mean())
    elif gt_n > 0:
        mean_dist = float(tolerance_px * 2)

    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
        "mean_edge_distance": mean_dist,
    }


def evaluate_boundary(
    image: np.ndarray,
    anns: list[dict],
    coco,
    canny_low: int = 50,
    canny_high: int = 150,
    gaussian_ksize: int = 3,
    tolerance_px: int = 3,
    person_band_px: int = 21,
) -> dict[str, float]:
    h, w = image.shape[:2]
    mask = union_person_mask(anns, coco, h, w)
    if mask.sum() == 0:
        return {
            "boundary_precision": 0.0,
            "boundary_recall": 0.0,
            "boundary_f1": 0.0,
            "mean_edge_distance": float("nan"),
        }
    gt = silhouette_boundary(mask, thickness=1)
    band = person_eval_band(mask, band_px=person_band_px)
    pred = canny_edges(image, low=canny_low, high=canny_high, gaussian_ksize=gaussian_ksize)
    return boundary_metrics(pred, gt, eval_band=band, tolerance_px=tolerance_px)
