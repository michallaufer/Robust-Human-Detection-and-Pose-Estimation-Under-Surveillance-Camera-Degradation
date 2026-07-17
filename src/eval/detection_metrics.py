"""Person detection matching metrics."""
from __future__ import annotations

from typing import Any

import numpy as np


def iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)), dtype=np.float32)
    lt = np.maximum(pred[:, None, :2], gt[None, :, :2])
    rb = np.minimum(pred[:, None, 2:], gt[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_p = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    area_g = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    union = area_p[:, None] + area_g[None, :] - inter
    return inter / np.clip(union, 1e-6, None)


def match_detections(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thr: float = 0.5,
) -> dict[str, Any]:
    n_pred, n_gt = len(pred_boxes), len(gt_boxes)
    order = np.argsort(-pred_scores) if n_pred else np.array([], dtype=int)
    matched_gt: set[int] = set()
    tp = 0
    ious = iou_matrix(pred_boxes, gt_boxes) if n_pred and n_gt else None
    for i in order:
        if n_gt == 0:
            break
        j = int(np.argmax(ious[i]))
        if ious[i, j] >= iou_thr and j not in matched_gt:
            tp += 1
            matched_gt.add(j)
    fp = n_pred - tp
    fn = n_gt - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "num_gt": n_gt,
        "num_predictions": n_pred,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "miss_rate": float(fn / n_gt) if n_gt else 0.0,
        "mean_confidence": float(pred_scores.mean()) if n_pred else 0.0,
    }
