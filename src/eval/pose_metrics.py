"""Pose metrics: OKS / PCK / normalized error."""
from __future__ import annotations

from typing import Any

import numpy as np

from src.common.constants import COCO_SIGMAS, JOINT_GROUPS
from src.eval.detection_metrics import iou_matrix


def oks_between(gt_kpts: np.ndarray, pred_kpts: np.ndarray, gt_box: np.ndarray) -> float:
    area = float(max(0.0, gt_box[2] - gt_box[0]) * max(0.0, gt_box[3] - gt_box[1]))
    area = max(area, 1.0)
    sigmas = np.asarray(COCO_SIGMAS, dtype=np.float64)
    dx = pred_kpts[:, 0] - gt_kpts[:, 0]
    dy = pred_kpts[:, 1] - gt_kpts[:, 1]
    visible = gt_kpts[:, 2] > 0
    if not np.any(visible):
        return 0.0
    e = (dx**2 + dy**2) / (2 * (sigmas**2) * (area + np.spacing(1)))
    return float(np.exp(-e)[visible].mean())


def match_poses(gt_boxes: np.ndarray, pred_boxes: np.ndarray, iou_thr: float = 0.3) -> list[tuple[int, int]]:
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return []
    ious = iou_matrix(pred_boxes, gt_boxes)
    pairs = []
    used_p: set[int] = set()
    used_g: set[int] = set()
    flat = [(float(ious[i, j]), i, j) for i in range(ious.shape[0]) for j in range(ious.shape[1])]
    flat.sort(reverse=True)
    for iou, i, j in flat:
        if iou < iou_thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        pairs.append((j, i))
    return pairs


def _box_scale(box: np.ndarray) -> float:
    """Normalization: max(box_w, box_h) for PCK; documented in README."""
    return max(float(box[2] - box[0]), float(box[3] - box[1]), 1.0)


def _box_diag(box: np.ndarray) -> float:
    w = max(float(box[2] - box[0]), 1.0)
    h = max(float(box[3] - box[1]), 1.0)
    return float(np.hypot(w, h))


def pose_pair_metrics(
    gt_kpts: np.ndarray,
    pred_kpts: np.ndarray,
    gt_box: np.ndarray,
) -> dict[str, Any]:
    visible = gt_kpts[:, 2] > 0
    scale = _box_scale(gt_box)
    diag = _box_diag(gt_box)
    dists = np.linalg.norm(pred_kpts[:, :2] - gt_kpts[:, :2], axis=1)
    norm_pck = dists / scale
    norm_err = dists / diag
    hits02 = (norm_pck <= 0.2) & visible
    hits05 = (norm_pck <= 0.5) & visible
    n_vis = int(visible.sum())
    out = {
        "pck_02": float(hits02[visible].mean()) if n_vis else 0.0,
        "pck_05": float(hits05[visible].mean()) if n_vis else 0.0,
        "normalized_keypoint_error": float(norm_err[visible].mean()) if n_vis else float("nan"),
        "oks": oks_between(gt_kpts, pred_kpts, gt_box),
        "per_joint_norm_err": np.where(visible, norm_err, np.nan),
    }
    for gname, idxs in JOINT_GROUPS.items():
        vals = out["per_joint_norm_err"][list(idxs)]
        out[f"err_group_{gname}"] = float(np.nanmean(vals)) if np.any(~np.isnan(vals)) else float("nan")
    return out


def evaluate_pose_image(
    gt_boxes: np.ndarray,
    gt_kpts_list: list[np.ndarray],
    pred_boxes: np.ndarray,
    pred_kpts_list: list[np.ndarray],
) -> dict[str, float]:
    pairs = match_poses(gt_boxes, pred_boxes)
    if not gt_boxes.shape[0]:
        return {
            "num_gt_people": 0,
            "num_matched_people": 0,
            "pck_02": 0.0,
            "pck_05": 0.0,
            "normalized_keypoint_error": float("nan"),
            "oks": 0.0,
        }
    metrics = []
    for gi, pi in pairs:
        metrics.append(pose_pair_metrics(gt_kpts_list[gi], pred_kpts_list[pi], gt_boxes[gi]))
    # unmatched GTs contribute oks=0
    oks_vals = [m["oks"] for m in metrics]
    oks_vals.extend([0.0] * (len(gt_boxes) - len(pairs)))
    return {
        "num_gt_people": int(len(gt_boxes)),
        "num_matched_people": int(len(pairs)),
        "pck_02": float(np.mean([m["pck_02"] for m in metrics])) if metrics else 0.0,
        "pck_05": float(np.mean([m["pck_05"] for m in metrics])) if metrics else 0.0,
        "normalized_keypoint_error": float(np.nanmean([m["normalized_keypoint_error"] for m in metrics]))
        if metrics
        else float("nan"),
        "oks": float(np.mean(oks_vals)) if oks_vals else 0.0,
    }
