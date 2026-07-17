"""Strict result row schema and helpers."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

COMMON_FIELDS = (
    "run_id",
    "timestamp",
    "dataset_name",
    "dataset_split",
    "subset_size",
    "subset_seed",
    "image_id",
    "file_name",
    "task",
    "condition",
    "distortion",
    "severity",
    "enhanced",
    "enhancement_method",
    "model_name",
    "weights_path",
    "device",
    "image_width",
    "image_height",
    "psnr",
    "mse",
    "success",
    "error_message",
)

BOUNDARY_FIELDS = (
    "boundary_precision",
    "boundary_recall",
    "boundary_f1",
    "mean_edge_distance",
)

DETECTION_FIELDS = (
    "num_gt",
    "num_predictions",
    "true_positives",
    "false_positives",
    "false_negatives",
    "precision",
    "recall",
    "f1",
    "miss_rate",
    "mean_confidence",
)

POSE_FIELDS = (
    "num_gt_people",
    "num_matched_people",
    "pck_02",
    "pck_05",
    "normalized_keypoint_error",
    "oks",
)

TASK_FIELDS = {
    "boundary": BOUNDARY_FIELDS,
    "detection": DETECTION_FIELDS,
    "pose": POSE_FIELDS,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def base_row(**kwargs: Any) -> dict[str, Any]:
    row = {k: None for k in COMMON_FIELDS}
    row["timestamp"] = utc_now_iso()
    row["enhanced"] = False
    row["success"] = True
    row["error_message"] = ""
    row["distortion"] = ""
    row["severity"] = ""
    row["enhancement_method"] = ""
    row["psnr"] = float("nan")
    row["mse"] = float("nan")
    row.update(kwargs)
    return row


def resume_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("image_id")),
            str(row.get("task")),
            str(row.get("condition")),
            str(row.get("model_name")),
            str(row.get("weights_path")),
            str(row.get("enhanced")),
        ]
    )


def validate_row(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for f in COMMON_FIELDS:
        if f not in row:
            errors.append(f"missing field: {f}")
    task = row.get("task")
    if task not in TASK_FIELDS:
        errors.append(f"invalid task: {task}")
    else:
        for f in TASK_FIELDS[task]:
            if f not in row:
                errors.append(f"missing task field: {f}")
    if row.get("dataset_name") not in {"coco_person", "demo"}:
        errors.append(f"unexpected dataset_name: {row.get('dataset_name')}")
    return errors


def metric_ranges_ok(row: dict[str, Any]) -> list[str]:
    """Soft numeric range checks."""
    problems: list[str] = []
    for key in ("boundary_precision", "boundary_recall", "boundary_f1", "precision", "recall", "f1", "pck_02", "pck_05", "oks"):
        if key in row and row[key] is not None:
            try:
                v = float(row[key])
                if not (v != v):  # not NaN
                    if v < -1e-6 or v > 1.0 + 1e-6:
                        problems.append(f"{key} out of [0,1]: {v}")
            except (TypeError, ValueError):
                problems.append(f"{key} not numeric")
    return problems
