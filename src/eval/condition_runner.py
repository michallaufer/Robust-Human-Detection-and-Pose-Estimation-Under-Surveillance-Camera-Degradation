"""Resumable multi-condition evaluation runner."""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from tqdm import tqdm

from src.common.constants import DATASET_COCO_PERSON, DISTORTION_NAMES, SEVERITIES
from src.common.device import resolve_device
from src.common.io import atomic_write_csv, ensure_dir, load_yaml, project_path, save_json
from src.dataset.coco_person import anns_to_boxes, anns_to_keypoints, iter_split
from src.dataset.splits import load_split_manifest
from src.distortions.registry import apply_distortion
from src.enhancement.registry import apply_enhancement
from src.eval.result_schema import base_row, resume_key
from src.tasks.boundary import load_yolo, run_boundary_task, run_detection, run_pose

logger = logging.getLogger(__name__)


def output_root_for_dataset(dataset_name: str) -> Path:
    if dataset_name == "demo":
        return project_path("results", "demo")
    return project_path("results", "coco_person")


def condition_label(distortion: str | None, severity: str | None, enhanced: bool) -> str:
    if distortion is None:
        return "clean"
    base = f"{distortion}_{severity}"
    return f"enhanced_{base}" if enhanced else base


def _load_done_keys(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return set()
    keys = set()
    for _, row in df.iterrows():
        keys.add(resume_key(row.to_dict()))
    return keys


def _append_row(csv_path: Path, row: dict[str, Any]) -> None:
    ensure_dir(csv_path.parent)
    df_new = pd.DataFrame([row])
    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    atomic_write_csv(csv_path, df)


def aggregate_csv(per_image_csv: Path, out_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(per_image_csv)
    if df.empty:
        atomic_write_csv(out_csv, df)
        return df
    # refuse mixed datasets
    if "dataset_name" in df.columns and df["dataset_name"].nunique() > 1:
        raise ValueError("Refusing to aggregate mixed dataset_name values")
    group_cols = [
        "task",
        "condition",
        "distortion",
        "severity",
        "enhanced",
        "model_name",
        "weights_path",
    ]
    group_cols = [c for c in group_cols if c in df.columns]
    numeric = [
        c
        for c in df.select_dtypes(include="number").columns.tolist()
        if c not in group_cols and c not in {"image_id", "subset_size", "subset_seed", "image_width", "image_height"}
    ]
    grouped = df.groupby(group_cols, dropna=False)
    agg = grouped[numeric].mean().reset_index()
    counts = grouped.size().rename("n_images").reset_index()
    agg = agg.merge(counts, on=group_cols, how="left")
    atomic_write_csv(out_csv, agg)
    return agg


def run_evaluation(
    split: str = "val",
    tasks: Iterable[str] = ("boundary", "detection", "pose"),
    conditions: Iterable[str] | None = None,
    max_images: int | None = None,
    device: str = "auto",
    imgsz: int = 416,
    seed: int = 42,
    resume: bool = True,
    output_dir: str | Path | None = None,
    det_weights: str = "yolov8n.pt",
    pose_weights: str = "yolov8n-pose.pt",
    include_enhanced: bool = True,
    pose_only_finetuned: bool = False,
) -> Path:
    """
    Evaluate selected tasks/conditions on the split.

    Writes per-image CSV under results/{dataset}/tables/.
    """
    manifest = load_split_manifest(split)
    dataset_name = manifest.get("dataset_name", DATASET_COCO_PERSON)
    out_root = Path(output_dir) if output_dir else output_root_for_dataset(dataset_name)
    tables = ensure_dir(out_root / "tables")
    run_id = uuid.uuid4().hex[:10]
    per_image_path = tables / f"{split}_per_image.csv"
    done = _load_done_keys(per_image_path) if resume else set()

    resolved = resolve_device(device)
    logger.info("Resolved device: %s (requested=%s)", resolved, device)

    dist_cfg = load_yaml(project_path("configs", "distortions.yaml"))
    boundary_cfg = dist_cfg.get("boundary", {})

    tasks = tuple(tasks)
    det_model = pose_model = None
    if "detection" in tasks and not pose_only_finetuned:
        det_model, resolved = load_yolo(det_weights, device=resolved)
    if "pose" in tasks:
        pose_model, resolved = load_yolo(pose_weights, device=resolved)

    # Build condition list
    wanted: list[tuple[str | None, str | None, bool]] = []
    if conditions:
        for c in conditions:
            c = c.strip()
            if c == "clean":
                wanted.append((None, None, False))
            elif c.startswith("enhanced_"):
                rest = c[len("enhanced_") :]
                d, s = rest.rsplit("_", 1)
                wanted.append((d, s, True))
            else:
                d, s = c.rsplit("_", 1)
                wanted.append((d, s, False))
    else:
        wanted.append((None, None, False))
        for d in DISTORTION_NAMES:
            for s in SEVERITIES:
                wanted.append((d, s, False))
                if include_enhanced:
                    wanted.append((d, s, True))

    subset_size = min(len(manifest["image_ids"]), max_images or len(manifest["image_ids"]))

    for image_id, image, info, anns, coco in tqdm(
        iter_split(split, max_images=max_images),
        total=subset_size,
        desc="images",
    ):
        gt_boxes = anns_to_boxes(anns)
        gt_kpts = anns_to_keypoints(anns)

        for distortion, severity, enhanced in wanted:
            # Build image for this condition
            dist_meta: dict[str, Any] = {}
            enh_meta: dict[str, Any] = {}
            mse = float("nan")
            psnr = float("nan")
            try:
                if distortion is None:
                    proc = image
                    condition = "clean"
                    enh_method = ""
                else:
                    proc, dist_meta = apply_distortion(
                        image, distortion, severity, seed=seed, image_id=image_id, cfg=dist_cfg
                    )
                    mse = float(dist_meta.get("mse", float("nan")))
                    psnr = float(dist_meta.get("psnr", float("nan")))
                    if enhanced:
                        proc, enh_meta = apply_enhancement(proc, distortion, dist_meta, cfg=dist_cfg)
                        # recompute quality vs clean after enhancement
                        from src.distortions.registry import compute_mse_psnr

                        mse, psnr = compute_mse_psnr(image, proc)
                    condition = condition_label(distortion, severity, enhanced)
                    enh_method = str(enh_meta.get("method_used", "")) if enhanced else ""

                for task in tasks:
                    if pose_only_finetuned and task != "pose":
                        continue
                    model_name = {
                        "boundary": "canny",
                        "detection": "yolov8n",
                        "pose": Path(pose_weights).stem if task == "pose" else "yolov8n-pose",
                    }[task]
                    weights_path = {
                        "boundary": "",
                        "detection": det_weights,
                        "pose": pose_weights,
                    }[task]
                    if task == "pose":
                        model_name = Path(pose_weights).stem

                    row = base_row(
                        run_id=run_id,
                        dataset_name=dataset_name,
                        dataset_split=split,
                        subset_size=subset_size,
                        subset_seed=seed,
                        image_id=image_id,
                        file_name=info.get("file_name", ""),
                        task=task,
                        condition=condition,
                        distortion=distortion or "",
                        severity=severity or "",
                        enhanced=bool(enhanced),
                        enhancement_method=enh_method,
                        model_name=model_name,
                        weights_path=weights_path,
                        device=resolved,
                        image_width=int(info.get("width", image.shape[1])),
                        image_height=int(info.get("height", image.shape[0])),
                        psnr=psnr,
                        mse=mse,
                    )
                    key = resume_key(row)
                    if key in done:
                        continue

                    # task-specific defaults
                    for f in (
                        "boundary_precision",
                        "boundary_recall",
                        "boundary_f1",
                        "mean_edge_distance",
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
                        "num_gt_people",
                        "num_matched_people",
                        "pck_02",
                        "pck_05",
                        "normalized_keypoint_error",
                        "oks",
                    ):
                        row[f] = None

                    try:
                        if task == "boundary":
                            metrics = run_boundary_task(proc, anns, coco, boundary_cfg)
                            row.update(metrics)
                        elif task == "detection":
                            assert det_model is not None
                            metrics = run_detection(
                                det_model, proc, gt_boxes, imgsz=imgsz, device=resolved
                            )
                            row.update(metrics)
                        elif task == "pose":
                            assert pose_model is not None
                            metrics = run_pose(
                                pose_model, proc, gt_boxes, gt_kpts, imgsz=imgsz, device=resolved
                            )
                            row.update(metrics)
                        row["success"] = True
                    except Exception as exc:
                        logger.exception("Fail image=%s task=%s condition=%s", image_id, task, condition)
                        row["success"] = False
                        row["error_message"] = str(exc)

                    _append_row(per_image_path, row)
                    done.add(key)
            except Exception as exc:
                logger.exception("Condition failure image=%s condition=%s", image_id, (distortion, severity, enhanced))
                # record a failed row for each task
                for task in tasks:
                    row = base_row(
                        run_id=run_id,
                        dataset_name=dataset_name,
                        dataset_split=split,
                        subset_size=subset_size,
                        subset_seed=seed,
                        image_id=image_id,
                        file_name=info.get("file_name", ""),
                        task=task,
                        condition=condition_label(distortion, severity, enhanced),
                        distortion=distortion or "",
                        severity=severity or "",
                        enhanced=bool(enhanced),
                        model_name=task,
                        weights_path="",
                        device=resolved,
                        image_width=int(info.get("width", 0)),
                        image_height=int(info.get("height", 0)),
                        success=False,
                        error_message=str(exc),
                    )
                    if resume_key(row) in done:
                        continue
                    _append_row(per_image_path, row)
                    done.add(resume_key(row))

    agg_path = tables / f"{split}_aggregate.csv"
    if per_image_path.exists():
        aggregate_csv(per_image_path, agg_path)
    save_json(
        tables / f"{split}_last_run.json",
        {
            "run_id": run_id,
            "device": resolved,
            "imgsz": imgsz,
            "max_images": max_images,
            "tasks": list(tasks),
            "pose_weights": pose_weights,
            "det_weights": det_weights,
            "per_image": str(per_image_path),
            "aggregate": str(agg_path),
        },
    )
    logger.info("Wrote %s", per_image_path)
    return per_image_path
