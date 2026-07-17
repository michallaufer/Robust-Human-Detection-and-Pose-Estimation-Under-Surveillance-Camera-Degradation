"""Task wrappers for Canny / YOLO detect / YOLO pose."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.common.device import resolve_device
from src.common.io import project_path
from src.eval.boundary_metrics import evaluate_boundary
from src.eval.detection_metrics import match_detections
from src.eval.pose_metrics import evaluate_pose_image


def _resolve_weights(weights: str) -> str:
    p = Path(weights)
    if p.exists():
        return str(p)
    for c in (
        project_path(weights),
        project_path("results", "coco_person", "checkpoints", Path(weights).name),
        project_path("results", "checkpoints", Path(weights).name),
        project_path(Path(weights).name),
    ):
        if c.exists():
            return str(c)
    return weights


def load_yolo(weights: str, device: str = "auto"):
    from ultralytics import YOLO

    dev = resolve_device(device)
    model = YOLO(_resolve_weights(weights))
    return model, dev


def run_detection(
    model,
    image: np.ndarray,
    gt_boxes: np.ndarray,
    conf: float = 0.25,
    imgsz: int = 416,
    person_cls: int = 0,
    iou_thr: float = 0.5,
    device: str = "cpu",
) -> dict[str, Any]:
    results = model.predict(image, conf=conf, imgsz=imgsz, device=device, verbose=False)
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        pred_boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros((0,), dtype=np.float32)
    else:
        boxes = r0.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = r0.boxes.conf.cpu().numpy().astype(np.float32)
        clss = r0.boxes.cls.cpu().numpy().astype(int)
        keep = clss == person_cls
        pred_boxes, scores = boxes[keep], scores[keep]
    return match_detections(pred_boxes, scores, gt_boxes, iou_thr=iou_thr)


def run_pose(
    model,
    image: np.ndarray,
    gt_boxes: np.ndarray,
    gt_kpts: list[np.ndarray],
    conf: float = 0.25,
    imgsz: int = 416,
    device: str = "cpu",
) -> dict[str, Any]:
    results = model.predict(image, conf=conf, imgsz=imgsz, device=device, verbose=False)
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        pred_boxes = np.zeros((0, 4), dtype=np.float32)
        pred_kpts: list[np.ndarray] = []
    else:
        pred_boxes = r0.boxes.xyxy.cpu().numpy().astype(np.float32)
        pred_kpts = []
        if r0.keypoints is not None and r0.keypoints.data is not None:
            kps = r0.keypoints.data.cpu().numpy()
            for i in range(kps.shape[0]):
                arr = np.zeros((17, 3), dtype=np.float32)
                arr[:, :2] = kps[i, :, :2]
                arr[:, 2] = kps[i, :, 2] if kps.shape[-1] >= 3 else 1.0
                pred_kpts.append(arr)
        else:
            pred_kpts = [np.zeros((17, 3), dtype=np.float32) for _ in range(len(pred_boxes))]
    return evaluate_pose_image(gt_boxes, gt_kpts, pred_boxes, pred_kpts)


def run_boundary_task(image, anns, coco, boundary_cfg: dict | None = None) -> dict[str, float]:
    cfg = boundary_cfg or {}
    return evaluate_boundary(
        image,
        anns,
        coco,
        canny_low=int(cfg.get("canny_low", 50)),
        canny_high=int(cfg.get("canny_high", 150)),
        gaussian_ksize=int(cfg.get("gaussian_ksize", 3)),
        tolerance_px=int(cfg.get("tolerance_px", 3)),
        person_band_px=int(cfg.get("person_band_px", 21)),
    )
