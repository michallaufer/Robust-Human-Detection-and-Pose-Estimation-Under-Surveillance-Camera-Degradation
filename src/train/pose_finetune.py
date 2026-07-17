"""Fine-tune YOLOv8n-pose on mixed-corruption data."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from src.common.device import device_info, resolve_device
from src.common.io import ensure_dir, load_yaml, project_path, save_json

logger = logging.getLogger(__name__)


def fine_tune_pose(
    data: str | Path | None = None,
    epochs: int | None = None,
    imgsz: int | None = None,
    batch: int | None = None,
    device: str = "auto",
    patience: int | None = None,
    workers: int | None = None,
    resume_training: bool = False,
    project: str | None = None,
    name: str | None = None,
) -> Path:
    from ultralytics import YOLO

    cfg = load_yaml(project_path("configs", "train_pose.yaml"))
    resolved = resolve_device(device)
    if resolved == "cpu":
        logger.warning(
            "Training on CPU may be very slow. Prefer Colab GPU (see colab/train_pose_colab.ipynb)."
        )

    data = data or project_path(cfg["data"])
    epochs = epochs if epochs is not None else int(cfg["epochs"])
    imgsz = imgsz if imgsz is not None else int(cfg["imgsz"])
    batch = batch if batch is not None else int(cfg["batch"])
    patience = patience if patience is not None else int(cfg["patience"])
    workers = workers if workers is not None else int(cfg["workers"])
    project = project or str(project_path(cfg["project"]))
    name = name or str(cfg["name"])

    env = device_info()
    logs = ensure_dir(project_path("results", "coco_person", "logs"))
    save_json(logs / "train_env.json", env)

    model = YOLO(str(cfg.get("model", "yolov8n-pose.pt")))
    results = model.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=resolved,
        project=project,
        name=name,
        exist_ok=True,
        patience=patience,
        workers=workers,
        lr0=float(cfg.get("lr0", 0.001)),
        optimizer=str(cfg.get("optimizer", "AdamW")),
        seed=int(cfg.get("seed", 42)),
        resume=resume_training,
        pretrained=True,
    )
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path(project) / name
    best = save_dir / "weights" / "best.pt"
    # Stable copy path for scripts
    stable = project_path("results", "coco_person", "checkpoints", "pose_robust_ft_best.pt")
    ensure_dir(stable.parent)
    if best.exists():
        import shutil

        shutil.copy2(best, stable)
    meta = {
        "best": str(best),
        "stable": str(stable),
        "device": resolved,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "data": str(data),
        "command": {
            "epochs": epochs,
            "imgsz": imgsz,
            "device": resolved,
            "patience": patience,
        },
        "env": env,
    }
    save_json(logs / "train_pose_meta.json", meta)
    logger.info("Fine-tune complete. best=%s stable=%s", best, stable)
    return stable if stable.exists() else best
