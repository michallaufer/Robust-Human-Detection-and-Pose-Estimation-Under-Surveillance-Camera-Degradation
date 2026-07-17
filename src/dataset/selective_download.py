"""Selective COCO image download (never full train2017.zip)."""
from __future__ import annotations

import logging
import ssl
import time
import zipfile
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.common.io import ensure_dir, load_yaml, project_path

logger = logging.getLogger(__name__)


def annotation_urls() -> dict[str, str]:
    return {
        "annotations_zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    }


def image_url_templates() -> dict[str, str]:
    """HTTP templates for selective per-image download (matches annotation zip scheme)."""
    return {
        "train": "http://images.cocodataset.org/train2017/{file_name}",
        "val": "http://images.cocodataset.org/val2017/{file_name}",
    }


def selective_image_url(split: str, file_name: str) -> str:
    """Build the selective download URL for one COCO image."""
    return image_url_templates()[split].format(file_name=file_name)


def ensure_annotations(coco_root: Path | None = None, force: bool = False) -> Path:
    """Download and extract annotation JSON files only (≈241 MB zip)."""
    cfg = load_yaml(project_path("configs", "dataset.yaml"))
    coco_root = coco_root or project_path(cfg["coco_root"])
    ensure_dir(coco_root)
    ann_dir = coco_root / "annotations"
    needed = [
        ann_dir / "instances_train2017.json",
        ann_dir / "instances_val2017.json",
        ann_dir / "person_keypoints_train2017.json",
        ann_dir / "person_keypoints_val2017.json",
    ]
    # Detect demo overwrite: real keypoints JSON is ~hundreds of MB
    too_small = any(p.exists() and p.stat().st_size < 1_000_000 for p in needed if p.name.startswith("person_"))
    if all(p.exists() for p in needed) and not force and not too_small:
        logger.info("COCO annotations already present under %s", ann_dir)
        return coco_root

    zip_path = coco_root / "annotations_trainval2017.zip"
    if not zip_path.exists() or force:
        url = annotation_urls()["annotations_zip"]
        logger.info("Downloading annotations zip: %s", url)
        _download_file(
            url,
            zip_path,
            retries=int(cfg["download"]["retries"]),
            timeout=int(cfg["download"]["timeout_sec"]),
        )
    logger.info("Extracting annotations to %s", coco_root)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(coco_root)
    missing = [p for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Annotations missing after extract: {missing}")
    # Re-check size
    for p in needed:
        if p.name.startswith("person_") and p.stat().st_size < 1_000_000:
            raise RuntimeError(f"Annotation looks too small (demo overwrite?): {p}")
    return coco_root


def _ssl_context() -> ssl.SSLContext:
    # Prefer certifi if available; otherwise default context.
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _download_file(url: str, dest: Path, retries: int = 3, timeout: int = 60) -> None:
    ensure_dir(dest.parent)
    last_err: Exception | None = None
    ctx = _ssl_context()
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": "RobustHumanPerception/1.0"})
            with urlopen(req, timeout=timeout, context=ctx) as resp, open(dest, "wb") as out:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    out.write(chunk)
            if dest.stat().st_size <= 0:
                raise IOError(f"Downloaded empty file: {dest}")
            return
        except (HTTPError, URLError, TimeoutError, OSError, ssl.SSLError) as exc:
            last_err = exc
            logger.warning("Download attempt %s/%s failed for %s: %s", attempt, retries, url, exc)
            if dest.exists():
                dest.unlink(missing_ok=True)
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Failed to download {url}") from last_err


def _try_extract_from_local_zip(file_name: str, split: str, dest: Path, coco_root: Path) -> bool:
    """If a full split zip exists locally, extract only the needed member."""
    zip_path = coco_root / f"{split}2017.zip"
    if not zip_path.exists():
        return False
    member = f"{split}2017/{file_name}"
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # COCO zips usually nest under val2017/file.jpg
            names = set(zf.namelist())
            if member not in names:
                # sometimes plain file_name
                candidates = [n for n in names if n.endswith("/" + file_name) or n == file_name]
                if not candidates:
                    return False
                member = candidates[0]
            ensure_dir(dest.parent)
            with zf.open(member) as src, open(dest, "wb") as out:
                out.write(src.read())
            return dest.exists() and dest.stat().st_size > 0
    except Exception as exc:
        logger.warning("Local zip extract failed for %s: %s", file_name, exc)
        return False


def download_selected_images(
    file_names: list[str],
    split: str,
    dest_dir: Path,
    force: bool = False,
    verify: bool = True,
) -> dict[str, Any]:
    """
    Download only listed COCO images for train or val split into dest_dir.

    Prefer extracting from a local `{split}2017.zip` if present (disk-saving
    selective extract). Otherwise HTTP GET individual images.
    Never downloads the full zip archive automatically.
    """
    cfg = load_yaml(project_path("configs", "dataset.yaml"))
    coco_root = project_path(cfg["coco_root"])
    template = image_url_templates()[split]
    retries = int(cfg["download"]["retries"])
    timeout = int(cfg["download"]["timeout_sec"])
    ensure_dir(dest_dir)

    downloaded = 0
    skipped = 0
    from_zip = 0
    failed: list[str] = []

    for name in file_names:
        dest = dest_dir / name
        if dest.exists() and dest.stat().st_size > 0 and not force:
            skipped += 1
            continue
        # Prefer local zip member extraction
        if _try_extract_from_local_zip(name, split, dest, coco_root):
            from_zip += 1
            downloaded += 1
            continue
        url = template.format(file_name=name)
        try:
            _download_file(url, dest, retries=retries, timeout=timeout)
            if verify and dest.stat().st_size <= 0:
                raise IOError("empty file after download")
            downloaded += 1
        except Exception as exc:
            logger.error("Failed %s: %s", url, exc)
            failed.append(name)
            if dest.exists():
                dest.unlink(missing_ok=True)

    return {
        "split": split,
        "requested": len(file_names),
        "downloaded": downloaded,
        "from_local_zip": from_zip,
        "skipped": skipped,
        "failed": failed,
    }
