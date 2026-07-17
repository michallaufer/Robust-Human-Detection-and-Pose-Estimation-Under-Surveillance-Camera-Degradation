"""Unit tests — no full COCO required."""
from __future__ import annotations

import warnings

import cv2
import numpy as np
import pytest

from src.common.device import resolve_device
from src.distortions.jpeg import apply_jpeg
from src.distortions.low_light import apply_low_light
from src.distortions.motion_blur import apply_motion_blur
from src.distortions.registry import apply_distortion, compute_mse_psnr
from src.enhancement.deblur import enhance_motion_blur
from src.enhancement.jpeg import enhance_jpeg
from src.enhancement.low_light import enhance_low_light
from src.eval.boundary_metrics import boundary_metrics, canny_edges
from src.eval.result_schema import base_row, resume_key, validate_row
from src.dataset.yolo_pose_export import annotation_to_yolo_line
from src.train.mixed_corruption_dataset import assign_condition
from src.dataset.selective_download import image_url_templates, selective_image_url


def _synth() -> np.ndarray:
    img = np.full((120, 160, 3), 180, dtype=np.uint8)
    cv2.rectangle(img, (40, 20), (100, 100), (40, 40, 200), -1)
    return img


def test_resolve_device_cpu():
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_auto():
    d = resolve_device("auto")
    assert d in {"cpu", "0"}


def test_resolve_device_zero_fallback():
    import torch

    if torch.cuda.is_available():
        assert resolve_device("0") == "0"
    else:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert resolve_device("0") == "cpu"
            assert any("Falling back to CPU" in str(x.message) for x in w)


def test_resolve_device_invalid():
    with pytest.raises(ValueError):
        resolve_device("cuda:1")


def test_low_light_range_shape():
    img = _synth()
    out, meta = apply_low_light(img, gamma=2.2)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert 0 <= out.min() <= out.max() <= 255
    assert meta["gamma"] == 2.2
    assert out.mean() < img.mean()


def test_motion_blur_deterministic():
    img = _synth()
    a, _ = apply_motion_blur(img, 15, 45)
    b, _ = apply_motion_blur(img, 15, 45)
    assert np.array_equal(a, b)


def test_jpeg_severity_ordering():
    img = _synth()
    mild, _ = apply_jpeg(img, 50)
    severe, _ = apply_jpeg(img, 5)
    _, p_mild = compute_mse_psnr(img, mild)
    _, p_sev = compute_mse_psnr(img, severe)
    assert p_mild >= p_sev


def test_enhancement_ranges():
    img = _synth()
    dark, meta = apply_low_light(img, 2.2)
    out, em = enhance_low_light(dark)
    assert out.shape == img.shape and out.dtype == np.uint8
    blur, bm = apply_motion_blur(img, 15, 30)
    out2, em2 = enhance_motion_blur(blur, 15, 30, method="unsharp")
    assert out2.shape == img.shape
    assert em2["method_used"] == "stable_unsharp_mask"
    jpg, _ = apply_jpeg(img, 20)
    out3, _ = enhance_jpeg(jpg)
    assert out3.shape == img.shape


def test_boundary_metrics_synthetic():
    img = _synth()
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (40, 20), (100, 100), 1, -1)
    from src.dataset.coco_masks import silhouette_boundary, person_eval_band

    gt = silhouette_boundary(mask)
    pred = canny_edges(img)
    m = boundary_metrics(pred, gt, eval_band=person_eval_band(mask), tolerance_px=3)
    assert 0 <= m["boundary_f1"] <= 1
    assert 0 <= m["boundary_precision"] <= 1
    assert 0 <= m["boundary_recall"] <= 1


def test_result_schema():
    row = base_row(
        run_id="x",
        dataset_name="coco_person",
        dataset_split="val",
        subset_size=10,
        subset_seed=42,
        image_id=1,
        file_name="a.jpg",
        task="boundary",
        condition="clean",
        model_name="canny",
        weights_path="",
        device="cpu",
        image_width=10,
        image_height=10,
        boundary_precision=0.5,
        boundary_recall=0.5,
        boundary_f1=0.5,
        mean_edge_distance=1.0,
    )
    errs = validate_row(row)
    assert errs == []
    assert "1|boundary|clean" in resume_key(row)


def test_yolo_pose_label_norm():
    ann = {
        "bbox": [10, 20, 30, 40],
        "keypoints": [0] * 51,
    }
    # set one visible kp
    ann["keypoints"][0:3] = [15, 25, 2]
    line = annotation_to_yolo_line(ann, 100, 100)
    assert line is not None
    parts = line.split()
    assert parts[0] == "0"
    vals = list(map(float, parts[1:5]))
    assert all(0 <= v <= 1 for v in vals)


def test_mixed_corruption_proportions_deterministic():
    # over many ids, roughly balanced
    from collections import Counter

    fam = Counter()
    for i in range(400):
        cond, dist, _ = assign_condition(i, seed=42)
        fam[dist or "clean"] += 1
    # each family within 15%-35%
    for k in ["clean", "low_light", "motion_blur", "jpeg"]:
        assert 0.15 <= fam[k] / 400 <= 0.35


def test_distortion_registry_seed():
    img = _synth()
    a, _ = apply_distortion(img, "jpeg", "medium", seed=1, image_id=7)
    b, _ = apply_distortion(img, "jpeg", "medium", seed=1, image_id=7)
    assert np.array_equal(a, b)


def test_resume_key_unique():
    r1 = base_row(image_id=1, task="pose", condition="clean", model_name="a", weights_path="x", enhanced=False)
    r2 = base_row(image_id=1, task="pose", condition="clean", model_name="a", weights_path="y", enhanced=False)
    assert resume_key(r1) != resume_key(r2)


def test_selective_image_urls_use_http():
    templates = image_url_templates()
    assert templates["train"].startswith("http://")
    assert templates["val"].startswith("http://")
    assert "https://" not in templates["train"]
    assert "https://" not in templates["val"]
    train_url = selective_image_url("train", "000000000001.jpg")
    val_url = selective_image_url("val", "000000000002.jpg")
    assert train_url == "http://images.cocodataset.org/train2017/000000000001.jpg"
    assert val_url == "http://images.cocodataset.org/val2017/000000000002.jpg"
