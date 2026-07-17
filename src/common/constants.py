"""Shared constants for Robust Human Perception project."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DISTORTION_NAMES = ("low_light", "motion_blur", "jpeg")
SEVERITIES = ("mild", "medium", "severe")
TASKS = ("boundary", "detection", "pose")

COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

JOINT_GROUPS = {
    "head": (0, 1, 2, 3, 4),
    "torso": (5, 6, 11, 12),
    "arms": (7, 8, 9, 10),
    "legs": (13, 14, 15, 16),
}

# COCO OKS sigmas
COCO_SIGMAS = (
    0.026,
    0.025,
    0.025,
    0.035,
    0.035,
    0.079,
    0.079,
    0.072,
    0.072,
    0.062,
    0.062,
    0.107,
    0.107,
    0.087,
    0.087,
    0.089,
    0.089,
)

PERSON_CATEGORY_ID = 1
DATASET_COCO_PERSON = "coco_person"
DATASET_DEMO = "demo"

# YOLO-Pose flip indices for COCO-17
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
