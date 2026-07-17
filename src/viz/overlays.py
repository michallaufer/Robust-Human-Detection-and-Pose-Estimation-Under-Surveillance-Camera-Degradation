"""Drawing overlays for qualitative grids."""
from __future__ import annotations

import cv2
import numpy as np


def draw_boxes(img: np.ndarray, boxes: np.ndarray, color=(0, 255, 0), thickness: int = 2) -> np.ndarray:
    out = img.copy()
    for b in boxes:
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out


def draw_keypoints(img: np.ndarray, kpts_list: list[np.ndarray], color=(0, 255, 255)) -> np.ndarray:
    out = img.copy()
    edges = [
        (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    ]
    for kps in kpts_list:
        for a, b in edges:
            if kps[a, 2] > 0 and kps[b, 2] > 0:
                cv2.line(out, tuple(map(int, kps[a, :2])), tuple(map(int, kps[b, :2])), color, 2)
        for j in range(17):
            if kps[j, 2] > 0:
                cv2.circle(out, tuple(map(int, kps[j, :2])), 3, (0, 0, 255), -1)
    return out


def overlay_edges(img: np.ndarray, edges: np.ndarray, color=(0, 255, 255)) -> np.ndarray:
    out = img.copy()
    out[edges > 0] = color
    return out
