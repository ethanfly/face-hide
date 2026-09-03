from __future__ import annotations

import cv2
import numpy as np

CANVAS = 640


def pad_scale(width: int, height: int, canvas: int = CANVAS) -> float:
    if width <= 0 or height <= 0:
        return 1.0
    return min(1.0, min(canvas / float(width), canvas / float(height)))


def fit_canvas(bgr: np.ndarray, canvas: int = CANVAS) -> tuple[np.ndarray, float]:
    height, width = bgr.shape[:2]
    scale = pad_scale(width, height, canvas)
    if scale < 1.0:
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        placed = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        placed = bgr
        new_w, new_h = width, height
        scale = 1.0
    canvas_bgr = np.zeros((canvas, canvas, 3), dtype=np.uint8)
    canvas_bgr[:new_h, :new_w] = placed
    return canvas_bgr, scale


def pad_bgr(bgr: np.ndarray, canvas: int = CANVAS) -> tuple[np.ndarray, float]:
    canvas_bgr, scale = fit_canvas(bgr, canvas)
    blob = np.transpose(canvas_bgr, (2, 0, 1))[None].astype(np.float32)
    return blob, scale


def unpad_rows(rows: np.ndarray, scale: float) -> np.ndarray:
    if rows.size == 0:
        return rows
    out = np.array(rows, dtype=np.float32, copy=True)
    if scale == 1.0 or scale <= 0:
        return out
    out[:, 0:14] /= float(scale)
    return out
