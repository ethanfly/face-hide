from __future__ import annotations

import cv2
import numpy as np

from facehide.infer.preprocess import CANVAS

STRIDES = (8, 16, 32)
YUNET_OUTPUTS = (
    "cls_8",
    "cls_16",
    "cls_32",
    "obj_8",
    "obj_16",
    "obj_32",
    "bbox_8",
    "bbox_16",
    "bbox_32",
    "kps_8",
    "kps_16",
    "kps_32",
)
YUNET_INPUT = "input"


def nms_indices(
    boxes: list[list[float]],
    scores: list[float],
    score_threshold: float,
    nms_threshold: float,
    top_k: int,
) -> list[int]:
    if not boxes:
        return []
    raw = cv2.dnn.NMSBoxes(boxes, scores, float(score_threshold), float(nms_threshold), top_k=int(top_k))
    if raw is None or len(raw) == 0:
        return []
    return [int(i) for i in np.asarray(raw).reshape(-1)]


def _decode_stride(
    cls: np.ndarray,
    obj: np.ndarray,
    bbox: np.ndarray,
    kps: np.ndarray,
    stride: int,
    canvas: int,
    score_threshold: float,
) -> np.ndarray:
    cls_v = np.clip(np.asarray(cls, dtype=np.float32).reshape(-1), 0.0, 1.0)
    obj_v = np.clip(np.asarray(obj, dtype=np.float32).reshape(-1), 0.0, 1.0)
    score = np.sqrt(cls_v * obj_v)
    keep = score >= float(score_threshold)
    if not np.any(keep):
        return np.zeros((0, 15), dtype=np.float32)
    idx = np.nonzero(keep)[0]
    cols = canvas // stride
    col = (idx % cols).astype(np.float32)
    row = (idx // cols).astype(np.float32)
    bb = np.asarray(bbox, dtype=np.float32).reshape(-1, 4)[idx]
    kp = np.asarray(kps, dtype=np.float32).reshape(-1, 10)[idx]
    cx = (col + bb[:, 0]) * stride
    cy = (row + bb[:, 1]) * stride
    w = np.exp(bb[:, 2]) * stride
    h = np.exp(bb[:, 3]) * stride
    out = np.empty((idx.size, 15), dtype=np.float32)
    out[:, 0] = cx - w * 0.5
    out[:, 1] = cy - h * 0.5
    out[:, 2] = w
    out[:, 3] = h
    for n in range(5):
        out[:, 4 + 2 * n] = (kp[:, 2 * n] + col) * stride
        out[:, 5 + 2 * n] = (kp[:, 2 * n + 1] + row) * stride
    out[:, 14] = score[idx]
    return out


def outputs_to_dict(names: list[str], arrays: list[np.ndarray]) -> dict[str, np.ndarray]:
    return {name: arr for name, arr in zip(names, arrays, strict=False)}


def decode_heads(
    outputs: dict[str, np.ndarray],
    *,
    score_threshold: float = 0.7,
    nms_threshold: float = 0.3,
    top_k: int = 5000,
    canvas: int = CANVAS,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for stride in STRIDES:
        parts.append(
            _decode_stride(
                outputs[f"cls_{stride}"],
                outputs[f"obj_{stride}"],
                outputs[f"bbox_{stride}"],
                outputs[f"kps_{stride}"],
                stride,
                canvas,
                score_threshold,
            )
        )
    stacked = np.concatenate(parts, axis=0) if parts else np.zeros((0, 15), dtype=np.float32)
    if stacked.size == 0:
        return np.zeros((0, 15), dtype=np.float32)
    boxes = stacked[:, :4].tolist()
    scores = stacked[:, 14].tolist()
    keep = nms_indices(boxes, scores, score_threshold, nms_threshold, top_k)
    if not keep:
        return np.zeros((0, 15), dtype=np.float32)
    return stacked[np.asarray(keep, dtype=np.int64)]


def empty_yunet_outputs() -> list[np.ndarray]:
    return [
        np.zeros((1, 6400, 1), dtype=np.float32),
        np.zeros((1, 1600, 1), dtype=np.float32),
        np.zeros((1, 400, 1), dtype=np.float32),
        np.zeros((1, 6400, 1), dtype=np.float32),
        np.zeros((1, 1600, 1), dtype=np.float32),
        np.zeros((1, 400, 1), dtype=np.float32),
        np.zeros((1, 6400, 4), dtype=np.float32),
        np.zeros((1, 1600, 4), dtype=np.float32),
        np.zeros((1, 400, 4), dtype=np.float32),
        np.zeros((1, 6400, 10), dtype=np.float32),
        np.zeros((1, 1600, 10), dtype=np.float32),
        np.zeros((1, 400, 10), dtype=np.float32),
    ]
