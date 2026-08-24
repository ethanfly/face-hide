from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def open_camera(index: int, width: int = 0, height: int = 0) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(index), cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(int(index))
    if cap.isOpened():
        if width and height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def read_bgr(cap: cv2.VideoCapture) -> np.ndarray | None:
    ok, frame = cap.read()
    if not ok or frame is None or frame.size == 0:
        return None
    return frame


def is_placeholder_frame(frame: np.ndarray) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    height, width = gray.shape[:2]
    if height < 16 or width < 16:
        return True
    ch, cw = max(1, height // 6), max(1, width // 6)
    corners = (
        gray[:ch, :cw],
        gray[:ch, -cw:],
        gray[-ch:, :cw],
        gray[-ch:, -cw:],
    )
    corner_mean = float(np.mean([tile.mean() for tile in corners]))
    corner_std = float(np.mean([tile.std() for tile in corners]))
    return corner_mean < 40 and corner_std < 8


@dataclass(frozen=True)
class CameraInfo:
    index: int
    width: int
    height: int
    placeholder: bool


def describe_cameras(limit: int = 4) -> list[CameraInfo]:
    found: list[CameraInfo] = []
    for index in range(limit):
        cap = open_camera(index)
        try:
            frame = read_bgr(cap) if cap.isOpened() else None
        finally:
            cap.release()
        if frame is None:
            continue
        height, width = frame.shape[:2]
        found.append(
            CameraInfo(
                index=index,
                width=width,
                height=height,
                placeholder=is_placeholder_frame(frame),
            )
        )
    return found


def pick_camera(infos: list[CameraInfo], preferred: int) -> int:
    if not infos:
        return preferred
    by_index = {info.index: info for info in infos}
    current = by_index.get(preferred)
    if current is not None and not current.placeholder:
        return preferred
    for info in infos:
        if not info.placeholder:
            return info.index
    return infos[0].index


def probe_cameras(limit: int = 4) -> list[int]:
    return [info.index for info in describe_cameras(limit)]
