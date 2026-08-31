from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from facehide.gallery import MatchResult, Person, best_match
from facehide.models import ensure_models, sface_path, yunet_path

# YuNet is created at 320×320; live ticks resize to this longest side.
DETECT_MAX_SIDE = 320


class NoFaceError(RuntimeError):
    pass


def detect_working_size(
    width: int,
    height: int,
    max_side: int = DETECT_MAX_SIDE,
) -> tuple[int, int]:
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        return max(1, width), max(1, height)
    if max_side <= 0:
        return width, height
    longest = max(width, height)
    if longest <= max_side:
        return width, height
    scale = max_side / float(longest)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def working_view(bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, float, float]:
    height, width = bgr.shape[:2]
    if max_side <= 0:
        return bgr, 1.0, 1.0
    work_w, work_h = detect_working_size(width, height, max_side)
    if work_w == width and work_h == height:
        return bgr, 1.0, 1.0
    work = cv2.resize(bgr, (work_w, work_h), interpolation=cv2.INTER_AREA)
    return work, width / float(work_w), height / float(work_h)


def map_box_to_source(
    x: int,
    y: int,
    w: int,
    h: int,
    src_size: tuple[int, int],
    work_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    src_w, src_h = src_size
    work_w, work_h = work_size
    if work_w <= 0 or work_h <= 0:
        return x, y, max(1, w), max(1, h)
    scale_x = src_w / float(work_w)
    scale_y = src_h / float(work_h)
    return (
        int(round(x * scale_x)),
        int(round(y * scale_y)),
        max(1, int(round(w * scale_x))),
        max(1, int(round(h * scale_y))),
    )


def scale_face_row(face: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    out = np.array(face, dtype=np.float32, copy=True)
    if scale_x == 1.0 and scale_y == 1.0:
        return out
    out[0] *= scale_x
    out[1] *= scale_y
    out[2] *= scale_x
    out[3] *= scale_y
    if out.size >= 14:
        out[4:14:2] *= scale_x
        out[5:14:2] *= scale_y
    return out


def should_extract_features(face_count: int) -> bool:
    return face_count > 0


@dataclass
class FaceHit:
    x: int
    y: int
    w: int
    h: int
    det_score: float
    raw: np.ndarray
    feature: np.ndarray | None
    match: MatchResult | None = None


def hits_from_detections(
    faces: np.ndarray | None,
    *,
    scale_x: float,
    scale_y: float,
    extract_features: bool,
    feature_fn: Callable[[np.ndarray], np.ndarray] | None,
) -> list[FaceHit]:
    if faces is None:
        return []
    rows = np.asarray(faces)
    if rows.size == 0:
        return []
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    count = len(rows)
    want_features = bool(extract_features) and should_extract_features(count) and feature_fn is not None
    hits: list[FaceHit] = []
    for face in rows:
        scaled = scale_face_row(face, scale_x, scale_y)
        x, y, fw, fh = [int(round(v)) for v in scaled[:4]]
        feature = feature_fn(scaled) if want_features else None
        hits.append(
            FaceHit(
                x=x,
                y=y,
                w=max(1, fw),
                h=max(1, fh),
                det_score=float(scaled[-1]) if scaled.size else 0.0,
                raw=scaled,
                feature=feature,
            )
        )
    return hits


def crop_face(bgr: np.ndarray, hit: FaceHit, pad: float = 0.25) -> np.ndarray:
    h, w = bgr.shape[:2]
    px = int(hit.w * pad)
    py = int(hit.h * pad)
    x1 = max(0, hit.x - px)
    y1 = max(0, hit.y - py)
    x2 = min(w, hit.x + hit.w + px)
    y2 = min(h, hit.y + hit.h + py)
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        raise NoFaceError("人脸裁剪失败")
    return crop


class FaceEngine:
    def __init__(
        self,
        det_model: Path | None = None,
        rec_model: Path | None = None,
        detect_score: float = 0.7,
    ) -> None:
        self._det_model = Path(det_model) if det_model else yunet_path()
        self._rec_model = Path(rec_model) if rec_model else sface_path()
        self.detect_score = detect_score
        self._lock = threading.Lock()
        self._det: cv2.FaceDetectorYN | None = None
        self._rec: cv2.FaceRecognizerSF | None = None
        self.extract_count = 0

    def ensure_ready(self) -> None:
        with self._lock:
            self._ensure_unlocked()

    def _ensure_unlocked(self) -> None:
        if self._det is not None and self._rec is not None:
            return
        if not self._det_model.exists() or not self._rec_model.exists():
            det, rec = ensure_models()
            self._det_model, self._rec_model = det, rec
        self._det = cv2.FaceDetectorYN.create(
            str(self._det_model),
            "",
            (320, 320),
            float(self.detect_score),
            0.3,
            5000,
        )
        self._rec = cv2.FaceRecognizerSF.create(str(self._rec_model), "")

    def detect(
        self,
        bgr: np.ndarray,
        extract_features: bool = True,
        *,
        max_side: int = 0,
    ) -> list[FaceHit]:
        if bgr is None or bgr.size == 0:
            return []
        with self._lock:
            self._ensure_unlocked()
            assert self._det is not None and self._rec is not None
            self._det.setScoreThreshold(float(self.detect_score))
            work, scale_x, scale_y = working_view(bgr, max_side)
            height, width = work.shape[:2]
            self._det.setInputSize((width, height))
            _retval, faces = self._det.detect(work)
            rec = self._rec
            src = bgr
            engine = self

            def feature_fn(raw: np.ndarray) -> np.ndarray:
                engine.extract_count += 1
                aligned = rec.alignCrop(src, raw)
                return np.asarray(rec.feature(aligned), dtype=np.float32).reshape(-1)

            return hits_from_detections(
                faces,
                scale_x=scale_x,
                scale_y=scale_y,
                extract_features=extract_features,
                feature_fn=feature_fn,
            )

    def annotate(
        self,
        bgr: np.ndarray,
        people: list[Person],
        threshold: float,
        *,
        max_side: int = 0,
        extract_features: bool = True,
    ) -> list[FaceHit]:
        hits = self.detect(bgr, extract_features=extract_features, max_side=max_side)
        if not people:
            return hits
        for hit in hits:
            if hit.feature is None:
                continue
            hit.match = best_match(hit.feature, people, threshold)
        return hits

    def enroll(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        faces = self.enroll_all(bgr)
        return faces[0]

    def enroll_all(self, bgr: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        hits = self.detect(bgr, extract_features=True)
        if not hits:
            raise NoFaceError("未检测到人脸，请换一张正脸清晰、光线充足的照片")
        out: list[tuple[np.ndarray, np.ndarray]] = []
        for hit in sorted(hits, key=lambda item: item.w * item.h, reverse=True):
            if hit.feature is None:
                continue
            out.append((hit.feature, crop_face(bgr, hit)))
        if not out:
            raise NoFaceError("特征提取失败")
        return out
