from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from facehide.gallery import MatchResult, Person, best_match
from facehide.models import ensure_models, sface_path, yunet_path


class NoFaceError(RuntimeError):
    pass


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

    def detect(self, bgr: np.ndarray, extract_features: bool = True) -> list[FaceHit]:
        if bgr is None or bgr.size == 0:
            return []
        with self._lock:
            self._ensure_unlocked()
            assert self._det is not None and self._rec is not None
            self._det.setScoreThreshold(float(self.detect_score))
            height, width = bgr.shape[:2]
            self._det.setInputSize((width, height))
            _retval, faces = self._det.detect(bgr)
            if faces is None or len(faces) == 0:
                return []
            hits: list[FaceHit] = []
            for face in faces:
                x, y, fw, fh = [int(v) for v in face[:4]]
                feature = None
                if extract_features:
                    aligned = self._rec.alignCrop(bgr, face)
                    feature = np.asarray(self._rec.feature(aligned), dtype=np.float32).reshape(-1)
                hits.append(
                    FaceHit(
                        x=x,
                        y=y,
                        w=max(1, fw),
                        h=max(1, fh),
                        det_score=float(face[-1]),
                        raw=face,
                        feature=feature,
                    )
                )
            return hits

    def annotate(
        self,
        bgr: np.ndarray,
        people: list[Person],
        threshold: float,
    ) -> list[FaceHit]:
        hits = self.detect(bgr, extract_features=True)
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
