from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from facehide.infer.types import BackendInfo, Detection


class OpenCvYunetDetector:
    uses_fixed_input = False

    def __init__(self, model: Path | str, detect_score: float = 0.7) -> None:
        self._det = cv2.FaceDetectorYN.create(
            str(model),
            "",
            (320, 320),
            float(detect_score),
            0.3,
            5000,
        )

    def detect(self, bgr: np.ndarray, score_threshold: float) -> list[Detection]:
        self._det.setScoreThreshold(float(score_threshold))
        height, width = bgr.shape[:2]
        self._det.setInputSize((width, height))
        _retval, faces = self._det.detect(bgr)
        if faces is None:
            return []
        rows = np.asarray(faces, dtype=np.float32)
        if rows.size == 0:
            return []
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        hits: list[Detection] = []
        for row in rows:
            raw = np.asarray(row[:15], dtype=np.float32).reshape(15)
            hits.append(
                Detection(
                    box=(float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])),
                    score=float(raw[-1]),
                    raw=raw,
                )
            )
        return hits

    def info(self) -> BackendInfo:
        return BackendInfo("opencv", "opencv", "CPU", "CPU", None, None, True)


class OpenCvSfaceRecognizer:
    def __init__(self, model: Path | str) -> None:
        self._rec = cv2.FaceRecognizerSF.create(str(model), "")

    def align_crop(self, bgr: np.ndarray, raw_row: np.ndarray) -> np.ndarray:
        return self._rec.alignCrop(bgr, raw_row)

    def embed(self, bgr: np.ndarray, raw_row: np.ndarray) -> np.ndarray:
        aligned = self._rec.alignCrop(bgr, raw_row)
        return np.asarray(self._rec.feature(aligned), dtype=np.float32).reshape(-1)

    def info(self) -> BackendInfo:
        return BackendInfo("opencv", "opencv", "CPU", "CPU", None, None, True)
