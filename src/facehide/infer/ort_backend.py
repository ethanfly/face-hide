from __future__ import annotations

import cv2
import numpy as np

from facehide.infer.preprocess import pad_bgr, unpad_rows
from facehide.infer.types import BackendInfo, Detection, DmlDeadFlag, InferenceFn
from facehide.infer.yunet import YUNET_INPUT, YUNET_OUTPUTS, decode_heads, outputs_to_dict


class OrtYunetDetector:
    uses_fixed_input = True

    def __init__(
        self,
        runner: InferenceFn,
        *,
        flag: DmlDeadFlag,
        info: BackendInfo,
        input_name: str = YUNET_INPUT,
        output_names: list[str] | None = None,
    ) -> None:
        self._fn = runner
        self._flag = flag
        self._info = info
        self._input = input_name or YUNET_INPUT
        names = getattr(runner, "output_names", None)
        self._outputs = list(output_names or names or YUNET_OUTPUTS)

    def detect(self, bgr: np.ndarray, score_threshold: float) -> list[Detection]:
        blob, scale = pad_bgr(bgr)
        try:
            arrays = self._fn.run(None, {self._input: blob})
        except Exception as exc:
            self._flag.trip(str(exc))
            return []
        mapped = outputs_to_dict(self._outputs, list(arrays))
        try:
            rows = decode_heads(mapped, score_threshold=float(score_threshold))
        except Exception as exc:
            self._flag.trip(str(exc))
            return []
        rows = unpad_rows(rows, scale)
        hits: list[Detection] = []
        for raw in rows:
            hits.append(
                Detection(
                    box=(float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])),
                    score=float(raw[14]),
                    raw=np.asarray(raw, dtype=np.float32).reshape(15),
                )
            )
        return hits

    def info(self) -> BackendInfo:
        return self._info


class OrtSfaceRecognizer:
    def __init__(
        self,
        runner: InferenceFn,
        *,
        aligner,
        flag: DmlDeadFlag,
        info: BackendInfo,
        input_name: str = "data",
    ) -> None:
        self._fn = runner
        self._aligner = aligner
        self._flag = flag
        self._info = info
        self._input = getattr(runner, "input_name", None) or input_name

    def embed(self, bgr: np.ndarray, raw_row: np.ndarray) -> np.ndarray:
        aligned = self._aligner.align_crop(bgr, raw_row)
        blob = cv2.dnn.blobFromImage(
            aligned,
            1.0,
            (112, 112),
            (0, 0, 0),
            swapRB=True,
            crop=False,
        )
        try:
            out = self._fn.run(None, {self._input: blob})[0]
        except Exception as exc:
            self._flag.trip(str(exc))
            return np.zeros(128, dtype=np.float32)
        return np.asarray(out, dtype=np.float32).reshape(-1)

    def info(self) -> BackendInfo:
        return self._info
