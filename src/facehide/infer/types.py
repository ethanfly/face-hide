from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

DeviceChoice = str

INFERENCE_DEVICES = ("auto", "gpu", "cpu")


def normalize_device(value: str | None) -> str:
    raw = str(value or "auto").strip().lower()
    return raw if raw in INFERENCE_DEVICES else "auto"


class InferenceFn(Protocol):
    def run(
        self,
        output_names: list[str] | None,
        input_feed: dict[str, np.ndarray],
    ) -> list[np.ndarray]: ...


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    score: float
    raw: np.ndarray


class Detector(Protocol):
    uses_fixed_input: bool

    def detect(self, bgr: np.ndarray, score_threshold: float) -> list[Detection]: ...

    def info(self) -> BackendInfo: ...


class Recognizer(Protocol):
    def embed(self, bgr: np.ndarray, raw_row: np.ndarray) -> np.ndarray: ...

    def info(self) -> BackendInfo: ...


@dataclass(frozen=True)
class BackendInfo:
    detector: str
    recognizer: str
    provider: str
    device_name: str
    dxgi_index: int | None
    dedicated_bytes: int | None
    fallback: bool


class DmlDeadFlag:
    def __init__(self) -> None:
        self.dead = False
        self.error = ""

    def trip(self, error: str) -> None:
        if not self.dead:
            self.dead = True
            self.error = str(error)

    def clear(self) -> None:
        self.dead = False
        self.error = ""
