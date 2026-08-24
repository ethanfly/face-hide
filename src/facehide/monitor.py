from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from facehide.actions import perform_switch
from facehide.camera import open_camera, read_bgr
from facehide.config import SettingsStore
from facehide.engine import FaceEngine, FaceHit
from facehide.gallery import Gallery, can_trigger
from facehide.i18n import t


@dataclass
class PreviewFrame:
    rgb: np.ndarray
    hits: list[FaceHit]
    fps: float
    streak: int
    matched_name: str | None
    camera_ok: bool
    message: str = ""
    camera_index: int = 0
    threshold: float = 0.4
    best_score: float = -1.0
    dev_mode: bool = False
    matched_armed: bool = False


@dataclass
class TriggerEvent:
    person_name: str
    score: float
    actions: list[str] = field(default_factory=list)
    error: str = ""
    dry_run: bool = False


class MonitorThread(QThread):
    frame_ready = Signal(object)
    triggered = Signal(object)
    status = Signal(str)

    def __init__(
        self,
        engine: FaceEngine,
        gallery: Gallery,
        store: SettingsStore,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._engine = engine
        self._gallery = gallery
        self._store = store
        self._armed = False
        self._stop = False
        self._latest_bgr: np.ndarray | None = None
        self._protected_hwnds: set[int] = set()

    def set_armed(self, armed: bool) -> None:
        self._armed = bool(armed)

    def set_protected_hwnds(self, hwnds: set[int]) -> None:
        self._protected_hwnds = set(hwnds)

    def latest_bgr(self) -> np.ndarray | None:
        if self._latest_bgr is None:
            return None
        return self._latest_bgr.copy()

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        self._stop = False
        cap = None
        opened_index = None
        streak = 0
        cooldown_until = 0.0
        last = time.perf_counter()
        fps = 0.0
        try:
            while not self._stop:
                settings = self._store.get()
                need_reopen = (
                    cap is None
                    or not cap.isOpened()
                    or opened_index != settings.camera_index
                )
                if need_reopen:
                    if cap is not None:
                        cap.release()
                    cap = open_camera(settings.camera_index, settings.frame_width, settings.frame_height)
                    opened_index = settings.camera_index
                    if not cap.isOpened():
                        self.frame_ready.emit(
                            PreviewFrame(
                                rgb=np.zeros((360, 640, 3), dtype=np.uint8),
                                hits=[],
                                fps=0.0,
                                streak=0,
                                matched_name=None,
                                camera_ok=False,
                                message=t("preview.cam_fail"),
                                camera_index=settings.camera_index,
                                threshold=settings.match_threshold,
                                dev_mode=settings.dev_mode,
                            )
                        )
                        self.status.emit(t("log.cam_unavailable"))
                        cap.release()
                        cap = None
                        self.msleep(1000)
                        continue
                    self.status.emit(t("log.cam_opened"))
                    streak = 0

                frame = read_bgr(cap)
                if frame is None:
                    self.msleep(30)
                    continue
                self._latest_bgr = frame
                people = self._gallery.people()
                self._engine.detect_score = settings.detect_score
                preview_threshold = -1.0 if settings.dev_mode and people else settings.match_threshold
                hits = self._engine.annotate(frame, people, preview_threshold)
                recognized = [
                    hit
                    for hit in hits
                    if hit.match is not None and hit.match.score >= settings.match_threshold
                ]
                triggerable = [hit for hit in recognized if can_trigger(hit.match, settings.match_threshold)]
                best_score = max((hit.match.score for hit in hits if hit.match), default=-1.0)
                if recognized:
                    shown = max(recognized, key=lambda hit: hit.match.score if hit.match else -1)
                    name = shown.match.person.name if shown.match else None
                else:
                    name = None
                if triggerable:
                    streak += 1
                    top = max(triggerable, key=lambda hit: hit.match.score if hit.match else -1)
                else:
                    streak = 0
                    top = None

                if (
                    self._armed
                    and top is not None
                    and streak >= settings.confirm_frames
                    and time.monotonic() >= cooldown_until
                ):
                    assert top.match is not None
                    try:
                        actions = perform_switch(
                            settings,
                            protected_hwnds=self._protected_hwnds,
                            protected_pids={os.getpid()},
                            dry_run=settings.dev_mode,
                        )
                        self.triggered.emit(
                            TriggerEvent(
                                person_name=top.match.person.name,
                                score=top.match.score,
                                actions=actions,
                                dry_run=settings.dev_mode,
                            )
                        )
                    except Exception as exc:  # noqa: BLE001
                        self.triggered.emit(
                            TriggerEvent(
                                person_name=top.match.person.name,
                                score=top.match.score,
                                error=str(exc),
                            )
                        )
                    cooldown_until = time.monotonic() + settings.cooldown_seconds
                    streak = 0

                now = time.perf_counter()
                dt = now - last
                last = now
                if dt > 0:
                    inst = 1.0 / dt
                    fps = inst if fps == 0 else fps * 0.85 + inst * 0.15

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(
                    PreviewFrame(
                        rgb=rgb,
                        hits=hits,
                        fps=fps,
                        streak=streak,
                        matched_name=name,
                        camera_ok=True,
                        camera_index=settings.camera_index,
                        threshold=settings.match_threshold,
                        best_score=best_score,
                        dev_mode=settings.dev_mode,
                        matched_armed=bool(triggerable),
                    )
                )
        finally:
            if cap is not None:
                cap.release()
            self.status.emit(t("log.cam_closed"))
