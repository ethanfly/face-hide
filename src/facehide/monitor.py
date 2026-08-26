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
from facehide.engine import FaceEngine, FaceHit, NoFaceError, crop_face
from facehide.gallery import Gallery, Person, best_match, can_trigger, cosine_similarity
from facehide.i18n import t


@dataclass(frozen=True)
class SeenFace:
    name: str
    score: float
    hide_enabled: bool
    blacklisted: bool = False
    nickname: str = ""


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
    seen: list[SeenFace] = field(default_factory=list)


def track_seen(
    active: dict[str, float],
    present: dict[str, SeenFace],
    now: float,
    *,
    grace: float = 1.5,
) -> tuple[dict[str, float], list[SeenFace]]:
    next_active = dict(active)
    newly: list[SeenFace] = []
    for name, face in present.items():
        if name not in next_active:
            newly.append(face)
        next_active[name] = now
    for name, stamp in list(next_active.items()):
        if name not in present and now - stamp > grace:
            del next_active[name]
    return next_active, newly


def enroll_unknown_faces(
    gallery: Gallery,
    frame: np.ndarray,
    hits: list[FaceHit],
    *,
    threshold: float,
    recent: list[tuple[float, np.ndarray]],
    now: float,
    grace: float = 30.0,
) -> list[Person]:
    kept = [(stamp, feature) for stamp, feature in recent if now - stamp <= grace]
    people = gallery.people()
    enrolled: list[Person] = []
    for hit in hits:
        if hit.feature is None:
            continue
        if best_match(hit.feature, people, threshold) is not None:
            continue
        feature = np.asarray(hit.feature, dtype=np.float32).reshape(-1)
        if any(cosine_similarity(feature, other) >= threshold for _stamp, other in kept):
            continue
        try:
            thumb = crop_face(frame, hit)
        except NoFaceError:
            continue
        name = t("faces.auto_unnamed", index=len(people) + len(enrolled) + 1)
        person = gallery.add_person(name, feature, thumb, enabled=False)
        kept.append((now, feature))
        enrolled.append(person)
    recent[:] = kept
    return enrolled


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
    faces_changed = Signal()

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
        recent_unknown: list[tuple[float, np.ndarray]] = []
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
                seen: list[SeenFace] = []
                for hit in recognized:
                    assert hit.match is not None
                    seen.append(
                        SeenFace(
                            name=hit.match.person.name,
                            score=hit.match.score,
                            hide_enabled=hit.match.person.enabled,
                            blacklisted=hit.match.person.blacklisted,
                            nickname=hit.match.person.nickname,
                        )
                    )
                triggerable = [hit for hit in recognized if can_trigger(hit.match, settings.match_threshold)]
                if settings.auto_enroll_unknown:
                    created = enroll_unknown_faces(
                        self._gallery,
                        frame,
                        hits,
                        threshold=settings.match_threshold,
                        recent=recent_unknown,
                        now=time.monotonic(),
                    )
                    if created:
                        self.status.emit(
                            t("log.auto_enrolled", count=len(created), name=", ".join(item.name for item in created))
                        )
                        self.faces_changed.emit()
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
                        seen=seen,
                    )
                )
        finally:
            if cap is not None:
                cap.release()
            self.status.emit(t("log.cam_closed"))
