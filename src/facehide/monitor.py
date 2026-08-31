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
from facehide.engine import DETECT_MAX_SIDE, FaceEngine, FaceHit, NoFaceError, crop_face
from facehide.gallery import Gallery, Person, best_match, can_trigger, cosine_similarity
from facehide.i18n import t

VISIBLE_TICK_MS = 50
HIDDEN_TICK_MS = 200


def preview_needed(window_shown: bool, extra: int = 0) -> bool:
    return bool(window_shown) or extra > 0


def should_build_preview_rgb(need_preview: bool, in_flight: bool) -> bool:
    return bool(need_preview) and not in_flight


def tick_sleep_ms(need_preview: bool) -> int:
    return VISIBLE_TICK_MS if need_preview else HIDDEN_TICK_MS


def remaining_sleep_ms(interval_ms: int, elapsed_s: float) -> int:
    left = int(interval_ms - elapsed_s * 1000.0)
    return left if left > 0 else 0


def empty_preview_rgb() -> np.ndarray:
    return np.empty((0, 0, 3), dtype=np.uint8)


def preview_rgb(bgr: np.ndarray | None, need_preview: bool) -> np.ndarray:
    if not need_preview or bgr is None or bgr.size == 0:
        return empty_preview_rgb()
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@dataclass(frozen=True)
class TickPlan:
    need_preview: bool
    sleep_ms: int
    detect_max_side: int


def plan_tick(window_shown: bool, extra_preview: int = 0) -> TickPlan:
    need = preview_needed(window_shown, extra_preview)
    return TickPlan(
        need_preview=need,
        sleep_ms=tick_sleep_ms(need),
        detect_max_side=DETECT_MAX_SIDE,
    )


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
        self._preview_needed = False
        self._preview_extra = 0
        self._preview_in_flight = False

    def set_armed(self, armed: bool) -> None:
        self._armed = bool(armed)

    def set_preview_needed(self, needed: bool) -> None:
        self._preview_needed = bool(needed)

    def add_preview_extra(self) -> None:
        self._preview_extra += 1

    def remove_preview_extra(self) -> None:
        if self._preview_extra > 0:
            self._preview_extra -= 1

    def ack_preview(self) -> None:
        self._preview_in_flight = False

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
                loop = self._store.loop_settings()
                plan = plan_tick(self._preview_needed, self._preview_extra)
                need_reopen = (
                    cap is None
                    or not cap.isOpened()
                    or opened_index != loop.camera_index
                )
                if need_reopen:
                    if cap is not None:
                        cap.release()
                    cap = open_camera(loop.camera_index, loop.frame_width, loop.frame_height)
                    opened_index = loop.camera_index
                    if not cap.isOpened():
                        self.frame_ready.emit(
                            PreviewFrame(
                                rgb=preview_rgb(None, False),
                                hits=[],
                                fps=0.0,
                                streak=0,
                                matched_name=None,
                                camera_ok=False,
                                message=t("preview.cam_fail"),
                                camera_index=loop.camera_index,
                                threshold=loop.match_threshold,
                                dev_mode=loop.dev_mode,
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
                tick_started = time.perf_counter()
                self._latest_bgr = frame
                people = self._gallery.people()
                self._engine.detect_score = loop.detect_score
                preview_threshold = -1.0 if loop.dev_mode and people else loop.match_threshold
                extract_features = bool(people) or loop.auto_enroll_unknown
                hits = self._engine.annotate(
                    frame,
                    people,
                    preview_threshold,
                    max_side=plan.detect_max_side,
                    extract_features=extract_features,
                )
                recognized = [
                    hit
                    for hit in hits
                    if hit.match is not None and hit.match.score >= loop.match_threshold
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
                triggerable = [hit for hit in recognized if can_trigger(hit.match, loop.match_threshold)]
                if loop.auto_enroll_unknown:
                    created = enroll_unknown_faces(
                        self._gallery,
                        frame,
                        hits,
                        threshold=loop.match_threshold,
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
                    and streak >= loop.confirm_frames
                    and time.monotonic() >= cooldown_until
                ):
                    assert top.match is not None
                    settings = self._store.get()
                    try:
                        actions = perform_switch(
                            settings,
                            protected_hwnds=self._protected_hwnds,
                            protected_pids={os.getpid()},
                            dry_run=loop.dev_mode,
                        )
                        self.triggered.emit(
                            TriggerEvent(
                                person_name=top.match.person.name,
                                score=top.match.score,
                                actions=actions,
                                dry_run=loop.dev_mode,
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
                    cooldown_until = time.monotonic() + loop.cooldown_seconds
                    streak = 0

                now = time.perf_counter()
                dt = now - last
                last = now
                if dt > 0:
                    inst = 1.0 / dt
                    fps = inst if fps == 0 else fps * 0.85 + inst * 0.15

                build_rgb = should_build_preview_rgb(plan.need_preview, self._preview_in_flight)
                rgb = preview_rgb(frame, build_rgb)
                if build_rgb:
                    self._preview_in_flight = True
                self.frame_ready.emit(
                    PreviewFrame(
                        rgb=rgb,
                        hits=hits,
                        fps=fps,
                        streak=streak,
                        matched_name=name,
                        camera_ok=True,
                        camera_index=loop.camera_index,
                        threshold=loop.match_threshold,
                        best_score=best_score,
                        dev_mode=loop.dev_mode,
                        matched_armed=bool(triggerable),
                        seen=seen,
                    )
                )
                self.msleep(remaining_sleep_ms(plan.sleep_ms, time.perf_counter() - tick_started))
        finally:
            if cap is not None:
                cap.release()
            self.status.emit(t("log.cam_closed"))
