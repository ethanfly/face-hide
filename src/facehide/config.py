from __future__ import annotations

import copy
import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from facehide.i18n import normalize_language
from facehide.paths import config_path


@dataclass
class WorkApp:
    id: str
    name: str
    path: str
    args: str = ""


@dataclass
class Settings:
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    detect_score: float = 0.7
    match_threshold: float = 0.40
    confirm_frames: int = 4
    cooldown_seconds: float = 12.0
    hide_foreground: bool = True
    minimize_other_windows: bool = False
    break_fullscreen: bool = True
    auto_start_monitor: bool = False
    start_minimized: bool = False
    dev_mode: bool = False
    auto_link_same_person: bool = True
    language: str = "zh"
    work_apps: list[WorkApp] = field(default_factory=list)
    entertainment_processes: list[str] = field(default_factory=list)

    def copy(self) -> Settings:
        return copy.deepcopy(self)


def _work_app_from(raw: Any) -> WorkApp | None:
    if not isinstance(raw, dict):
        return None
    path = str(raw.get("path") or "").strip()
    if not path:
        return None
    return WorkApp(
        id=str(raw.get("id") or path),
        name=str(raw.get("name") or Path(path).stem),
        path=path,
        args=str(raw.get("args") or ""),
    )


def settings_from_dict(data: dict[str, Any]) -> Settings:
    apps = []
    for item in data.get("work_apps") or []:
        app = _work_app_from(item)
        if app:
            apps.append(app)
    processes = []
    for item in data.get("entertainment_processes") or []:
        name = str(item).strip().lower()
        if name:
            processes.append(name)
    return Settings(
        camera_index=int(data.get("camera_index", 0)),
        frame_width=int(data.get("frame_width", 640)),
        frame_height=int(data.get("frame_height", 480)),
        detect_score=float(data.get("detect_score", 0.7)),
        match_threshold=float(data.get("match_threshold", 0.40)),
        confirm_frames=max(1, int(data.get("confirm_frames", 4))),
        cooldown_seconds=max(1.0, float(data.get("cooldown_seconds", 12.0))),
        hide_foreground=bool(data.get("hide_foreground", True)),
        minimize_other_windows=bool(data.get("minimize_other_windows", False)),
        break_fullscreen=bool(data.get("break_fullscreen", True)),
        auto_start_monitor=bool(data.get("auto_start_monitor", False)),
        start_minimized=bool(data.get("start_minimized", False)),
        dev_mode=bool(data.get("dev_mode", False)),
        auto_link_same_person=bool(data.get("auto_link_same_person", True)),
        language=normalize_language(data.get("language", "zh")),
        work_apps=apps,
        entertainment_processes=processes,
    )


def settings_to_dict(settings: Settings) -> dict[str, Any]:
    return asdict(settings)


def load_settings(path: Path | None = None) -> Settings:
    target = path or config_path()
    if not target.exists():
        return Settings()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return Settings()
    if not isinstance(data, dict):
        return Settings()
    return settings_from_dict(data)


def save_settings(settings: Settings, path: Path | None = None) -> None:
    target = path or config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(settings_to_dict(settings), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class SettingsStore:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path or config_path()
        self._lock = threading.Lock()
        self._settings = load_settings(self._path)

    def get(self) -> Settings:
        with self._lock:
            return self._settings.copy()

    def replace(self, settings: Settings) -> Settings:
        with self._lock:
            self._settings = settings.copy()
            save_settings(self._settings, self._path)
            return self._settings.copy()
