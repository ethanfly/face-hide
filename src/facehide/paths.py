from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "FaceHide"


def app_data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    path = Path(base) / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    path = app_data_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def gallery_dir() -> Path:
    path = app_data_dir() / "gallery"
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_path() -> Path:
    return app_data_dir() / "config.json"


def gallery_index_path() -> Path:
    return app_data_dir() / "gallery.json"
