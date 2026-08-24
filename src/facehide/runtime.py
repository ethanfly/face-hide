from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def bundle_dir() -> Path:
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parent


def exe_dir() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def asset_path(name: str) -> Path:
    candidates = [
        bundle_dir() / "facehide" / "ui" / name,
        bundle_dir() / "assets" / name,
        Path(__file__).resolve().parent / "ui" / name,
        exe_dir() / name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]
