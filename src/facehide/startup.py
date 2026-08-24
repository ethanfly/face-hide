from __future__ import annotations

import sys
from pathlib import Path

from facehide.runtime import is_frozen

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
VALUE_NAME = "FaceHide"


def launch_command() -> str:
    if is_frozen():
        exe = str(Path(sys.executable).resolve())
        return f'"{exe}" --minimized'
    python = str(Path(sys.executable).resolve())
    return f'"{python}" -m facehide --minimized'


def is_enabled(name: str = VALUE_NAME) -> bool:
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_QUERY_VALUE) as key:
            value, _kind = winreg.QueryValueEx(key, name)
    except OSError:
        return False
    return bool(str(value or "").strip())


def sync_startup(enabled: bool, name: str = VALUE_NAME, command: str | None = None) -> None:
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
        if enabled:
            winreg.SetValueEx(key, name, 0, winreg.REG_SZ, command or launch_command())
            return
        try:
            winreg.DeleteValue(key, name)
        except FileNotFoundError:
            pass
