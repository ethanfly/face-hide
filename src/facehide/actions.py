from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import psutil

from facehide.config import Settings, WorkApp
from facehide.i18n import t

VK_LWIN = 0x5B
VK_ESCAPE = 0x1B
KEYEVENTF_KEYUP = 0x0002


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    pid: int
    exe: str
    path: str = ""


@dataclass(frozen=True)
class OpenApp:
    name: str
    path: str
    title: str
    exe: str
    pid: int = 0


SKIP_OPEN_APP_EXES = {
    "applicationframehost.exe",
    "systemsettings.exe",
    "textinputhost.exe",
    "searchhost.exe",
    "runtimebroker.exe",
    "shellexperiencehost.exe",
    "startmenuexperiencehost.exe",
    "lockapp.exe",
    "dwm.exe",
    "ctfmon.exe",
    "sihost.exe",
    "securityhealthsystray.exe",
    "widgets.exe",
    "widgetservice.exe",
    "phoneexperiencehost.exe",
    "crossdeviceresume.exe",
    "searchapp.exe",
}


@dataclass
class SwitchPlan:
    break_fullscreen: bool
    minimize_hwnds: list[int]
    show_desktop: bool
    focus_hwnds: list[int]
    launch: list[WorkApp]
    notes: list[str] = field(default_factory=list)

    def describe(self) -> list[str]:
        lines: list[str] = []
        if self.break_fullscreen:
            lines.append(t("action.fullscreen"))
        if self.minimize_hwnds:
            lines.append(t("action.minimize", count=len(self.minimize_hwnds)))
        if self.show_desktop:
            lines.append(t("action.desktop"))
        for hwnd in self.focus_hwnds:
            lines.append(t("action.focus_hwnd", hwnd=hwnd))
        for app in self.launch:
            lines.append(t("action.launch", name=app.name))
        lines.extend(self.notes)
        return lines


def describe_dev_switch(plan: SwitchPlan) -> list[str]:
    lines = plan.describe()
    return [t("action.dev", line=line) for line in lines] or [t("action.dev_none")]


def _win32():
    import win32con
    import win32gui
    import win32process

    return win32con, win32gui, win32process


def process_path(pid: int) -> str:
    try:
        return str(Path(psutil.Process(pid).exe()))
    except (psutil.Error, OSError):
        return ""


def process_exe(pid: int) -> str:
    path = process_path(pid)
    return Path(path).name.lower() if path else ""


def file_description(path: str) -> str:
    try:
        import win32api

        translations = win32api.GetFileVersionInfo(path, r"\VarFileInfo\Translation")
        lang, codepage = translations[0]
        key = rf"\StringFileInfo\{lang:04x}{codepage:04x}\FileDescription"
        desc = str(win32api.GetFileVersionInfo(path, key) or "").strip()
        return desc
    except Exception:
        return ""


def app_display_name(path: str, title: str = "") -> str:
    desc = file_description(path)
    if desc:
        return desc
    stem = Path(path).stem
    return stem or (title.strip() or "未命名")


def collect_windows() -> list[WindowInfo]:
    win32con, win32gui, win32process = _win32()
    found: list[WindowInfo] = []

    def callback(hwnd: int, _: object) -> bool:
        if not win32gui.IsWindowVisible(hwnd):
            return True
        style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        if style & win32con.WS_EX_TOOLWINDOW:
            return True
        title = win32gui.GetWindowText(hwnd).strip()
        if not title:
            return True
        _tid, pid = win32process.GetWindowThreadProcessId(hwnd)
        path = process_path(pid)
        exe = Path(path).name.lower() if path else process_exe(pid)
        found.append(WindowInfo(hwnd=hwnd, title=title, pid=pid, exe=exe, path=path))
        return True

    win32gui.EnumWindows(callback, None)
    return found


def open_apps_from_windows(
    windows: list[WindowInfo],
    *,
    exclude_pids: set[int] | None = None,
    skip_exes: set[str] | None = None,
) -> list[OpenApp]:
    excluded = exclude_pids or set()
    skipped = {name.lower() for name in (skip_exes or SKIP_OPEN_APP_EXES)}
    by_path: dict[str, OpenApp] = {}
    for window in windows:
        if window.pid in excluded:
            continue
        path = window.path.strip()
        if not path:
            continue
        exe = Path(path).name.lower()
        if exe in skipped:
            continue
        key = os.path.normcase(path)
        title = window.title.strip()
        existing = by_path.get(key)
        if existing is None or len(title) > len(existing.title):
            by_path[key] = OpenApp(
                name=app_display_name(path, title),
                path=path,
                title=title,
                exe=exe,
                pid=window.pid,
            )
    return sorted(by_path.values(), key=lambda item: (item.name.lower(), item.path.lower()))


def list_open_apps(*, exclude_pids: set[int] | None = None) -> list[OpenApp]:
    return open_apps_from_windows(collect_windows(), exclude_pids=exclude_pids)


def foreground_hwnd() -> int:
    import win32gui

    try:
        return int(win32gui.GetForegroundWindow() or 0)
    except Exception:
        return 0


def work_exes(settings: Settings) -> set[str]:
    return {Path(app.path).name.lower() for app in settings.work_apps if app.path}


def windows_to_minimize(
    windows: list[WindowInfo],
    settings: Settings,
    *,
    fg_hwnd: int,
    protected_hwnds: set[int],
    protected_pids: set[int],
) -> list[WindowInfo]:
    entertainment = {name.lower() for name in settings.entertainment_processes}
    protected_exes = work_exes(settings)
    picked: list[WindowInfo] = []
    seen: set[int] = set()
    for window in windows:
        if window.hwnd in protected_hwnds or window.pid in protected_pids:
            continue
        if window.exe in protected_exes:
            continue
        hide = False
        if window.exe and window.exe in entertainment:
            hide = True
        elif settings.minimize_other_windows:
            hide = True
        elif settings.hide_foreground and window.hwnd == fg_hwnd:
            hide = True
        if hide and window.hwnd not in seen:
            seen.add(window.hwnd)
            picked.append(window)
    return picked


def plan_switch(
    settings: Settings,
    windows: list[WindowInfo],
    *,
    fg_hwnd: int,
    protected_hwnds: set[int],
    protected_pids: set[int],
) -> SwitchPlan:
    hide = windows_to_minimize(
        windows,
        settings,
        fg_hwnd=fg_hwnd,
        protected_hwnds=protected_hwnds,
        protected_pids=protected_pids,
    )
    focus: list[int] = []
    launch: list[WorkApp] = []
    for app in settings.work_apps:
        exe = Path(app.path).name.lower()
        existing = [item for item in windows if item.exe == exe and item.hwnd not in protected_hwnds]
        if existing:
            focus.append(existing[0].hwnd)
        else:
            launch.append(app)
    show_desktop = not settings.work_apps
    return SwitchPlan(
        break_fullscreen=bool(settings.break_fullscreen and fg_hwnd and fg_hwnd not in protected_hwnds),
        minimize_hwnds=[item.hwnd for item in hide],
        show_desktop=show_desktop,
        focus_hwnds=focus,
        launch=launch,
    )


def _tap(vk: int) -> None:
    import ctypes

    user32 = ctypes.windll.user32
    user32.keybd_event(vk, 0, 0, 0)
    user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)


def break_exclusive_fullscreen() -> None:
    _tap(VK_LWIN)
    time.sleep(0.12)
    _tap(VK_ESCAPE)
    time.sleep(0.08)


def minimize_hwnd(hwnd: int) -> None:
    import win32con
    import win32gui

    try:
        win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
    except Exception:
        pass


def restore_and_focus(hwnd: int) -> None:
    import ctypes

    import win32api
    import win32con
    import win32gui
    import win32process

    try:
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        fg = win32gui.GetForegroundWindow()
        cur_tid = win32api.GetCurrentThreadId()
        fg_tid, _ = win32process.GetWindowThreadProcessId(fg) if fg else (0, 0)
        tgt_tid, _ = win32process.GetWindowThreadProcessId(hwnd)
        if fg_tid and fg_tid != cur_tid:
            ctypes.windll.user32.AttachThreadInput(cur_tid, fg_tid, True)
        if tgt_tid and tgt_tid != cur_tid:
            ctypes.windll.user32.AttachThreadInput(cur_tid, tgt_tid, True)
        win32gui.SetForegroundWindow(hwnd)
        win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
        if tgt_tid and tgt_tid != cur_tid:
            ctypes.windll.user32.AttachThreadInput(cur_tid, tgt_tid, False)
        if fg_tid and fg_tid != cur_tid:
            ctypes.windll.user32.AttachThreadInput(cur_tid, fg_tid, False)
    except Exception:
        pass


def show_desktop() -> None:
    import ctypes

    user32 = ctypes.windll.user32
    user32.keybd_event(VK_LWIN, 0, 0, 0)
    user32.keybd_event(0x4D, 0, 0, 0)
    user32.keybd_event(0x4D, 0, KEYEVENTF_KEYUP, 0)
    user32.keybd_event(VK_LWIN, 0, KEYEVENTF_KEYUP, 0)


def launch_app(app: WorkApp) -> str:
    path = Path(app.path)
    if not path.exists():
        raise FileNotFoundError(app.path)
    if app.args.strip():
        argv = [str(path), *shlex.split(app.args, posix=False)]
        subprocess.Popen(argv, cwd=str(path.parent), close_fds=True)
    else:
        os.startfile(str(path))  # noqa: S606 — 用户指定的本地程序
    return t("action.launch", name=app.name)


def execute_plan(plan: SwitchPlan) -> list[str]:
    done: list[str] = []
    if plan.break_fullscreen:
        break_exclusive_fullscreen()
        done.append(t("action.fullscreen"))
    for hwnd in plan.minimize_hwnds:
        minimize_hwnd(hwnd)
    if plan.minimize_hwnds:
        done.append(t("action.minimize", count=len(plan.minimize_hwnds)))
    if plan.show_desktop:
        show_desktop()
        done.append(t("action.desktop"))
    for hwnd in plan.focus_hwnds:
        restore_and_focus(hwnd)
        done.append(t("action.focus_work"))
    for app in plan.launch:
        try:
            done.append(launch_app(app))
        except OSError as exc:
            done.append(t("action.launch_fail", name=app.name, error=exc))
    return done


def perform_switch(
    settings: Settings,
    *,
    protected_hwnds: set[int],
    protected_pids: set[int],
    dry_run: bool = False,
) -> list[str]:
    windows = collect_windows()
    plan = plan_switch(
        settings,
        windows,
        fg_hwnd=foreground_hwnd(),
        protected_hwnds=protected_hwnds,
        protected_pids=protected_pids,
    )
    if dry_run:
        return describe_dev_switch(plan)
    return execute_plan(plan)


def common_work_apps() -> list[tuple[str, str]]:
    candidates = [
        (
            "Microsoft Edge",
            [
                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            ],
        ),
        (
            "Google Chrome",
            [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            ],
        ),
        (
            "Firefox",
            [
                r"C:\Program Files\Mozilla Firefox\firefox.exe",
                r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe",
            ],
        ),
        (
            "Visual Studio Code",
            [os.path.expandvars(r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe")],
        ),
        ("记事本", [r"C:\Windows\System32\notepad.exe"]),
        (
            "Word",
            [
                r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE",
                r"C:\Program Files (x86)\Microsoft Office\root\Office16\WINWORD.EXE",
            ],
        ),
        (
            "Excel",
            [
                r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE",
                r"C:\Program Files (x86)\Microsoft Office\root\Office16\EXCEL.EXE",
            ],
        ),
    ]
    found: list[tuple[str, str]] = []
    for name, paths in candidates:
        for path in paths:
            if Path(path).exists():
                found.append((name, path))
                break
    return found


def running_process_names() -> list[str]:
    names: set[str] = set()
    for proc in psutil.process_iter(["name"]):
        name = (proc.info.get("name") or "").lower()
        if name:
            names.add(name)
    return sorted(names)
