import unittest

from facehide.actions import (
    SwitchPlan,
    WindowInfo,
    describe_dev_switch,
    open_apps_from_windows,
    plan_switch,
    windows_to_minimize,
)
from facehide.config import Settings, WorkApp
from facehide.i18n import set_language


def _win(hwnd: int, exe: str, pid: int = 0) -> WindowInfo:
    return WindowInfo(hwnd=hwnd, title=exe, pid=pid or hwnd, exe=exe)


class ActionPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        set_language("zh")

    def test_hide_entertainment_and_foreground(self) -> None:
        settings = Settings(
            hide_foreground=True,
            entertainment_processes=["game.exe"],
            work_apps=[WorkApp(id="1", name="Edge", path=r"C:\msedge.exe")],
        )
        windows = [_win(1, "game.exe"), _win(2, "chrome.exe"), _win(3, "msedge.exe")]
        hidden = windows_to_minimize(
            windows,
            settings,
            fg_hwnd=2,
            protected_hwnds={99},
            protected_pids={100},
        )
        self.assertEqual({item.exe for item in hidden}, {"game.exe", "chrome.exe"})

    def test_never_hides_protected_or_work(self) -> None:
        settings = Settings(minimize_other_windows=True, work_apps=[WorkApp(id="1", name="Code", path=r"C:\Code.exe")])
        windows = [_win(10, "code.exe", pid=7), _win(11, "notepad.exe", pid=8), _win(12, "python.exe", pid=9)]
        hidden = windows_to_minimize(
            windows,
            settings,
            fg_hwnd=11,
            protected_hwnds={12},
            protected_pids={9},
        )
        self.assertEqual([item.exe for item in hidden], ["notepad.exe"])

    def test_empty_work_apps_shows_desktop(self) -> None:
        settings = Settings(work_apps=[], hide_foreground=True)
        plan = plan_switch(settings, [_win(1, "game.exe")], fg_hwnd=1, protected_hwnds=set(), protected_pids=set())
        self.assertTrue(plan.show_desktop)
        self.assertEqual(plan.launch, [])
        self.assertEqual(plan.focus_hwnds, [])

    def test_running_work_app_is_focused(self) -> None:
        settings = Settings(work_apps=[WorkApp(id="1", name="Edge", path=r"C:\msedge.exe")])
        plan = plan_switch(
            settings,
            [_win(5, "msedge.exe")],
            fg_hwnd=1,
            protected_hwnds=set(),
            protected_pids=set(),
        )
        self.assertFalse(plan.show_desktop)
        self.assertEqual(plan.focus_hwnds, [5])
        self.assertEqual(plan.launch, [])

    def test_missing_work_app_is_launched(self) -> None:
        app = WorkApp(id="1", name="Edge", path=r"C:\msedge.exe")
        settings = Settings(work_apps=[app], break_fullscreen=False)
        plan = plan_switch(settings, [], fg_hwnd=0, protected_hwnds=set(), protected_pids=set())
        self.assertEqual(plan.launch, [app])
        self.assertFalse(plan.show_desktop)

    def test_describe_dev_switch_is_dry_run(self) -> None:
        plan = SwitchPlan(
            break_fullscreen=True,
            minimize_hwnds=[1, 2],
            show_desktop=True,
            focus_hwnds=[],
            launch=[],
        )
        lines = describe_dev_switch(plan)
        self.assertTrue(all(line.startswith("[开发]") for line in lines))
        self.assertIn("[开发] 退出全屏", lines)
        self.assertIn("[开发] 显示桌面", lines)

    def test_empty_dev_switch(self) -> None:
        plan = SwitchPlan(False, [], False, [], [])
        self.assertEqual(describe_dev_switch(plan), ["[开发] 无窗口动作"])

    def test_open_apps_dedupes_by_path(self) -> None:
        windows = [
            WindowInfo(1, "Doc A", 10, "winword.exe", r"C:\Office\WINWORD.EXE"),
            WindowInfo(2, "A longer Word document title", 10, "winword.exe", r"C:\Office\WINWORD.EXE"),
            WindowInfo(3, "New Tab", 11, "chrome.exe", r"C:\Chrome\chrome.exe"),
        ]
        apps = open_apps_from_windows(windows)
        self.assertEqual([item.exe for item in apps], ["chrome.exe", "winword.exe"])
        word = next(item for item in apps if item.exe == "winword.exe")
        self.assertEqual(word.title, "A longer Word document title")

    def test_open_apps_skips_hosts_and_excluded(self) -> None:
        windows = [
            WindowInfo(1, "Settings", 1, "systemsettings.exe", r"C:\Windows\System32\SystemSettings.exe"),
            WindowInfo(2, "记事本", 2, "notepad.exe", r"C:\not-real\notepad.exe"),
            WindowInfo(3, "Self", 99, "python.exe", r"C:\not-real\python.exe"),
            WindowInfo(4, "No path", 3, "foo.exe", ""),
        ]
        apps = open_apps_from_windows(windows, exclude_pids={99})
        self.assertEqual([item.exe for item in apps], ["notepad.exe"])
        self.assertEqual(apps[0].name, "notepad")


if __name__ == "__main__":
    unittest.main()
