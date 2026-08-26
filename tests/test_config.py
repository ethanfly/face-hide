import tempfile
import unittest
from pathlib import Path

from facehide.config import (
    KvPair,
    MessageChannel,
    Settings,
    SettingsStore,
    WorkApp,
    load_settings,
    save_settings,
    settings_from_dict,
)


class ConfigTests(unittest.TestCase):
    def test_defaults_and_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            settings = Settings(
                match_threshold=0.51,
                work_apps=[WorkApp(id="a", name="Edge", path=r"C:\edge.exe")],
                entertainment_processes=["steam.exe", "GAME.EXE"],
            )
            save_settings(settings, path)
            loaded = load_settings(path)
            self.assertAlmostEqual(loaded.match_threshold, 0.51)
            self.assertEqual(loaded.work_apps[0].name, "Edge")
            self.assertEqual(loaded.entertainment_processes, ["steam.exe", "game.exe"])

    def test_corrupt_file_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text("{nope", encoding="utf-8")
            loaded = load_settings(path)
            self.assertEqual(loaded.work_apps, [])

    def test_store_copy_is_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SettingsStore(Path(tmp) / "config.json")
            one = store.get()
            one.camera_index = 3
            self.assertEqual(store.get().camera_index, 0)
            store.replace(one)
            self.assertEqual(store.get().camera_index, 3)

    def test_skips_empty_work_app(self) -> None:
        settings = settings_from_dict({"work_apps": [{"name": "x", "path": "  "}, {"path": "C:\\a.exe"}]})
        self.assertEqual(len(settings.work_apps), 1)
        self.assertEqual(settings.work_apps[0].path, "C:\\a.exe")

    def test_dev_mode_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            save_settings(Settings(dev_mode=True), path)
            self.assertTrue(load_settings(path).dev_mode)
            self.assertFalse(settings_from_dict({}).dev_mode)

    def test_auto_enroll_unknown_default_and_roundtrip(self) -> None:
        self.assertTrue(settings_from_dict({}).auto_enroll_unknown)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            save_settings(Settings(auto_enroll_unknown=False), path)
            self.assertFalse(load_settings(path).auto_enroll_unknown)

    def test_auto_link_defaults_on(self) -> None:
        self.assertTrue(settings_from_dict({}).auto_link_same_person)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            save_settings(Settings(auto_link_same_person=False), path)
            self.assertFalse(load_settings(path).auto_link_same_person)

    def test_start_minimized_defaults_off(self) -> None:
        self.assertFalse(settings_from_dict({}).start_minimized)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            save_settings(Settings(start_minimized=True), path)
            self.assertTrue(load_settings(path).start_minimized)

    def test_start_on_boot_defaults_off(self) -> None:
        self.assertFalse(settings_from_dict({}).start_on_boot)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            save_settings(Settings(start_on_boot=True), path)
            self.assertTrue(load_settings(path).start_on_boot)

    def test_language_defaults_and_normalizes(self) -> None:
        self.assertEqual(settings_from_dict({}).language, "zh")
        self.assertEqual(settings_from_dict({"language": "en-US"}).language, "en")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            save_settings(Settings(language="en"), path)
            self.assertEqual(load_settings(path).language, "en")

    def test_notify_template_defaults_and_roundtrip(self) -> None:
        self.assertEqual(settings_from_dict({}).notify_template, "classic")
        self.assertEqual(settings_from_dict({}).notify_name_mode, "full")
        self.assertEqual(settings_from_dict({"notify_template": "nope"}).notify_template, "classic")
        self.assertEqual(settings_from_dict({"notify_name_mode": "secret"}).notify_name_mode, "full")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            save_settings(Settings(notify_template="privacy", notify_name_mode="initial"), path)
            loaded = load_settings(path)
            self.assertEqual(loaded.notify_template, "privacy")
            self.assertEqual(loaded.notify_name_mode, "initial")

    def test_channels_roundtrip(self) -> None:
        self.assertEqual(settings_from_dict({}).channels, [])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            channel = MessageChannel(
                id="c1",
                kind="dingtalk_group",
                name="值班群",
                auth_mode="keyword",
                webhook="https://oapi.dingtalk.com/robot/send?access_token=x",
                keyword="告警",
                headers=[KvPair("X-Token", "abc")],
            )
            save_settings(Settings(channels=[channel]), path)
            loaded = load_settings(path)
            self.assertEqual(len(loaded.channels), 1)
            self.assertEqual(loaded.channels[0].kind, "dingtalk_group")
            self.assertEqual(loaded.channels[0].keyword, "告警")
            self.assertEqual(loaded.channels[0].headers[0].key, "X-Token")
            self.assertEqual(settings_from_dict({"channels": [{"kind": "nope"}]}).channels, [])


if __name__ == "__main__":
    unittest.main()
