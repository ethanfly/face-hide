import uuid
import unittest
from unittest.mock import patch

from facehide.startup import is_enabled, launch_command, sync_startup


class StartupTests(unittest.TestCase):
    def test_frozen_command_quotes_exe(self) -> None:
        with patch("facehide.startup.is_frozen", return_value=True), patch(
            "facehide.startup.sys.executable", r"C:\Program Files\FaceHide\FaceHide.exe"
        ):
            cmd = launch_command()
        self.assertEqual(cmd, r'"C:\Program Files\FaceHide\FaceHide.exe" --minimized')

    def test_dev_command_uses_module(self) -> None:
        with patch("facehide.startup.is_frozen", return_value=False), patch(
            "facehide.startup.sys.executable", r"C:\Python\python.exe"
        ):
            cmd = launch_command()
        self.assertEqual(cmd, r'"C:\Python\python.exe" -m facehide --minimized')

    def test_registry_roundtrip(self) -> None:
        name = f"FaceHide.Test.{uuid.uuid4().hex}"
        self.addCleanup(lambda: sync_startup(False, name=name))
        sync_startup(True, name=name, command='"C:\\\\FaceHide.exe" --minimized')
        self.assertTrue(is_enabled(name))
        sync_startup(False, name=name)
        self.assertFalse(is_enabled(name))

    def test_disable_when_run_key_missing(self) -> None:
        def missing(*_args, **_kwargs):
            raise FileNotFoundError(2, "The system cannot find the file specified")

        with patch("winreg.OpenKey", missing):
            sync_startup(False, name="FaceHide.Missing")

    def test_enable_creates_run_key(self) -> None:
        source = __import__("inspect").getsource(sync_startup)
        self.assertIn("CreateKeyEx", source)
        self.assertIn("KEY_SET_VALUE", source)


if __name__ == "__main__":
    unittest.main()
