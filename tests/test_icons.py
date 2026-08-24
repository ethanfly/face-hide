import tempfile
import unittest
from pathlib import Path

from PIL import Image

from facehide.mark import (
    ICON_SIZES,
    STATUS_IDLE,
    STATUS_WATCHING,
    render_mark,
    save_ico,
)


class MarkTests(unittest.TestCase):
    def test_sizes_and_alpha(self) -> None:
        for size in ICON_SIZES:
            image = render_mark(size)
            self.assertEqual(image.size, (size, size))
            self.assertEqual(image.mode, "RGBA")
            self.assertLess(image.getpixel((0, 0))[3], 40)
            mid = image.getpixel((size // 2, size // 2))
            self.assertGreater(mid[3], 200)

    def test_status_changes_pixels(self) -> None:
        idle = render_mark(64, STATUS_IDLE)
        watching = render_mark(64, STATUS_WATCHING)
        self.assertNotEqual(idle.tobytes(), watching.tobytes())

    def test_save_ico(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = save_ico(Path(tmp) / "FaceHide.ico")
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 1024)
            with Image.open(path) as loaded:
                self.assertEqual(loaded.format, "ICO")
                sizes = loaded.info.get("sizes") or set()
                self.assertIn((16, 16), sizes)
                self.assertIn((32, 32), sizes)
                self.assertIn((256, 256), sizes)


class GlyphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_glyphs_and_app_icon(self) -> None:
        from facehide.ui.icons import app_icon, glyph_icon, glyph_pixmap, tray_status

        names = (
            "monitor",
            "faces",
            "work",
            "hide",
            "settings",
            "play",
            "stop",
            "bolt",
            "upload",
            "camera",
            "window",
            "power",
        )
        for name in names:
            pix = glyph_pixmap(name, 18)
            self.assertFalse(pix.isNull())
            self.assertFalse(glyph_icon(name).isNull())
        icon = app_icon()
        self.assertFalse(icon.isNull())
        self.assertGreater(len(icon.availableSizes()), 3)
        self.assertEqual(tray_status(False, False), "idle")
        self.assertEqual(tray_status(True, False), "watching")
        self.assertEqual(tray_status(True, True), "dev")
        with self.assertRaises(KeyError):
            glyph_pixmap("not-a-glyph")


if __name__ == "__main__":
    unittest.main()
