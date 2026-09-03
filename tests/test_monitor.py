import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np

from facehide.config import Settings
from facehide.engine import DETECT_MAX_SIDE, FaceHit
from facehide.gallery import Gallery
from facehide.i18n import set_language
from facehide.monitor import (
    MonitorThread,
    SeenFace,
    enroll_unknown_faces,
    plan_tick,
    preview_needed,
    preview_rgb,
    remaining_sleep_ms,
    should_build_preview_rgb,
    tick_sleep_ms,
    track_seen,
)


class TrackSeenTests(unittest.TestCase):
    def test_logs_new_face_once_until_gone(self) -> None:
        ada = SeenFace("Ada", 0.82, True)
        active, newly = track_seen({}, {"Ada": ada}, 1.0)
        self.assertEqual([item.name for item in newly], ["Ada"])
        active, newly = track_seen(active, {"Ada": ada}, 1.2)
        self.assertEqual(newly, [])
        active, newly = track_seen(active, {}, 1.4)
        self.assertEqual(newly, [])
        self.assertIn("Ada", active)
        active, newly = track_seen(active, {}, 3.0)
        self.assertEqual(active, {})
        self.assertEqual(newly, [])
        active, newly = track_seen(active, {"Ada": ada}, 3.1)
        self.assertEqual([item.name for item in newly], ["Ada"])

    def test_brief_dropout_does_not_relog(self) -> None:
        bob = SeenFace("Bob", 0.77, False)
        active, newly = track_seen({}, {"Bob": bob}, 10.0)
        self.assertEqual(len(newly), 1)
        active, newly = track_seen(active, {}, 10.5)
        self.assertEqual(newly, [])
        active, newly = track_seen(active, {"Bob": bob}, 11.0)
        self.assertEqual(newly, [])
        self.assertIn("Bob", active)


def _hit(x: int, feature: np.ndarray) -> FaceHit:
    return FaceHit(x=x, y=4, w=20, h=20, det_score=0.9, raw=np.zeros(15), feature=feature)


def _frame() -> np.ndarray:
    return np.full((64, 96, 3), 120, dtype=np.uint8)


class AutoEnrollUnknownTests(unittest.TestCase):
    def setUp(self) -> None:
        set_language("zh")

    def tearDown(self) -> None:
        set_language("zh")

    def test_unknown_face_enrolled_disabled_and_unnamed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gallery = Gallery(root / "gallery.json", root)
            recent: list[tuple[float, np.ndarray]] = []
            created = enroll_unknown_faces(
                gallery,
                _frame(),
                [_hit(8, np.array([1.0, 0.0], dtype=np.float32))],
                threshold=0.4,
                recent=recent,
                now=1.0,
            )
            self.assertEqual(len(created), 1)
            person = created[0]
            self.assertFalse(person.enabled)
            self.assertFalse(person.blacklisted)
            self.assertEqual(person.name, "未知人脸 1")
            self.assertEqual(len(gallery.people()), 1)

    def test_known_face_not_enrolled_again(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gallery = Gallery(root / "gallery.json", root)
            gallery.add_person("老板", np.array([1.0, 0.0], dtype=np.float32), _frame()[4:24, 8:28])
            recent: list[tuple[float, np.ndarray]] = []
            created = enroll_unknown_faces(
                gallery,
                _frame(),
                [_hit(8, np.array([0.99, 0.01], dtype=np.float32))],
                threshold=0.4,
                recent=recent,
                now=1.0,
            )
            self.assertEqual(created, [])
            self.assertEqual(len(gallery.people()), 1)

    def test_recent_grace_prevents_duplicate_enroll(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gallery = Gallery(root / "gallery.json", root)
            recent: list[tuple[float, np.ndarray]] = []
            feature = np.array([1.0, 0.0], dtype=np.float32)
            first = enroll_unknown_faces(
                gallery, _frame(), [_hit(8, feature)], threshold=0.4, recent=recent, now=1.0
            )
            second = enroll_unknown_faces(
                gallery, _frame(), [_hit(8, feature)], threshold=0.4, recent=recent, now=2.0
            )
            self.assertEqual(len(first), 1)
            self.assertEqual(second, [])
            self.assertEqual(len(gallery.people()), 1)

    def test_two_unknown_faces_get_numbered_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gallery = Gallery(root / "gallery.json", root)
            recent: list[tuple[float, np.ndarray]] = []
            created = enroll_unknown_faces(
                gallery,
                _frame(),
                [
                    _hit(8, np.array([1.0, 0.0], dtype=np.float32)),
                    _hit(40, np.array([0.0, 1.0], dtype=np.float32)),
                ],
                threshold=0.4,
                recent=recent,
                now=1.0,
            )
            self.assertEqual([item.name for item in created], ["未知人脸 1", "未知人脸 2"])


class TickHelperTests(unittest.TestCase):
    def test_hidden_tick_skips_full_size_rgb(self) -> None:
        src = np.zeros((720, 1280, 3), dtype=np.uint8)
        src[:, :] = (20, 80, 160)
        hidden = preview_rgb(src, preview_needed(False))
        shown = preview_rgb(src, preview_needed(True))
        self.assertEqual(hidden.size, 0)
        self.assertLess(hidden.size, src.size)
        self.assertEqual(shown.shape, (720, 1280, 3))
        self.assertGreater(shown.size, hidden.size)
        extra = preview_rgb(src, preview_needed(False, extra=1))
        self.assertEqual(extra.shape[:2], src.shape[:2])
        dropped = preview_rgb(src, should_build_preview_rgb(True, True))
        self.assertEqual(dropped.size, 0)
        inflight = preview_rgb(src, should_build_preview_rgb(True, False))
        self.assertEqual(inflight.shape[:2], src.shape[:2])
        self.assertFalse(should_build_preview_rgb(False, False))

    def test_success_path_interval_is_positive_and_hidden_stays_near_a_second(self) -> None:
        hidden = plan_tick(False)
        visible = plan_tick(True)
        self.assertGreater(hidden.sleep_ms, 0)
        self.assertGreater(visible.sleep_ms, 0)
        self.assertGreater(tick_sleep_ms(False), 0)
        self.assertLessEqual(hidden.sleep_ms, 1000)
        confirm = Settings().confirm_frames
        self.assertLessEqual(confirm * hidden.sleep_ms, 1000)
        self.assertGreater(hidden.sleep_ms, visible.sleep_ms)
        leftover = remaining_sleep_ms(hidden.sleep_ms, 0.0)
        self.assertGreater(leftover, 0)
        self.assertEqual(remaining_sleep_ms(hidden.sleep_ms, hidden.sleep_ms / 1000.0 + 1.0), 0)

    def test_plan_uses_downscaled_detect_side(self) -> None:
        plan = plan_tick(False)
        src_w, src_h = 1280, 720
        self.assertLess(plan.detect_max_side * plan.detect_max_side, src_w * src_h)
        self.assertEqual(plan.detect_max_side, DETECT_MAX_SIDE)
        self.assertLessEqual(plan.detect_max_side, 640)

    def test_run_loop_calls_shipped_tick_helpers(self) -> None:
        source = inspect.getsource(MonitorThread.run)
        self.assertIn("plan_tick", source)
        self.assertIn("preview_rgb", source)
        self.assertIn("should_build_preview_rgb", source)
        self.assertIn("remaining_sleep_ms", source)
        self.assertIn("loop_settings", source)
        self.assertIn("detect_max_side", source)
        self.assertIn("extract_features", source)
        self.assertNotIn("cvtColor", source)
        self.assertIn("match_threshold", source)
        self.assertIn("confirm_frames", source)
        self.assertIn("auto_enroll_unknown", source)
        self.assertIn("inference_device", source)
        self.assertIn("dev_mode", source)
        self.assertIn("cooldown_seconds", source)
        self.assertIn("perform_switch", source)
        self.assertIn("dry_run", source)


class PreviewUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_hidden_payload_does_not_render_pixmap(self) -> None:
        from facehide.monitor import PreviewFrame
        from facehide.ui.main_window import CaptureDialog, MainWindow, _render_preview

        src = np.zeros((720, 1280, 3), dtype=np.uint8)
        hidden = PreviewFrame(
            rgb=preview_rgb(src, preview_needed(False)),
            hits=[],
            fps=12.0,
            streak=0,
            matched_name=None,
            camera_ok=True,
        )
        shown = PreviewFrame(
            rgb=preview_rgb(src, preview_needed(True)),
            hits=[],
            fps=12.0,
            streak=0,
            matched_name=None,
            camera_ok=True,
        )
        hidden_pix = _render_preview(hidden)
        shown_pix = _render_preview(shown)
        self.assertTrue(hidden_pix.isNull())
        self.assertFalse(shown_pix.isNull())
        self.assertEqual((shown_pix.width(), shown_pix.height()), (1280, 720))
        sync = inspect.getsource(MainWindow._sync_preview_needed)
        self.assertIn("set_preview_needed", sync)
        self.assertIn("isMinimized", sync)
        on_frame = inspect.getsource(MainWindow._on_frame)
        self.assertIn("_flush_preview", on_frame)
        self.assertIn("_update_pills", on_frame)
        self.assertIn("add_preview_extra", inspect.getsource(CaptureDialog.__init__))
        self.assertIn("remove_preview_extra", inspect.getsource(CaptureDialog.done))

    def test_settings_persist_inference_device(self) -> None:
        from facehide.ui.main_window import MainWindow

        collect = inspect.getsource(MainWindow._collect_settings)
        reload_all = inspect.getsource(MainWindow.reload_all)
        apply_lang = inspect.getsource(MainWindow._apply_language)
        self.assertIn("inference_device", collect)
        self.assertIn("inference_device", reload_all)
        self.assertIn("settings.device", apply_lang)
        self.assertIn("device_box", inspect.getsource(MainWindow._build_settings))



if __name__ == "__main__":
    unittest.main()
