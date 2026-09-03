import inspect
import unittest

import numpy as np

from facehide.engine import (
    DETECT_MAX_SIDE,
    FaceEngine,
    detect_working_size,
    hits_from_detections,
    map_box_to_source,
    should_extract_features,
    working_view,
)
from facehide.infer.yunet import YUNET_OUTPUTS, empty_yunet_outputs
from facehide.models import sface_path, yunet_path


class _BoomDml:
    def __init__(self) -> None:
        self.n = 0
        self.input_name = "input"
        self.output_names = list(YUNET_OUTPUTS)

    def run(self, names, feed):
        if "data" in feed:
            return [np.zeros((1, 128), dtype=np.float32)]
        self.n += 1
        if self.n == 2:
            raise RuntimeError("dml boom")
        return empty_yunet_outputs()


class EngineTests(unittest.TestCase):
    def test_blank_image_has_no_faces(self) -> None:
        if not yunet_path().exists() or not sface_path().exists():
            self.skipTest("models not downloaded")
        engine = FaceEngine(device="cpu")
        hits = engine.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertEqual(hits, [])
        self.assertEqual(engine.extract_count, 0)

    def test_enroll_rejects_blank(self) -> None:
        if not yunet_path().exists() or not sface_path().exists():
            self.skipTest("models not downloaded")
        from facehide.engine import NoFaceError

        engine = FaceEngine(device="cpu")
        with self.assertRaises(NoFaceError):
            engine.enroll(np.zeros((480, 640, 3), dtype=np.uint8))

    def test_working_image_is_capped_below_720p(self) -> None:
        src_w, src_h = 1280, 720
        work_w, work_h = detect_working_size(src_w, src_h)
        self.assertLess(work_w * work_h, src_w * src_h)
        self.assertLessEqual(max(work_w, work_h), DETECT_MAX_SIDE)
        huge_w, huge_h = detect_working_size(1920, 1080)
        self.assertLess(huge_w * huge_h, 1920 * 1080)
        self.assertLessEqual(max(huge_w, huge_h), DETECT_MAX_SIDE)
        native_w, native_h = detect_working_size(src_w, src_h, max_side=0)
        self.assertEqual((native_w, native_h), (src_w, src_h))
        src = np.zeros((src_h, src_w, 3), dtype=np.uint8)
        work, scale_x, scale_y = working_view(src, DETECT_MAX_SIDE)
        self.assertEqual((work.shape[1], work.shape[0]), (work_w, work_h))
        self.assertAlmostEqual(work.shape[1] * scale_x, src_w, delta=2)
        self.assertAlmostEqual(work.shape[0] * scale_y, src_h, delta=2)

    def test_boxes_map_back_to_source_coordinates(self) -> None:
        src = (1280, 720)
        work = detect_working_size(*src)
        x, y, w, h = 40, 20, 30, 24
        mx, my, mw, mh = map_box_to_source(x, y, w, h, src, work)
        self.assertGreater(mw, w)
        self.assertGreater(mh, h)
        self.assertAlmostEqual(mx / float(x), src[0] / float(work[0]), delta=0.05)
        self.assertAlmostEqual(my / float(y), src[1] / float(work[1]), delta=0.05)
        self.assertLessEqual(mx + mw, src[0] + 2)
        self.assertLessEqual(my + mh, src[1] + 2)

    def test_zero_faces_does_not_request_features(self) -> None:
        self.assertFalse(should_extract_features(0))
        self.assertTrue(should_extract_features(1))
        called: list[np.ndarray] = []

        def feature_fn(raw: np.ndarray) -> np.ndarray:
            called.append(raw)
            return np.ones(4, dtype=np.float32)

        none_hits = hits_from_detections(
            None, scale_x=2.0, scale_y=2.0, extract_features=True, feature_fn=feature_fn
        )
        empty_hits = hits_from_detections(
            np.zeros((0, 15), dtype=np.float32),
            scale_x=2.0,
            scale_y=2.0,
            extract_features=True,
            feature_fn=feature_fn,
        )
        self.assertEqual(none_hits, [])
        self.assertEqual(empty_hits, [])
        self.assertEqual(called, [])

        face = np.array(
            [10, 8, 16, 12, 12, 10, 20, 10, 16, 14, 12, 18, 20, 18, 0.9],
            dtype=np.float32,
        )
        hits = hits_from_detections(
            np.stack([face]),
            scale_x=2.0,
            scale_y=2.0,
            extract_features=True,
            feature_fn=feature_fn,
        )
        self.assertEqual(len(hits), 1)
        self.assertEqual(len(called), 1)
        self.assertEqual((hits[0].x, hits[0].y, hits[0].w, hits[0].h), (20, 16, 32, 24))
        skipped = hits_from_detections(
            np.stack([face]),
            scale_x=2.0,
            scale_y=2.0,
            extract_features=False,
            feature_fn=feature_fn,
        )
        self.assertEqual(len(called), 1)
        self.assertIsNone(skipped[0].feature)

    def test_detect_uses_working_view_and_skips_sface_on_empty(self) -> None:
        source = inspect.getsource(FaceEngine.detect)
        self.assertIn("working_view", source)
        self.assertIn("hits_from_detections", source)
        self.assertIn("max_side", source)
        if not yunet_path().exists() or not sface_path().exists():
            return
        engine = FaceEngine(device="cpu")
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        hits = engine.detect(blank, extract_features=True, max_side=DETECT_MAX_SIDE)
        self.assertEqual(hits, [])
        self.assertEqual(engine.extract_count, 0)

    def test_enroll_stays_on_native_resolution(self) -> None:
        source = inspect.getsource(FaceEngine.enroll_all)
        self.assertIn("extract_features=True", source)
        self.assertNotIn("max_side", source)

    def test_stub_second_run_falls_back_to_cpu(self) -> None:
        if not yunet_path().exists() or not sface_path().exists():
            self.skipTest("models not downloaded")
        from facehide.infer.session import OrtSessionFactory

        runner = _BoomDml()
        engine = FaceEngine(device="gpu", session_factory=OrtSessionFactory(runner))
        blank = np.full((480, 640, 3), 40, dtype=np.uint8)
        first = engine.detect(blank, extract_features=False)
        self.assertEqual(first, [])
        self.assertTrue(engine._live_det.uses_fixed_input)
        self.assertTrue(engine._flag.dead)
        self.assertEqual(runner.n, 2)
        second = engine.detect(blank, extract_features=False)
        self.assertEqual(second, [])
        self.assertEqual(runner.n, 2)
        self.assertFalse(engine._live_is_dml)
        info = engine.backend_info()
        self.assertTrue(info.fallback)
        self.assertEqual(info.provider, "CPU")
        third = engine.detect(blank, extract_features=False)
        self.assertEqual(third, [])
        self.assertEqual(runner.n, 2)
        self.assertFalse(engine._live_is_dml)
        self.assertEqual(engine.backend_info().provider, "CPU")
        self.assertIsNotNone(engine.consume_fallback())

    def test_reconfigure_gpu_retries_after_fallback(self) -> None:
        if not yunet_path().exists() or not sface_path().exists():
            self.skipTest("models not downloaded")
        from facehide.infer.session import OrtSessionFactory

        runner = _BoomDml()
        engine = FaceEngine(device="gpu", session_factory=OrtSessionFactory(runner))
        blank = np.full((480, 640, 3), 40, dtype=np.uint8)
        engine.detect(blank, extract_features=False)
        self.assertEqual(runner.n, 2)
        self.assertTrue(engine._flag.dead)
        engine.detect(blank, extract_features=False)
        self.assertFalse(engine._live_is_dml)
        engine.reconfigure("gpu")
        after = engine.detect(blank, extract_features=False)
        self.assertEqual(after, [])
        self.assertGreater(runner.n, 2)
        self.assertTrue(engine._live_is_dml)
        self.assertTrue(engine._live_det.uses_fixed_input)
        self.assertEqual(engine.backend_info().provider, "DmlExecutionProvider")
        self.assertFalse(engine.backend_info().fallback)

