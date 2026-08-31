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
from facehide.models import sface_path, yunet_path


class EngineTests(unittest.TestCase):
    def test_blank_image_has_no_faces(self) -> None:
        if not yunet_path().exists() or not sface_path().exists():
            self.skipTest("models not downloaded")
        engine = FaceEngine()
        hits = engine.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertEqual(hits, [])
        self.assertEqual(engine.extract_count, 0)

    def test_enroll_rejects_blank(self) -> None:
        if not yunet_path().exists() or not sface_path().exists():
            self.skipTest("models not downloaded")
        from facehide.engine import NoFaceError

        engine = FaceEngine()
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
        engine = FaceEngine()
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        hits = engine.detect(blank, extract_features=True, max_side=DETECT_MAX_SIDE)
        self.assertEqual(hits, [])
        self.assertEqual(engine.extract_count, 0)

    def test_enroll_stays_on_native_resolution(self) -> None:
        source = inspect.getsource(FaceEngine.enroll_all)
        self.assertIn("extract_features=True", source)
        self.assertNotIn("max_side", source)