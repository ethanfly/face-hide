import unittest

import numpy as np

from facehide.camera import CameraInfo, is_placeholder_frame, pick_camera


class CameraTests(unittest.TestCase):
    def test_logo_on_black_is_placeholder(self) -> None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[100:140, 140:180] = (40, 180, 220)
        self.assertTrue(is_placeholder_frame(frame))

    def test_dark_gray_logo_is_placeholder(self) -> None:
        frame = np.full((240, 320, 3), 27, dtype=np.uint8)
        frame[100:140, 140:180] = (40, 180, 220)
        self.assertTrue(is_placeholder_frame(frame))

    def test_textured_scene_is_not_placeholder(self) -> None:
        rng = np.random.default_rng(1)
        frame = rng.integers(30, 210, (240, 320, 3), dtype=np.uint8)
        self.assertFalse(is_placeholder_frame(frame))

    def test_pick_skips_placeholder(self) -> None:
        infos = [
            CameraInfo(0, 1280, 720, True),
            CameraInfo(1, 640, 480, False),
        ]
        self.assertEqual(pick_camera(infos, 0), 1)
        self.assertEqual(pick_camera(infos, 1), 1)

    def test_pick_keeps_only_placeholder_if_needed(self) -> None:
        infos = [CameraInfo(0, 640, 480, True)]
        self.assertEqual(pick_camera(infos, 0), 0)
