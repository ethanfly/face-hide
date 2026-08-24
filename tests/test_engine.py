import unittest

import numpy as np

from facehide.engine import FaceEngine
from facehide.models import sface_path, yunet_path


class EngineTests(unittest.TestCase):
    def test_blank_image_has_no_faces(self) -> None:
        if not yunet_path().exists() or not sface_path().exists():
            self.skipTest("models not downloaded")
        engine = FaceEngine()
        hits = engine.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertEqual(hits, [])


    def test_enroll_rejects_blank(self) -> None:
        if not yunet_path().exists() or not sface_path().exists():
            self.skipTest("models not downloaded")
        from facehide.engine import NoFaceError

        engine = FaceEngine()
        with self.assertRaises(NoFaceError):
            engine.enroll(np.zeros((480, 640, 3), dtype=np.uint8))