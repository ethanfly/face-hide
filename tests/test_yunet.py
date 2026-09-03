from __future__ import annotations

import unittest

import numpy as np

from facehide.infer.preprocess import fit_canvas, pad_bgr, pad_scale, unpad_rows
from facehide.infer.yunet import decode_heads, empty_yunet_outputs, outputs_to_dict, YUNET_OUTPUTS


class PadTests(unittest.TestCase):
    def test_never_upscale_320x240(self) -> None:
        self.assertEqual(pad_scale(320, 240), 1.0)
        src = np.full((240, 320, 3), 7, dtype=np.uint8)
        blob, scale = pad_bgr(src)
        self.assertEqual(scale, 1.0)
        self.assertEqual(blob.shape, (1, 3, 640, 640))
        canvas = np.transpose(blob[0], (1, 2, 0))
        self.assertTrue(np.all(canvas[:240, :320] == 7))
        self.assertTrue(np.all(canvas[240:, :] == 0))
        self.assertTrue(np.all(canvas[:, 320:] == 0))

    def test_640x480_bottom_pad(self) -> None:
        self.assertEqual(pad_scale(640, 480), 1.0)
        src = np.full((480, 640, 3), 9, dtype=np.uint8)
        blob, scale = pad_bgr(src)
        self.assertEqual(scale, 1.0)
        canvas = np.transpose(blob[0], (1, 2, 0))
        self.assertTrue(np.all(canvas[:480, :640] == 9))
        self.assertTrue(np.all(canvas[480:, :] == 0))
        canvas_u8, cscale = fit_canvas(src)
        self.assertEqual(cscale, 1.0)
        self.assertEqual(canvas_u8.shape, (640, 640, 3))
        self.assertTrue(np.all(canvas_u8[:480, :640] == 9))

    def test_1280x720_downscale(self) -> None:
        self.assertAlmostEqual(pad_scale(1280, 720), 0.5)
        src = np.full((720, 1280, 3), 11, dtype=np.uint8)
        blob, scale = pad_bgr(src)
        self.assertAlmostEqual(scale, 0.5)
        canvas = np.transpose(blob[0], (1, 2, 0))
        self.assertTrue(np.all(canvas[360:, :] == 0))

    def test_unpad_invert(self) -> None:
        for height, width in ((240, 320), (480, 640), (720, 1280)):
            scale = pad_scale(width, height)
            if (height, width) == (240, 320):
                self.assertEqual(scale, 1.0)
            row = np.array(
                [10, 20, 30, 40, 12, 22, 28, 22, 25, 30, 18, 36, 32, 36, 0.9],
                dtype=np.float32,
            ).reshape(1, 15)
            canvas = row.copy()
            canvas[:, 0:14] *= scale
            back = unpad_rows(canvas, scale)
            np.testing.assert_allclose(back, row, rtol=0, atol=1e-5)


class DecodeTests(unittest.TestCase):
    def test_single_cell_stride8(self) -> None:
        arrays = empty_yunet_outputs()
        mapped = outputs_to_dict(list(YUNET_OUTPUTS), arrays)
        idx = 5 * 80 + 10
        mapped["cls_8"][0, idx, 0] = 1.0
        mapped["obj_8"][0, idx, 0] = 1.0
        faces = decode_heads(mapped, score_threshold=0.5, nms_threshold=0.3, top_k=5000)
        self.assertEqual(len(faces), 1)
        x, y, w, h, score = faces[0, 0], faces[0, 1], faces[0, 2], faces[0, 3], faces[0, 14]
        self.assertAlmostEqual(float(w), 8.0, places=3)
        self.assertAlmostEqual(float(h), 8.0, places=3)
        self.assertAlmostEqual(float(x), 76.0, places=3)
        self.assertAlmostEqual(float(y), 36.0, places=3)
        self.assertAlmostEqual(float(score), 1.0, places=3)
        self.assertEqual(faces[0].shape, (15,))
