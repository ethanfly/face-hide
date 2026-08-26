import tempfile
import unittest
from pathlib import Path

import numpy as np

from facehide.engine import FaceHit
from facehide.gallery import Gallery
from facehide.i18n import set_language
from facehide.monitor import SeenFace, enroll_unknown_faces, track_seen


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


if __name__ == "__main__":
    unittest.main()
