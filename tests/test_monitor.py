import unittest

from facehide.monitor import SeenFace, track_seen


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


if __name__ == "__main__":
    unittest.main()
