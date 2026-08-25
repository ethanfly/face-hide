import tempfile
import unittest
from pathlib import Path

import numpy as np

from facehide.gallery import (
    Gallery,
    best_match,
    can_trigger,
    cluster_indices,
    cosine_similarity,
    decide_link,
    rank_people,
)


def _thumb() -> np.ndarray:
    return np.full((40, 40, 3), 80, dtype=np.uint8)


class GalleryTests(unittest.TestCase):
    def test_cosine_identical(self) -> None:
        vec = np.array([0.2, 0.4, 0.8], dtype=np.float32)
        self.assertAlmostEqual(cosine_similarity(vec, vec), 1.0, places=5)

    def test_best_match_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gallery = Gallery(root / "gallery.json", root)
            person = gallery.add_person("老板", np.array([1.0, 0.0, 0.0], dtype=np.float32), _thumb())
            hit = best_match(np.array([0.99, 0.01, 0.0], dtype=np.float32), gallery.people(), 0.8)
            miss = best_match(np.array([0.0, 1.0, 0.0], dtype=np.float32), gallery.people(), 0.8)
            self.assertIsNotNone(hit)
            assert hit is not None
            self.assertEqual(hit.person.id, person.id)
            self.assertIsNone(miss)

    def test_rank_people_orders_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gallery = Gallery(root / "gallery.json", root)
            far = gallery.add_person("路人", np.array([0.0, 1.0, 0.0], dtype=np.float32), _thumb())
            near = gallery.add_person("同事", np.array([1.0, 0.0, 0.0], dtype=np.float32), _thumb())
            ranked = rank_people(np.array([0.95, 0.05, 0.0], dtype=np.float32), gallery.people())
            self.assertEqual([item.person.id for item in ranked], [near.id, far.id])
            self.assertGreater(ranked[0].score, ranked[1].score)

    def test_cluster_same_and_different(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.98, 0.02], dtype=np.float32)
        c = np.array([0.0, 1.0], dtype=np.float32)
        groups = cluster_indices([a, c, b], 0.8)
        grouped = {frozenset(group) for group in groups}
        self.assertIn(frozenset({0, 2}), grouped)
        self.assertIn(frozenset({1}), grouped)

    def test_decide_link_defaults_to_auto(self) -> None:
        self.assertEqual(decide_link(0.72, 0.40, True), "auto")
        self.assertEqual(decide_link(0.72, 0.40, False), "ask")
        self.assertEqual(decide_link(0.32, 0.40, True), "ask")
        self.assertEqual(decide_link(0.10, 0.40, True), "new")

    def test_merge_and_split_people(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gallery = Gallery(root / "gallery.json", root)
            keep = gallery.add_person("甲", np.ones(8, dtype=np.float32), _thumb())
            absorb = gallery.add_person("乙", np.zeros(8, dtype=np.float32), _thumb())
            gallery.merge_people(keep.id, absorb.id)
            people = gallery.people()
            self.assertEqual(len(people), 1)
            self.assertEqual(len(people[0].samples), 2)
            split = gallery.split_sample(keep.id, people[0].samples[1].id, "拆出")
            self.assertEqual(len(gallery.people()), 2)
            self.assertEqual(split.name, "拆出")
            self.assertEqual(len(gallery.person(keep.id).samples), 1)

    def test_persist_source_and_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gallery = Gallery(root / "gallery.json", root)
            person = gallery.add_person("同事", np.ones(8, dtype=np.float32), _thumb(), source="enroll")
            gallery.add_sample(person.id, np.zeros(8, dtype=np.float32), _thumb(), source="auto")
            again = Gallery(root / "gallery.json", root)
            loaded = again.people()[0]
            self.assertEqual([sample.source for sample in loaded.samples], ["enroll", "auto"])
            self.assertTrue(loaded.auto_linked)

    def test_persist_reload_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gallery = Gallery(root / "gallery.json", root)
            person = gallery.add_person("同事", np.ones(8, dtype=np.float32), _thumb())
            gallery.add_sample(person.id, np.zeros(8, dtype=np.float32), _thumb())
            again = Gallery(root / "gallery.json", root)
            self.assertEqual(len(again.people()), 1)
            self.assertEqual(len(again.people()[0].samples), 2)
            again.remove_person(person.id)
            self.assertEqual(again.people(), [])
            self.assertEqual(Gallery(root / "gallery.json", root).people(), [])

    def test_enabled_defaults_and_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gallery = Gallery(root / "gallery.json", root)
            person = gallery.add_person("同事", np.ones(8, dtype=np.float32), _thumb())
            self.assertTrue(person.enabled)
            feature = np.ones(8, dtype=np.float32)
            hit = best_match(feature, gallery.people(), 0.5)
            self.assertTrue(can_trigger(hit, 0.5))
            gallery.set_enabled(person.id, False)
            self.assertFalse(gallery.person(person.id).enabled)
            again = Gallery(root / "gallery.json", root)
            loaded = again.people()[0]
            self.assertFalse(loaded.enabled)
            hit = best_match(feature, again.people(), 0.5)
            self.assertFalse(can_trigger(hit, 0.5))
            again.add_sample(loaded.id, np.zeros(8, dtype=np.float32), _thumb())
            split = again.split_sample(loaded.id, again.people()[0].samples[1].id, "拆出")
            self.assertFalse(split.enabled)

    def test_blacklist_persists_and_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gallery = Gallery(root / "gallery.json", root)
            person = gallery.add_person("同事", np.ones(8, dtype=np.float32), _thumb())
            self.assertFalse(person.blacklisted)
            gallery.set_blacklisted(person.id, True)
            again = Gallery(root / "gallery.json", root)
            self.assertTrue(again.people()[0].blacklisted)
            other = gallery.add_person("路人", np.zeros(8, dtype=np.float32), _thumb())
            gallery.merge_people(other.id, person.id)
            self.assertTrue(gallery.person(other.id).blacklisted)

    def test_nickname_persists_and_merge_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gallery = Gallery(root / "gallery.json", root)
            person = gallery.add_person("同事", np.ones(8, dtype=np.float32), _thumb())
            self.assertEqual(person.nickname, "")
            gallery.set_nickname(person.id, "  老板  ")
            self.assertEqual(gallery.person(person.id).nickname, "老板")
            again = Gallery(root / "gallery.json", root)
            self.assertEqual(again.people()[0].nickname, "老板")
            other = gallery.add_person("路人", np.zeros(8, dtype=np.float32), _thumb())
            gallery.merge_people(other.id, person.id)
            self.assertEqual(gallery.person(other.id).nickname, "老板")
            gallery.add_sample(other.id, np.ones(8, dtype=np.float32), _thumb())
            split = gallery.split_sample(other.id, gallery.person(other.id).samples[1].id, "拆出")
            self.assertEqual(split.nickname, "")
            self.assertEqual(gallery.person(other.id).nickname, "老板")


if __name__ == "__main__":
    unittest.main()
