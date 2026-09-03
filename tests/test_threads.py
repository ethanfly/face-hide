from __future__ import annotations

import inspect
import os
import unittest
from pathlib import Path

from facehide.threads import apply, apply_env, clamped_facehide_threads


class ThreadCapTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {
            key: os.environ.get(key)
            for key in (
                "FACEHIDE_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        }

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        apply(dml_active=False)

    def test_apply_env_overwrites_existing_omp(self) -> None:
        os.environ["OMP_NUM_THREADS"] = "24"
        os.environ.pop("FACEHIDE_THREADS", None)
        n = apply_env()
        self.assertEqual(n, 4)
        self.assertEqual(os.environ["OMP_NUM_THREADS"], "4")
        self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "4")
        self.assertEqual(os.environ["MKL_NUM_THREADS"], "4")
        self.assertEqual(os.environ["NUMEXPR_NUM_THREADS"], "4")
        source = inspect.getsource(apply_env)
        self.assertNotIn("setdefault", source)
        self.assertIn('os.environ[key] = str(n)', source)

    def test_facehide_threads_clamps(self) -> None:
        os.environ["FACEHIDE_THREADS"] = "2"
        self.assertEqual(clamped_facehide_threads(), 2)
        os.environ["FACEHIDE_THREADS"] = "99"
        self.assertEqual(clamped_facehide_threads(), 8)
        os.environ["FACEHIDE_THREADS"] = "0"
        self.assertEqual(clamped_facehide_threads(), 1)

    def test_apply_sets_opencv_threads(self) -> None:
        import cv2

        os.environ.pop("FACEHIDE_THREADS", None)
        apply_env()
        apply(dml_active=False)
        self.assertEqual(cv2.getNumThreads(), 4)
        apply(dml_active=True)
        self.assertEqual(cv2.getNumThreads(), 1)

    def test_main_and_entry_pin_before_app_import(self) -> None:
        root = Path(__file__).resolve().parents[1]
        main_src = (root / "src" / "facehide" / "__main__.py").read_text(encoding="utf-8")
        entry_src = (root / "pack" / "entry.py").read_text(encoding="utf-8")
        self.assertLess(main_src.find("apply_env"), main_src.find("facehide.ui.app"))
        self.assertLess(entry_src.find("apply_env"), entry_src.find("facehide.ui.app"))
        self.assertNotIn("setdefault", inspect.getsource(apply_env))
