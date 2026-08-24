import unittest
import uuid

from PySide6.QtWidgets import QApplication

from facehide.instance import SingleInstance, notify_existing


def _app() -> QApplication:
    existing = QApplication.instance()
    if existing is not None:
        return existing
    return QApplication([])


class SingleInstanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _app()

    def test_second_instance_notifies_first(self) -> None:
        name = f"FaceHide.Test.{uuid.uuid4().hex}"
        hits: list[int] = []
        first = SingleInstance(self.app, name=name, on_activate=lambda: hits.append(1))
        self.addCleanup(first.close)
        self.assertTrue(first.acquire())
        self.assertTrue(first.owned)
        self.assertTrue(notify_existing(name))
        for _ in range(20):
            self.app.processEvents()
            if hits:
                break
        self.assertEqual(hits, [1])
        second = SingleInstance(self.app, name=name)
        self.addCleanup(second.close)
        self.assertFalse(second.acquire())
        self.assertFalse(second.owned)


if __name__ == "__main__":
    unittest.main()
