import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pack.version import bump_version, parse_version, read_version, write_version


class VersionTests(unittest.TestCase):
    def test_parse_and_bump(self) -> None:
        self.assertEqual(parse_version("0.1.0"), (0, 1, 0))
        self.assertEqual(bump_version("0.1.0", "patch"), "0.1.1")
        self.assertEqual(bump_version("0.1.9", "minor"), "0.2.0")
        self.assertEqual(bump_version("1.4.2", "major"), "2.0.0")
        self.assertEqual(bump_version("1.4.2", "none"), "1.4.2")

    def test_write_syncs_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src" / "facehide").mkdir(parents=True)
            (root / "VERSION").write_text("0.1.0\n", encoding="utf-8")
            (root / "pyproject.toml").write_text('[project]\nversion = "0.1.0"\n', encoding="utf-8")
            (root / "src" / "facehide" / "__init__.py").write_text('__version__ = "0.1.0"\n', encoding="utf-8")
            write_version("1.2.3", root)
            self.assertEqual(read_version(root), "1.2.3")
            self.assertIn('version = "1.2.3"', (root / "pyproject.toml").read_text(encoding="utf-8"))
            self.assertIn('__version__ = "1.2.3"', (root / "src" / "facehide" / "__init__.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
