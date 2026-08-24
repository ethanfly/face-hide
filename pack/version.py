from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "VERSION"
PYPROJECT = ROOT / "pyproject.toml"
INIT = ROOT / "src" / "facehide" / "__init__.py"

_SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def parse_version(text: str) -> tuple[int, int, int]:
    match = _SEMVER.fullmatch(text.strip())
    if not match:
        raise ValueError(f"invalid version: {text!r}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def format_version(parts: tuple[int, int, int]) -> str:
    return f"{parts[0]}.{parts[1]}.{parts[2]}"


def read_version(root: Path | None = None) -> str:
    version_file = (root / "VERSION") if root else VERSION_FILE
    if version_file.is_file():
        return version_file.read_text(encoding="utf-8").strip()
    init = (root / "src" / "facehide" / "__init__.py") if root else INIT
    match = re.search(r'__version__\s*=\s*"([^"]+)"', init.read_text(encoding="utf-8"))
    if match:
        return match.group(1)
    return "0.1.0"


def bump_version(current: str, kind: str) -> str:
    major, minor, patch = parse_version(current)
    if kind == "major":
        return format_version((major + 1, 0, 0))
    if kind == "minor":
        return format_version((major, minor + 1, 0))
    if kind == "patch":
        return format_version((major, minor, patch + 1))
    if kind == "none":
        return format_version((major, minor, patch))
    raise ValueError(f"unknown bump kind: {kind}")


def write_version(version: str, root: Path | None = None) -> str:
    parse_version(version)
    base = root or ROOT
    version_file = base / "VERSION"
    pyproject = base / "pyproject.toml"
    init = base / "src" / "facehide" / "__init__.py"
    version_file.write_text(version + "\n", encoding="utf-8")
    py_text = pyproject.read_text(encoding="utf-8")
    updated, count = re.subn(r'(?m)^version\s*=\s*"[^"]+"', f'version = "{version}"', py_text, count=1)
    if count != 1:
        raise ValueError("pyproject.toml version field not found")
    pyproject.write_text(updated, encoding="utf-8")
    init_text = init.read_text(encoding="utf-8")
    updated_init, count = re.subn(r'(?m)^__version__\s*=\s*"[^"]+"', f'__version__ = "{version}"', init_text, count=1)
    if count != 1:
        raise ValueError("__init__.py __version__ not found")
    init.write_text(updated_init, encoding="utf-8")
    return version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Maintain FaceHide version files")
    parser.add_argument("action", choices=("show", "set", "bump"), help="show, set, or bump the version")
    parser.add_argument("value", nargs="?", help="version for set, or bump kind")
    parser.add_argument("--kind", choices=("major", "minor", "patch", "none"), default="patch")
    args = parser.parse_args(argv)
    current = read_version()
    if args.action == "show":
        print(current)
        return 0
    if args.action == "set":
        if not args.value:
            raise SystemExit("set requires a version, e.g. 1.2.3")
        print(write_version(args.value))
        return 0
    kind = args.value or args.kind
    print(write_version(bump_version(current, kind)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
