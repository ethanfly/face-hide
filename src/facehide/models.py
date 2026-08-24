from __future__ import annotations

import shutil
import urllib.request
from collections.abc import Callable
from pathlib import Path

from facehide.paths import models_dir
from facehide.runtime import bundle_dir, exe_dir

Progress = Callable[[str, int, int], None]

YUNET_NAME = "face_detection_yunet_2023mar.onnx"
SFACE_NAME = "face_recognition_sface_2021dec.onnx"

_MODELS: dict[str, dict] = {
    YUNET_NAME: {
        "min_bytes": 200_000,
        "urls": (
            "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        ),
    },
    SFACE_NAME: {
        "min_bytes": 1_000_000,
        "urls": (
            "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        ),
    },
}


class ModelError(RuntimeError):
    pass


def yunet_path(root: Path | None = None) -> Path:
    return (root or models_dir()) / YUNET_NAME


def sface_path(root: Path | None = None) -> Path:
    return (root or models_dir()) / SFACE_NAME


def _valid(path: Path, min_bytes: int) -> bool:
    return path.is_file() and path.stat().st_size >= min_bytes


def _download(url: str, dest: Path, label: str, progress: Progress | None) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "FaceHide/0.1"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("wb") as fh:
            while True:
                chunk = resp.read(1024 * 64)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if progress:
                    progress(label, done, total)
    tmp.replace(dest)


def bundled_models_dir() -> Path | None:
    for folder in (exe_dir() / "models", bundle_dir() / "models"):
        if folder.is_dir() and any(folder.glob("*.onnx")):
            return folder
    return None


def _seed_from_bundle(folder: Path) -> None:
    source = bundled_models_dir()
    if source is None:
        return
    for name in _MODELS:
        dest = folder / name
        src = source / name
        if dest.exists() or not src.is_file():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def ensure_models(root: Path | None = None, progress: Progress | None = None) -> tuple[Path, Path]:
    folder = root or models_dir()
    folder.mkdir(parents=True, exist_ok=True)
    _seed_from_bundle(folder)
    paths = []
    for name, spec in _MODELS.items():
        dest = folder / name
        if _valid(dest, spec["min_bytes"]):
            paths.append(dest)
            continue
        last_error: Exception | None = None
        for url in spec["urls"]:
            try:
                if progress:
                    progress(f"下载 {name}", 0, 0)
                _download(url, dest, f"下载 {name}", progress)
                if _valid(dest, spec["min_bytes"]):
                    last_error = None
                    break
                dest.unlink(missing_ok=True)
                last_error = ModelError(f"{name} 下载不完整")
            except Exception as exc:  # noqa: BLE001 — 多源回退
                last_error = exc
                dest.unlink(missing_ok=True)
                dest.with_suffix(dest.suffix + ".part").unlink(missing_ok=True)
        if last_error:
            raise ModelError(f"无法获取模型 {name}: {last_error}") from last_error
        paths.append(dest)
    return paths[0], paths[1]
