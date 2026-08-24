from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK = Path(__file__).resolve().parent
SRC = ROOT / "src"
DIST = ROOT / "dist" / "FaceHide"
ICON = PACK / "FaceHide.ico"
ENTRY = PACK / "entry.py"
SPEC = PACK / "facehide.spec"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pack.version import read_version  # noqa: E402

MODEL_NAMES = (
    "face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx",
)


def _run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.check_call(args, cwd=str(ROOT))


def ensure_pyinstaller() -> None:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        _run([sys.executable, "-m", "pip", "install", "pyinstaller>=6.11"])


def write_icon() -> Path:
    from PIL import Image, ImageDraw

    masters: list[Image.Image] = []
    for size in (16, 24, 32, 48, 64, 128, 256):
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        radius = max(2, int(size * 0.22))
        draw.rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=(47, 111, 237, 255))
        draw.rounded_rectangle(
            (int(size * 0.18), int(size * 0.20), int(size * 0.82), int(size * 0.62)),
            radius=max(1, size // 16),
            fill=(16, 19, 26, 255),
        )
        draw.ellipse(
            (int(size * 0.36), int(size * 0.28), int(size * 0.64), int(size * 0.56)),
            fill=(126, 182, 255, 255),
        )
        draw.ellipse(
            (int(size * 0.44), int(size * 0.36), int(size * 0.56), int(size * 0.48)),
            fill=(16, 19, 26, 255),
        )
        draw.polygon(
            [
                (int(size * 0.22), int(size * 0.78)),
                (int(size * 0.32), int(size * 0.70)),
                (int(size * 0.80), int(size * 0.22)),
                (int(size * 0.70), int(size * 0.30)),
            ],
            fill=(255, 139, 123, 255),
        )
        masters.append(img)
    ICON.parent.mkdir(parents=True, exist_ok=True)
    masters[-1].save(ICON, format="ICO", sizes=[(image.width, image.height) for image in masters])
    return ICON


def local_models() -> list[Path]:
    found: list[Path] = []
    seen: set[str] = set()
    folders = [
        ROOT / "models",
        Path(os.environ.get("LOCALAPPDATA", "")) / "FaceHide" / "models",
    ]
    for folder in folders:
        for name in MODEL_NAMES:
            path = folder / name
            if path.is_file() and name not in seen:
                found.append(path)
                seen.add(name)
    return found


def prepare_models() -> list[Path]:
    found = local_models()
    if len(found) >= len(MODEL_NAMES):
        return found
    from facehide.models import ensure_models

    dest = ROOT / "models"
    dest.mkdir(parents=True, exist_ok=True)
    print("downloading face models into", dest, flush=True)
    ensure_models(root=dest)
    return local_models()


def write_spec(models: list[Path]) -> Path:
    check_png = SRC / "facehide" / "ui" / "check.png"
    data_items = [(str(check_png), "facehide/ui")]
    data_items.extend((str(path), "models") for path in models)
    spec = f"""# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

SKIP = (
    "WebEngine", "Qt3D", "Bluetooth", "Multimedia", "Qml", "Quick", "Pdf",
    "Charts", "DataVisualization", "Graphs", "RemoteObjects", "Sensors",
    "SerialPort", "Nfc", "Positioning", "Location", "HttpServer", "WebSockets",
    "WebChannel", "WebView", "SpatialAudio", "TextToSpeech", "Designer",
    "Help", "Example", "VirtualKeyboard",
)

def keep(item):
    text = str(item[0] if isinstance(item, (tuple, list)) else item)
    return not any(token in text for token in SKIP)

datas, binaries, hiddenimports = [], [], []
for package in ("PySide6", "cv2", "numpy"):
    collected_datas, collected_binaries, collected_hidden = collect_all(package)
    datas += [item for item in collected_datas if keep(item)]
    binaries += [item for item in collected_binaries if keep(item)]
    hiddenimports += [item for item in collected_hidden if keep(item)]
try:
    binaries += collect_dynamic_libs("pywin32")
except Exception:
    pass
datas += {data_items!r}

hiddenimports += [
    "win32api",
    "win32con",
    "win32gui",
    "win32process",
    "pythoncom",
    "pywintypes",
    "psutil",
    "PIL",
    "facehide",
    "facehide.ui.app",
    "facehide.ui.main_window",
    "facehide.ui.styles",
    "facehide.ui.icons",
]

a = Analysis(
    [{str(ENTRY)!r}],
    pathex=[{str(SRC)!r}],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "IPython", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FaceHide",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon={str(ICON)!r},
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="FaceHide",
)
"""
    SPEC.write_text(spec, encoding="utf-8")
    return SPEC


def copy_models(dist: Path, models: list[Path]) -> None:
    dest = dist / "models"
    dest.mkdir(parents=True, exist_ok=True)
    for path in models:
        target = dest / path.name
        if not target.exists():
            shutil.copy2(path, target)
        print("model", target, target.stat().st_size)


def write_readme(dist: Path, version: str) -> None:
    text = f"""当面隐藏 / FaceHide {version}

双击 FaceHide.exe 启动。

首次启动会准备人脸模型（已随包附带则无需联网）。
人脸和设置只保存在本机 %LOCALAPPDATA%\\FaceHide。

可选参数：
  FaceHide.exe --dev     开发模式，命中只演练
  FaceHide.exe --check   自检摄像头和模型

请把整个 FaceHide 文件夹一起拷贝，不要只拷 exe。
"""
    (dist / "使用说明.txt").write_text(text, encoding="utf-8")
    (dist / "VERSION").write_text(version + "\n", encoding="utf-8")


def zip_dist(version: str) -> Path:
    out = ROOT / "dist" / f"FaceHide-{version}-win64"
    archive = shutil.make_archive(str(out), "zip", root_dir=str(ROOT / "dist"), base_dir="FaceHide")
    path = Path(archive)
    print("zip", path, path.stat().st_size)
    return path


def main() -> int:
    version = read_version()
    print("FaceHide", version, flush=True)
    models = prepare_models()
    if len(models) < len(MODEL_NAMES):
        raise SystemExit("face models missing; pack cannot continue")
    ensure_pyinstaller()
    write_icon()
    write_spec(models)
    _run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", str(SPEC)])
    if not DIST.is_dir():
        raise SystemExit(f"未生成目录：{DIST}")
    copy_models(DIST, models)
    write_readme(DIST, version)
    exe = DIST / "FaceHide.exe"
    print("OK", exe, "size", exe.stat().st_size)
    zip_dist(version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
