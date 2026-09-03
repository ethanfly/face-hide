from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK = Path(__file__).resolve().parent
SRC = ROOT / "src"
DIST = ROOT / "dist" / "FaceHide"
ICON = PACK / "FaceHide.ico"
ENTRY = PACK / "entry.py"
SPEC = PACK / "facehide.spec"
ISS = PACK / "facehide.iss"
INNO_CACHE = PACK / "_inno"
INNO_URL = "https://github.com/jrsoftware/issrc/releases/download/is-6_7_3/innosetup-6.7.3.exe"

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


def ensure_directml() -> None:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit("packed GPU build requires onnxruntime-directml") from exc
    providers = ort.get_available_providers()
    if "DmlExecutionProvider" not in providers:
        raise SystemExit(f"DmlExecutionProvider missing: {providers}")


def ensure_pyinstaller() -> None:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        _run([sys.executable, "-m", "pip", "install", "pyinstaller>=6.11"])


def write_icon() -> Path:
    from facehide.mark import save_ico

    ICON.parent.mkdir(parents=True, exist_ok=True)
    return save_ico(ICON)


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
from pathlib import Path
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
    import onnxruntime
    capi = Path(onnxruntime.__file__).resolve().parent / "capi"
    for item in capi.glob("*"):
        if item.suffix.lower() in {{".dll", ".pyd"}}:
            binaries.append((str(item), "onnxruntime/capi"))
except Exception:
    pass
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
    "facehide.mark",
    "facehide.logbook",
    "facehide.notify",
    "facehide.startup",
    "facehide.ui.channels",
    "facehide.threads",
    "facehide.infer",
    "facehide.infer.session",
    "facehide.infer.device",
    "facehide.infer.ort_backend",
    "onnxruntime",
    "onnxruntime.capi",
    "openpyxl",
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
    excludes=["tkinter", "matplotlib", "IPython", "pytest", "torch", "torchvision", "torchaudio"],
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

关闭或最小化会缩到托盘，托盘图标右键可退出。同时只运行一个实例。
识别到已登记人脸时会写入窗口日志。
识别设置可选择自动 / GPU / CPU。有独显时默认用 DirectML 加速。

可选参数：
  FaceHide.exe --dev        开发模式，命中只演练
  FaceHide.exe --minimized  启动后最小化到托盘
  FaceHide.exe --check      自检摄像头和模型

请把整个 FaceHide 文件夹一起拷贝，不要只拷 exe。
"""
    (dist / "使用说明.txt").write_text(text, encoding="utf-8")
    (dist / "VERSION").write_text(version + "\n", encoding="utf-8")


def find_iscc() -> Path | None:
    which = shutil.which("ISCC") or shutil.which("iscc")
    if which:
        return Path(which)
    roots = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Inno Setup 6",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Inno Setup 6",
        INNO_CACHE / "app",
        Path(r"C:\Program Files (x86)\Inno Setup 6"),
        Path(r"C:\Program Files\Inno Setup 6"),
        Path(r"C:\Program Files (x86)\Inno Setup 7"),
        Path(r"C:\Program Files\Inno Setup 7"),
    ]
    for root in roots:
        candidate = root / "ISCC.exe"
        if candidate.is_file():
            return candidate
    return None


def ensure_inno() -> Path:
    existing = find_iscc()
    if existing is not None:
        return existing
    INNO_CACHE.mkdir(parents=True, exist_ok=True)
    installer = INNO_CACHE / "innosetup.exe"
    if not installer.is_file():
        print("downloading Inno Setup", INNO_URL, flush=True)
        urllib.request.urlretrieve(INNO_URL, installer)
    dest = INNO_CACHE / "app"
    dest.mkdir(parents=True, exist_ok=True)
    print("installing Inno Setup into", dest, flush=True)
    subprocess.check_call(
        [
            str(installer),
            "/VERYSILENT",
            "/SUPPRESSMSGBOXES",
            "/NORESTART",
            "/SP-",
            "/CURRENTUSER",
            f"/DIR={dest}",
        ]
    )
    iscc = dest / "ISCC.exe"
    if not iscc.is_file():
        found = find_iscc()
        if found is None:
            raise SystemExit("Inno Setup ISCC.exe not found after install")
        return found
    return iscc


def _iss_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def write_installer(version: str) -> Path:
    iscc = ensure_inno()
    out_dir = ROOT / "dist"
    out_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            str(iscc),
            f"/DMyAppVersion={version}",
            f"/DDistDir={_iss_path(DIST)}",
            f"/DIconFile={_iss_path(ICON)}",
            f"/DOutDir={_iss_path(out_dir)}",
            str(ISS),
        ]
    )
    setup = out_dir / f"FaceHide-{version}-win64-setup.exe"
    if not setup.is_file():
        raise SystemExit(f"未生成安装程序：{setup}")
    print("setup", setup, setup.stat().st_size)
    return setup


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
    ensure_directml()
    ensure_pyinstaller()
    write_icon()
    write_spec(models)
    _run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", str(SPEC)])
    if not DIST.is_dir():
        raise SystemExit(f"未生成目录：{DIST}")
    internal = DIST / "_internal"
    dml_ok = internal.is_dir() and any(internal.rglob("DirectML.dll"))
    if not dml_ok:
        raise SystemExit("packed build missing DirectML.dll")
    copy_models(DIST, models)
    write_readme(DIST, version)
    exe = DIST / "FaceHide.exe"
    print("OK", exe, "size", exe.stat().st_size)
    zip_dist(version)
    write_installer(version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
