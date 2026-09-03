from __future__ import annotations

import sys
from collections.abc import Sequence

from PySide6.QtCore import QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QSystemTrayIcon,
)

from facehide import __version__
from facehide.config import SettingsStore
from facehide.engine import FaceEngine
from facehide.gallery import Gallery
from facehide.i18n import set_language, t
from facehide.instance import SingleInstance
from facehide.startup import sync_startup
from facehide.models import ModelError, ensure_models
from facehide.threads import apply as apply_threads
from facehide.ui.icons import COLOR_MUTED, app_icon, glyph_icon
from facehide.ui.main_window import MainWindow
from facehide.ui.styles import APP_QSS


def run_self_check() -> int:
    import time
    import urllib.request

    import cv2
    import numpy as np

    from facehide.camera import probe_cameras
    from facehide.engine import FaceEngine, NoFaceError
    from facehide.gallery import cosine_similarity
    from facehide.infer.device import coinitialize_mta, couninitialize, dml_available, ort_available
    from facehide.models import ensure_models
    from facehide.threads import clamped_facehide_threads, intra_op

    apply_threads(dml_active=False)
    print(f"FaceHide {__version__}")
    print(
        f"线程     OpenCV {cv2.getNumThreads()}  ORT intra {intra_op()}  "
        f"FACEHIDE_THREADS={clamped_facehide_threads()}"
    )
    print("下载/校验模型…")
    det, rec = ensure_models()
    print(f"YuNet  {det} ({det.stat().st_size} bytes)")
    print(f"SFace  {rec} ({rec.stat().st_size} bytes)")
    com_ok = coinitialize_mta()
    try:
        engine = FaceEngine(det, rec, device="auto")
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        hits = engine.detect(blank)
        first_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        hits = engine.detect(blank)
        steady_ms = (time.perf_counter() - t0) * 1000
        print(f"空画面检测 {len(hits)} 张脸")
        if hits:
            raise SystemExit("空画面不应检出人脸")
        try:
            engine.enroll(blank)
            raise SystemExit("空画面不应登记成功")
        except NoFaceError:
            print("空画面登记被拒绝")
        info = engine.backend_info()
        dxgi = "none" if info.dxgi_index is None else str(info.dxgi_index)
        dedicated = "n/a" if info.dedicated_bytes is None else str(info.dedicated_bytes)
        print(
            f"推理设备  auto → {info.provider} / {info.device_name}  "
            f"dxgi={dxgi}  dedicated={dedicated}"
        )
        print(
            f"YuNet    {info.detector}  first {first_ms:.1f} ms  "
            f"steady {steady_ms:.1f} ms  faces={len(hits)}"
        )
        if not ort_available():
            print("ORT 未安装")
        elif not dml_available():
            print("DirectML 不可用，使用 CPU")

        cpu = FaceEngine(det, rec, device="cpu")
        cpu_blank = cpu.detect(blank)
        if cpu_blank:
            raise SystemExit("CPU 空画面不应检出人脸")

        sample_url = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg"
        print("下载样本图并自匹配…")
        image = None
        try:
            req = urllib.request.Request(sample_url, headers={"User-Agent": "FaceHide/0.1"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = np.frombuffer(resp.read(), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception as exc:
            print(f"样本图下载失败，跳过自匹配: {exc}")
        if image is None:
            cameras = probe_cameras()
            print("摄像头:", cameras if cameras else "未发现")
            print("OK")
            return 0
        feature, _thumb = engine.enroll(image)
        again = engine.detect(image)
        if not again or again[0].feature is None:
            raise SystemExit("样本图二次检测失败")
        score = cosine_similarity(feature, again[0].feature)
        print(f"样本自匹配 {score:.3f}")
        if score < 0.6:
            raise SystemExit(f"自匹配过低: {score}")
        if ort_available():
            cpu_feat, _ = cpu.enroll(image)
            cpu_again = cpu.detect(image)
            if not cpu_again or cpu_again[0].feature is None:
                raise SystemExit("CPU 样本图二次检测失败")
            opencv_score = cosine_similarity(cpu_feat, cpu_again[0].feature)
            cross = cosine_similarity(cpu_feat, again[0].feature)
            print(f"自匹配   OpenCV {opencv_score:.3f}  live {score:.3f}  cross {cross:.3f}")
            if opencv_score < 0.90 or cross < 0.90:
                raise SystemExit(f"ORT 交叉匹配过低: opencv={opencv_score:.3f} cross={cross:.3f}")

        cameras = probe_cameras()
        print("摄像头:", cameras if cameras else "未发现")
        print("OK")
        return 0
    finally:
        if com_ok:
            couninitialize()


def _install_tray(app: QApplication, window: MainWindow) -> QSystemTrayIcon:
    tray = QSystemTrayIcon(app_icon(), app)
    tray.setToolTip(t("app.name"))
    menu = QMenu(window)
    menu.setObjectName("TrayMenu")
    show = QAction(glyph_icon("window", 16, COLOR_MUTED), t("tray.open"), window)
    show.setMenuRole(QAction.MenuRole.NoRole)
    show.triggered.connect(window.reveal)
    toggle = QAction(glyph_icon("monitor", 16, COLOR_MUTED), t("tray.toggle"), window)
    toggle.setMenuRole(QAction.MenuRole.NoRole)
    toggle.triggered.connect(window.toggle_monitor)
    quit_action = QAction(glyph_icon("power", 16, COLOR_MUTED), t("tray.quit"), window)
    quit_action.setMenuRole(QAction.MenuRole.NoRole)
    quit_action.triggered.connect(window.request_quit)
    menu.addAction(show)
    menu.addAction(toggle)
    menu.addSeparator()
    menu.addAction(quit_action)
    tray.setContextMenu(menu)
    tray.activated.connect(
        lambda reason: window.reveal()
        if reason == QSystemTrayIcon.ActivationReason.Trigger
        else None
    )
    tray.show()
    window.bind_tray(tray, show, toggle, quit_action)
    return tray


def _download_models(app: QApplication) -> bool:
    dialog = QProgressDialog(t("models.preparing"), None, 0, 100, None)
    dialog.setWindowTitle(t("app.name"))
    dialog.setWindowIcon(app_icon())
    dialog.setCancelButton(None)
    dialog.setMinimumDuration(0)
    dialog.setValue(0)
    dialog.show()
    app.processEvents()

    def progress(label: str, done: int, total: int) -> None:
        dialog.setLabelText(label)
        if total > 0:
            dialog.setMaximum(total)
            dialog.setValue(done)
        app.processEvents()

    try:
        ensure_models(progress=progress)
        dialog.close()
        return True
    except ModelError as exc:
        dialog.close()
        QMessageBox.critical(None, t("app.name"), t("models.fail", error=exc))
        return False


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv if argv is None else argv)
    apply_threads(dml_active=False)
    if "--check" in args:
        return run_self_check()

    store = SettingsStore()
    if "--dev" in args:
        settings = store.get()
        settings.dev_mode = True
        store.replace(settings)
    set_language(store.get().language)
    try:
        if store.get().start_on_boot:
            sync_startup(True)
    except OSError:
        pass

    app = QApplication(args)
    app.setApplicationName(t("app.name"))
    app.setApplicationVersion(__version__)
    app.setWindowIcon(app_icon())
    app.setQuitOnLastWindowClosed(False)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_QSS)

    holder: dict[str, MainWindow | None] = {"window": None}

    def activate() -> None:
        window = holder["window"]
        if window is not None:
            window.reveal()

    guard = SingleInstance(app, on_activate=activate)
    if not guard.acquire():
        return 0

    if not _download_models(app):
        guard.close()
        return 1

    gallery = Gallery()
    engine = FaceEngine(device=store.get().inference_device)
    window = MainWindow(store, gallery, engine)
    holder["window"] = window
    tray = _install_tray(app, window)
    settings = store.get()
    start_hidden = ("--minimized" in args or settings.start_minimized) and not settings.dev_mode
    if start_hidden:
        window.hide()
        QTimer.singleShot(
            400,
            lambda: tray.showMessage(
                t("app.name"),
                t("tray.started_hidden"),
                QSystemTrayIcon.MessageIcon.Information,
                2500,
            ),
        )
    else:
        window.show()
    if settings.auto_start_monitor and not start_hidden:
        QTimer.singleShot(
            600,
            lambda: tray.showMessage(
                t("app.name"),
                t("tray.autostart"),
                QSystemTrayIcon.MessageIcon.Information,
                2500,
            ),
        )
    code = app.exec()
    guard.close()
    return code
