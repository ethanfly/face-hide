from __future__ import annotations

import sys
from collections.abc import Sequence

from PySide6.QtCore import QTimer
from PySide6.QtGui import QAction
from PySide6.QtNetwork import QLocalServer, QLocalSocket
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
from facehide.models import ModelError, ensure_models
from facehide.ui.icons import app_icon
from facehide.ui.main_window import MainWindow
from facehide.ui.styles import APP_QSS

INSTANCE_NAME = "FaceHide.SingleInstance"


def run_self_check() -> int:
    import urllib.request

    import cv2
    import numpy as np

    from facehide.camera import probe_cameras
    from facehide.engine import FaceEngine, NoFaceError
    from facehide.gallery import cosine_similarity
    from facehide.models import ensure_models

    print(f"FaceHide {__version__}")
    print("下载/校验模型…")
    det, rec = ensure_models()
    print(f"YuNet  {det} ({det.stat().st_size} bytes)")
    print(f"SFace  {rec} ({rec.stat().st_size} bytes)")
    engine = FaceEngine(det, rec)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    hits = engine.detect(blank)
    print(f"空画面检测 {len(hits)} 张脸")
    if hits:
        raise SystemExit("空画面不应检出人脸")
    try:
        engine.enroll(blank)
        raise SystemExit("空画面不应登记成功")
    except NoFaceError:
        print("空画面登记被拒绝")

    sample_url = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg"
    print("下载样本图并自匹配…")
    req = urllib.request.Request(sample_url, headers={"User-Agent": "FaceHide/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = np.frombuffer(resp.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit("样本图解码失败")
    feature, _thumb = engine.enroll(image)
    again = engine.detect(image)
    if not again or again[0].feature is None:
        raise SystemExit("样本图二次检测失败")
    score = cosine_similarity(feature, again[0].feature)
    print(f"样本自匹配 {score:.3f}")
    if score < 0.6:
        raise SystemExit(f"自匹配过低: {score}")

    cameras = probe_cameras()
    print("摄像头:", cameras if cameras else "未发现")
    print("OK")
    return 0


def _raise_existing() -> bool:
    socket = QLocalSocket()
    socket.connectToServer(INSTANCE_NAME)
    if not socket.waitForConnected(150):
        return False
    socket.write(b"raise")
    socket.flush()
    socket.waitForBytesWritten(150)
    return True


def _listen(app: QApplication, window: MainWindow) -> QLocalServer:
    QLocalServer.removeServer(INSTANCE_NAME)
    server = QLocalServer(app)
    server.listen(INSTANCE_NAME)

    def on_connect() -> None:
        incoming = server.nextPendingConnection()
        if incoming:
            incoming.readyRead.connect(lambda: incoming.readAll())
        window.show()
        window.raise_()
        window.activateWindow()

    server.newConnection.connect(on_connect)
    return server


def _install_tray(app: QApplication, window: MainWindow) -> QSystemTrayIcon:
    tray = QSystemTrayIcon(app_icon(), app)
    tray.setToolTip(t("app.name"))
    menu = QMenu()
    show = QAction(t("tray.open"))
    show.triggered.connect(window.showNormal)
    toggle = QAction(t("tray.toggle"))
    toggle.triggered.connect(window.toggle_monitor)
    quit_action = QAction(t("tray.quit"))
    quit_action.triggered.connect(window.request_quit)
    menu.addAction(show)
    menu.addAction(toggle)
    menu.addSeparator()
    menu.addAction(quit_action)
    tray.setContextMenu(menu)
    tray.activated.connect(
        lambda reason: window.showNormal()
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
    if "--check" in args:
        return run_self_check()

    store = SettingsStore()
    if "--dev" in args:
        settings = store.get()
        settings.dev_mode = True
        store.replace(settings)
    set_language(store.get().language)

    if _raise_existing():
        return 0

    app = QApplication(args)
    app.setApplicationName(t("app.name"))
    app.setApplicationVersion(__version__)
    app.setWindowIcon(app_icon())
    app.setQuitOnLastWindowClosed(False)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_QSS)

    if not _download_models(app):
        return 1

    gallery = Gallery()
    engine = FaceEngine()
    window = MainWindow(store, gallery, engine)
    _listen(app, window)
    tray = _install_tray(app, window)
    window.show()
    if store.get().auto_start_monitor:
        QTimer.singleShot(600, lambda: tray.showMessage(t("app.name"), t("tray.autostart"), QSystemTrayIcon.MessageIcon.Information, 2500))
    return app.exec()
