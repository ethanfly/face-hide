from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DOCS = ROOT / "docs"
SHOTS = DOCS / "screenshots"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("QT_SCALE_FACTOR", "1")
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "0")


def _qpixmap_from_pil(image):
    from PySide6.QtGui import QImage, QPixmap

    rgba = image.convert("RGBA")
    qimage = QImage(
        rgba.tobytes("raw", "RGBA"),
        rgba.width,
        rgba.height,
        QImage.Format.Format_RGBA8888,
    )
    return QPixmap.fromImage(qimage.copy())


def write_icon() -> None:
    from facehide.mark import render_mark

    DOCS.mkdir(parents=True, exist_ok=True)
    render_mark(256).save(DOCS / "icon.png")
    print("icon", DOCS / "icon.png")


def _placeholder_preview(width: int, height: int):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QFont, QPainter, QPixmap

    from facehide.mark import render_mark

    pix = QPixmap(width, height)
    pix.fill(QColor("#080c12"))
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    mark = _qpixmap_from_pil(render_mark(96))
    x = (width - mark.width()) // 2
    y = (height - mark.height()) // 2 - 24
    painter.drawPixmap(x, y, mark)
    painter.setPen(QColor("#8b95a8"))
    painter.setFont(QFont("Microsoft YaHei UI", 12))
    painter.drawText(
        0,
        y + mark.height() + 8,
        width,
        28,
        Qt.AlignmentFlag.AlignHCenter,
        "摄像头预览",
    )
    painter.end()
    return pix


def capture() -> None:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    import facehide.ui.main_window as mw
    from facehide.config import SettingsStore, WorkApp
    from facehide.engine import FaceEngine
    from facehide.gallery import Gallery
    from facehide.i18n import set_language, t
    from facehide.ui.main_window import MainWindow
    from facehide.ui.styles import APP_QSS

    mw.list_open_apps = lambda exclude_pids=None: []
    mw.describe_cameras = lambda: []
    MainWindow._boot_cameras = lambda self: None

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_QSS)
    set_language("zh")

    tmp = Path(tempfile.mkdtemp(prefix="facehide-shots-"))
    store = SettingsStore(tmp / "config.json")
    settings = store.get()
    settings.work_apps = [
        WorkApp(id="edge", name="Microsoft Edge", path=r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        WorkApp(id="code", name="Visual Studio Code", path=r"C:\Users\user\AppData\Local\Programs\Microsoft VS Code\Code.exe"),
    ]
    settings.entertainment_processes = ["steam.exe", "game.exe"]
    settings.hide_foreground = True
    settings.break_fullscreen = True
    store.replace(settings)
    gallery = Gallery(tmp / "gallery.json", tmp / "gallery")
    window = MainWindow(store, gallery, FaceEngine())
    window.setWindowTitle(t("app.name"))
    window.resize(1180, 760)
    window.show()
    SHOTS.mkdir(parents=True, exist_ok=True)

    pages = (
        ("monitor", 0),
        ("faces", 1),
        ("work", 2),
        ("hide", 3),
        ("settings", 4),
    )

    def shoot() -> None:
        preview = window.preview.size()
        window.preview.setPixmap(_placeholder_preview(max(640, preview.width()), max(360, preview.height())))
        window.pill_fps.setText(t("pill.fps", fps=28))
        window.pill_match.setText(t("pill.seen", name="示例"))
        window.pill_match.setProperty("state", "on")
        window.pill_match.style().unpolish(window.pill_match)
        window.pill_match.style().polish(window.pill_match)
        window.pill_faces.setText(t("pill.faces", count=1))
        window.log.clear()
        window._log(t("log.monitor_start"))
        window._log(t("log.cam_opened"))
        window._log(t("log.seen", name="示例", score=0.86))
        window._armed = True
        window.side_state.setText(t("status.watching"))
        window.side_state.setProperty("state", "on")
        window.side_state.style().unpolish(window.side_state)
        window.side_state.style().polish(window.side_state)
        window._sync_monitor_button()
        for name, index in pages:
            for i, btn in enumerate(window.nav_buttons):
                btn.setChecked(i == index)
            window._refresh_nav_icons()
            window.stack.setCurrentIndex(index)
            app.processEvents()
            grabbed = window.grab()
            dest = SHOTS / f"{name}.png"
            grabbed.save(str(dest), "PNG")
            print("shot", dest, grabbed.width(), grabbed.height())
        window.close()
        app.quit()

    QTimer.singleShot(80, shoot)
    app.exec()


def main() -> int:
    write_icon()
    capture()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
