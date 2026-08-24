from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QObject, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QCloseEvent, QColor, QCursor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QStackedWidget,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from facehide.actions import (
    OpenApp,
    common_work_apps,
    list_open_apps,
    perform_switch,
    running_process_names,
)
from facehide.camera import CameraInfo, describe_cameras, pick_camera
from facehide.config import MessageChannel, Settings, SettingsStore, WorkApp
from facehide.engine import FaceEngine, NoFaceError
from facehide.gallery import (
    Gallery,
    MatchResult,
    Person,
    Sample,
    decide_link,
    new_id,
    rank_people,
)
from facehide.i18n import current_language, set_language, t
from facehide.logbook import LogRecord, format_log_line, write_xlsx
from facehide.monitor import MonitorThread, PreviewFrame, SeenFace, TriggerEvent, track_seen
from facehide.notify import NotifyEvent, dispatch
from facehide.startup import sync_startup
from facehide.ui.icons import (
    COLOR_ACTIVE,
    COLOR_EMPTY,
    COLOR_MUTED,
    COLOR_ON_PRIMARY,
    app_icon,
    app_pixmap,
    glyph_icon,
    glyph_pixmap,
    tray_status,
)
from facehide.ui.channels import ChannelDialog, channel_summary
from facehide.ui.styles import APP_QSS, apply_dark_surface

NAV_ITEMS = (
    ("nav.monitor", "monitor"),
    ("nav.faces", "faces"),
    ("nav.work", "work"),
    ("nav.hide", "hide"),
    ("nav.notify", "notify"),
    ("nav.settings", "settings"),
)
NAV_KEYS = tuple(key for key, _glyph in NAV_ITEMS)
SAMPLE_SOURCE_KEYS = {"auto": "sample.auto", "manual": "sample.manual", "enroll": "sample.enroll"}


class _NotifyWorker(QObject):
    done = Signal(str)

    def start(self, event: NotifyEvent, channels: list[MessageChannel]) -> None:
        threading.Thread(target=self._run, args=(event, list(channels)), daemon=True).start()

    def _run(self, event: NotifyEvent, channels: list[MessageChannel]) -> None:
        for line in dispatch(channels, event):
            self.done.emit(line)


def _imread(path: str) -> np.ndarray | None:
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _pixmap_from_rgb(rgb: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(rgb)
    h, w, ch = rgb.shape
    image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


def _thumb_pixmap(path: Path, size: int) -> QPixmap:
    pix = QPixmap(str(path))
    return pix.scaled(
        size,
        size,
        Qt.AspectRatioMode.KeepAspectRatioByExpanding,
        Qt.TransformationMode.SmoothTransformation,
    )


def _join_actions(actions: list[str]) -> str:
    return "；".join(actions)


def _prepare_dialog(dialog: QDialog) -> None:
    dialog.setWindowIcon(app_icon())
    dialog.setStyleSheet(APP_QSS)
    apply_dark_surface(dialog)


def _ok_cancel(dialog: QDialog) -> QDialogButtonBox:
    buttons = QDialogButtonBox()
    ok = buttons.addButton(t("ok"), QDialogButtonBox.ButtonRole.AcceptRole)
    cancel = buttons.addButton(t("cancel"), QDialogButtonBox.ButtonRole.RejectRole)
    ok.setObjectName("Primary")
    ok.clicked.connect(dialog.accept)
    cancel.clicked.connect(dialog.reject)
    return buttons


def _render_preview(frame: PreviewFrame, *, hud: bool = True) -> QPixmap:
    canvas = cv2.flip(frame.rgb, 1)
    pixmap = _pixmap_from_rgb(canvas)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setFont(QFont("Microsoft YaHei UI", 11, QFont.Weight.DemiBold))
    width = canvas.shape[1]
    for hit in frame.hits:
        x = width - hit.x - hit.w
        y, w, h = hit.y, hit.w, hit.h
        if hit.match and hit.match.score >= frame.threshold:
            if hit.match.person.enabled:
                color = QColor("#ff8b7b")
            else:
                color = QColor("#7dd3c7")
            label = f"{hit.match.person.name} {hit.match.score:.2f}"
        elif hit.match:
            color = QColor("#f3c16b")
            label = f"{hit.match.person.name} {hit.match.score:.2f}"
        else:
            color = QColor("#9ec3ff")
            label = t("preview.unknown", score=hit.det_score)
        painter.setPen(QPen(color, 2))
        painter.drawRect(x, y, w, h)
        painter.drawText(x, max(16, y - 6), label)
    if hud and frame.dev_mode:
        nearest = f"{frame.best_score:.2f}" if frame.best_score >= 0 else t("preview.none")
        painter.setPen(QPen(QColor("#f3c16b")))
        painter.setFont(QFont("Microsoft YaHei UI", 10))
        painter.drawText(10, 20, t("preview.dev1", index=frame.camera_index, threshold=frame.threshold))
        painter.drawText(10, 38, t("preview.dev2", faces=len(frame.hits), score=nearest, streak=frame.streak))
        painter.drawText(10, 56, t("preview.dev3"))
    painter.end()
    return pixmap


class ProcessPicker(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("dialog.process_title"))
        _prepare_dialog(self)
        self.resize(420, 520)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        self._filter = QLineEdit()
        self._filter.setPlaceholderText(t("dialog.filter_process"))
        self._list = QListWidget()
        self._names = running_process_names()
        self._list.addItems(self._names)
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        layout.addWidget(self._filter)
        layout.addWidget(self._list, 1)
        layout.addWidget(_ok_cancel(self))
        self._filter.textChanged.connect(self._apply_filter)

    def _apply_filter(self, text: str) -> None:
        needle = text.strip().lower()
        self._list.clear()
        self._list.addItems([name for name in self._names if needle in name])

    def selected(self) -> list[str]:
        return [item.text() for item in self._list.selectedItems()]


class OpenAppPicker(QDialog):
    def __init__(self, parent: QWidget | None, apps: list[OpenApp]) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("dialog.open_title"))
        _prepare_dialog(self)
        self.resize(560, 560)
        self._apps = apps
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        hint = QLabel(t("dialog.open_hint"), objectName="Hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self._filter = QLineEdit()
        self._filter.setPlaceholderText(t("dialog.filter_app"))
        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        layout.addWidget(self._filter)
        layout.addWidget(self._list, 1)
        layout.addWidget(_ok_cancel(self))
        self._filter.textChanged.connect(self._apply_filter)
        self._apply_filter("")

    def _apply_filter(self, text: str) -> None:
        needle = text.strip().lower()
        self._list.clear()
        for app in self._apps:
            hay = f"{app.name} {app.title} {app.exe} {app.path}".lower()
            if needle and needle not in hay:
                continue
            item = QListWidgetItem(f"{app.name}\n{app.title}\n{app.path}")
            item.setData(Qt.ItemDataRole.UserRole, app)
            self._list.addItem(item)

    def selected(self) -> list[OpenApp]:
        picked: list[OpenApp] = []
        for item in self._list.selectedItems():
            app = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(app, OpenApp):
                picked.append(app)
        return picked


class SamePersonDialog(QDialog):
    SAME = 1
    NEW = 2

    def __init__(
        self,
        parent: QWidget,
        thumb: np.ndarray,
        match: MatchResult,
        extra: int = 0,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("dialog.same_title"))
        _prepare_dialog(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        layout.addWidget(QLabel(t("dialog.similar"), objectName="Title"))
        more = t("dialog.same_more", extra=extra) if extra else ""
        hint = QLabel(t("dialog.same_hint", name=match.person.name, score=match.score, more=more), objectName="Hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        row = QHBoxLayout()
        incoming = QLabel(objectName="Thumb")
        incoming.setFixedSize(96, 96)
        rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
        incoming.setPixmap(
            _pixmap_from_rgb(rgb).scaled(
                96, 96, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation
            )
        )
        existing = QLabel(objectName="Thumb")
        existing.setFixedSize(96, 96)
        if match.person.samples and match.person.samples[0].thumb_path.exists():
            existing.setPixmap(_thumb_pixmap(match.person.samples[0].thumb_path, 96))
        row.addWidget(incoming)
        row.addWidget(QLabel("≈"))
        row.addWidget(existing)
        row.addStretch(1)
        layout.addLayout(row)
        buttons = QDialogButtonBox()
        same = buttons.addButton(t("dialog.same"), QDialogButtonBox.ButtonRole.AcceptRole)
        new = buttons.addButton(t("dialog.new"), QDialogButtonBox.ButtonRole.ActionRole)
        cancel = buttons.addButton(t("dialog.cancel"), QDialogButtonBox.ButtonRole.RejectRole)
        same.setObjectName("Primary")
        same.clicked.connect(lambda: self.done(self.SAME))
        new.clicked.connect(lambda: self.done(self.NEW))
        cancel.clicked.connect(self.reject)
        layout.addWidget(buttons)


class MergePersonDialog(QDialog):
    def __init__(self, parent: QWidget, people: list[Person], current_id: str) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("dialog.merge_title"))
        _prepare_dialog(self)
        self.resize(420, 360)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        hint = QLabel(t("dialog.merge_hint"), objectName="Hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self._list = QListWidget()
        for person in people:
            if person.id == current_id:
                continue
            item = QListWidgetItem(t("dialog.merge_item", name=person.name, count=len(person.samples)))
            item.setData(Qt.ItemDataRole.UserRole, person.id)
            self._list.addItem(item)
        layout.addWidget(self._list, 1)
        layout.addWidget(_ok_cancel(self))

    def target_id(self) -> str | None:
        item = self._list.currentItem()
        if not item:
            return None
        return str(item.data(Qt.ItemDataRole.UserRole))


class EnrollFaceDialog(QDialog):
    SAME = 1
    NEW = 2
    SKIP = 3

    def __init__(
        self,
        parent: QWidget,
        thumb: np.ndarray,
        index: int,
        total: int,
        match: MatchResult | None = None,
        forced_name: str | None = None,
        default_name: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("dialog.enroll_title", index=index, total=total))
        _prepare_dialog(self)
        self.resize(480, 420)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        layout.addWidget(QLabel(t("dialog.enroll_head"), objectName="Title"))
        hint = QLabel(t("dialog.enroll_hint", total=total, index=index), objectName="Hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        incoming = QLabel(objectName="Thumb")
        incoming.setFixedSize(160, 160)
        incoming.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
        incoming.setPixmap(
            _pixmap_from_rgb(rgb).scaled(
                160,
                160,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        layout.addWidget(incoming, alignment=Qt.AlignmentFlag.AlignHCenter)

        if match is not None:
            row = QHBoxLayout()
            existing = QLabel(objectName="Thumb")
            existing.setFixedSize(72, 72)
            if match.person.samples and match.person.samples[0].thumb_path.exists():
                existing.setPixmap(_thumb_pixmap(match.person.samples[0].thumb_path, 72))
            row.addStretch(1)
            row.addWidget(QLabel(t("dialog.closest"), objectName="Hint"))
            row.addWidget(existing)
            row.addWidget(QLabel(f"{match.person.name}  {match.score:.2f}", objectName="Hint"))
            row.addStretch(1)
            layout.addLayout(row)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel(t("dialog.name")))
        self._name = QLineEdit()
        self._name.setPlaceholderText(t("dialog.name_ph"))
        self._name.setText(default_name)
        name_row.addWidget(self._name, 1)
        layout.addLayout(name_row)

        buttons = QDialogButtonBox()
        if forced_name:
            same = buttons.addButton(t("dialog.add_to", name=forced_name), QDialogButtonBox.ButtonRole.AcceptRole)
            same.setObjectName("Primary")
            same.clicked.connect(lambda: self.done(self.SAME))
        elif match is not None:
            same = buttons.addButton(t("dialog.link_to", name=match.person.name), QDialogButtonBox.ButtonRole.AcceptRole)
            same.setObjectName("Primary")
            same.clicked.connect(lambda: self.done(self.SAME))
        new = buttons.addButton(t("dialog.new"), QDialogButtonBox.ButtonRole.ActionRole)
        if forced_name is None and match is None:
            new.setObjectName("Primary")
        new.clicked.connect(lambda: self.done(self.NEW))
        skip = buttons.addButton(t("dialog.skip"), QDialogButtonBox.ButtonRole.ActionRole)
        skip.clicked.connect(lambda: self.done(self.SKIP))
        cancel = buttons.addButton(t("dialog.cancel_rest"), QDialogButtonBox.ButtonRole.RejectRole)
        cancel.clicked.connect(self.reject)
        layout.addWidget(buttons)
        self._name.setFocus()

    def chosen_name(self) -> str:
        return self._name.text().strip() or t("dialog.unnamed")


class CaptureDialog(QDialog):
    def __init__(self, parent: QWidget, monitor: MonitorThread) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("capture.title"))
        _prepare_dialog(self)
        self.resize(820, 640)
        self._monitor = monitor
        self._latest_bgr: np.ndarray | None = None
        self._captured: np.ndarray | None = None
        self._frozen = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(12)
        layout.addWidget(QLabel(t("app.eyebrow"), objectName="Eyebrow"))
        layout.addWidget(QLabel(t("capture.head"), objectName="Title"))
        self._hint = QLabel(t("capture.hint"), objectName="Hint")
        self._hint.setWordWrap(True)
        layout.addWidget(self._hint)
        shell = QFrame(objectName="PreviewShell")
        shell_box = QVBoxLayout(shell)
        shell_box.setContentsMargins(10, 10, 10, 10)
        self._preview = QLabel(t("preview.opening"))
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(640, 380)
        self._preview.setStyleSheet("background:#080c12; border-radius:12px;")
        shell_box.addWidget(self._preview)
        layout.addWidget(shell, 1)
        row = QHBoxLayout()
        self._status = QLabel(t("capture.waiting"), objectName="Pill")
        row.addWidget(self._status)
        row.addStretch(1)
        self._btn_cancel = QPushButton(t("cancel"))
        self._btn_cancel.clicked.connect(self._on_cancel)
        self._btn_shot = QPushButton(t("capture.shot"), objectName="Primary")
        self._btn_shot.setIconSize(QSize(16, 16))
        self._btn_shot.setIcon(glyph_icon("camera", 16, COLOR_ON_PRIMARY))
        self._btn_shot.setEnabled(False)
        self._btn_shot.clicked.connect(self._on_shot)
        row.addWidget(self._btn_cancel)
        row.addWidget(self._btn_shot)
        layout.addLayout(row)
        self._monitor.frame_ready.connect(self._on_frame)

    def captured_bgr(self) -> np.ndarray | None:
        return None if self._captured is None else self._captured.copy()

    def _show_pixmap(self, pixmap: QPixmap) -> None:
        self._preview.setPixmap(
            pixmap.scaled(
                self._preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _on_frame(self, frame: PreviewFrame) -> None:
        if self._frozen:
            return
        self._latest_bgr = self._monitor.latest_bgr()
        if not frame.camera_ok:
            self._preview.setText(frame.message or t("capture.no_cam"))
            self._status.setText(t("capture.cam_bad"))
            self._btn_shot.setEnabled(False)
            return
        self._show_pixmap(_render_preview(frame, hud=False))
        count = len(frame.hits)
        self._status.setText(t("capture.faces", count=count) if count else t("capture.no_face"))
        self._btn_shot.setEnabled(self._latest_bgr is not None)

    def _on_shot(self) -> None:
        if not self._frozen:
            if self._latest_bgr is None:
                return
            self._captured = self._latest_bgr.copy()
            self._frozen = True
            rgb = cv2.cvtColor(cv2.flip(self._captured, 1), cv2.COLOR_BGR2RGB)
            self._show_pixmap(_pixmap_from_rgb(rgb))
            self._hint.setText(t("capture.frozen_hint"))
            self._status.setText(t("capture.frozen"))
            self._btn_shot.setText(t("capture.use"))
            self._btn_cancel.setText(t("capture.retake"))
            return
        self.accept()

    def _on_cancel(self) -> None:
        self.reject()

    def reject(self) -> None:
        if self._frozen:
            self._frozen = False
            self._captured = None
            self._hint.setText(t("capture.hint"))
            self._btn_shot.setText(t("capture.shot"))
            self._btn_cancel.setText(t("cancel"))
            return
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._frozen = False
        super().closeEvent(event)

    def done(self, result: int) -> None:
        try:
            self._monitor.frame_ready.disconnect(self._on_frame)
        except RuntimeError:
            pass
        super().done(result)


class MainWindow(QMainWindow):
    def __init__(self, store: SettingsStore, gallery: Gallery, engine: FaceEngine) -> None:
        super().__init__()
        self.store = store
        self.gallery = gallery
        self.engine = engine
        self._armed = False
        self._allow_quit = False
        self._quitting = False
        self._camera_infos: list[CameraInfo] = []
        self._tray: QSystemTrayIcon | None = None
        self._tray_show: QAction | None = None
        self._tray_toggle: QAction | None = None
        self._tray_quit: QAction | None = None
        self._tray_hint_shown = False
        self._seen_active: dict[str, float] = {}
        self._notify_until: dict[str, float] = {}
        self._log_records: list[LogRecord] = []
        self._notify_worker = _NotifyWorker(self)
        self._notify_worker.done.connect(self._log)
        self.setWindowIcon(app_icon())
        self.resize(1180, 760)
        self.setStyleSheet(APP_QSS)

        root = QWidget(objectName="Root")
        self.setCentralWidget(root)
        shell = QHBoxLayout(root)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)
        shell.addWidget(self._build_sidebar())

        self.stack = QStackedWidget()
        apply_dark_surface(self.stack)
        self.page_monitor = self._build_monitor()
        self.page_faces = self._build_faces()
        self.page_work = self._build_work()
        self.page_hide = self._build_hide()
        self.page_notify = self._build_notify()
        self.page_settings = self._build_settings()
        for page in (
            self.page_monitor,
            self.page_faces,
            self.page_work,
            self.page_hide,
            self.page_notify,
            self.page_settings,
        ):
            page.setObjectName("Page")
            apply_dark_surface(page)
            self.stack.addWidget(page)
        shell.addWidget(self.stack, 1)

        self.monitor = MonitorThread(engine, gallery, store, self)
        self.monitor.frame_ready.connect(self._on_frame)
        self.monitor.triggered.connect(self._on_triggered)
        self.monitor.status.connect(self._log)
        self._apply_language()
        QTimer.singleShot(200, self._boot_cameras)
        self.reload_all()

    def bind_tray(self, tray: QSystemTrayIcon, show: QAction, toggle: QAction, quit_action: QAction) -> None:
        self._tray = tray
        self._tray_show = show
        self._tray_toggle = toggle
        self._tray_quit = quit_action
        self._apply_tray_language()
        self._refresh_tray_icon()

    def _build_sidebar(self) -> QWidget:
        side = QFrame(objectName="Sidebar")
        side.setFixedWidth(232)
        layout = QVBoxLayout(side)
        layout.setContentsMargins(18, 22, 18, 18)
        layout.setSpacing(6)
        self.brand_mark = QLabel(objectName="BrandMark")
        self.brand = QLabel(objectName="Brand")
        self.brand_sub = QLabel(objectName="BrandSub")
        self.brand_sub.setWordWrap(True)
        self.brand_logo = QLabel(objectName="BrandLogo")
        self.brand_logo.setFixedSize(40, 40)
        self.brand_logo.setPixmap(app_pixmap(40))
        brand_row = QHBoxLayout()
        brand_row.setContentsMargins(0, 0, 0, 0)
        brand_row.setSpacing(12)
        titles = QVBoxLayout()
        titles.setContentsMargins(0, 0, 0, 0)
        titles.setSpacing(0)
        titles.addWidget(self.brand_mark)
        titles.addWidget(self.brand)
        brand_row.addWidget(self.brand_logo, 0, Qt.AlignmentFlag.AlignVCenter)
        brand_row.addLayout(titles, 1)
        layout.addLayout(brand_row)
        layout.addWidget(self.brand_sub)
        layout.addSpacing(16)
        self.nav_buttons: list[QPushButton] = []
        for idx, (_key, glyph) in enumerate(NAV_ITEMS):
            btn = QPushButton(objectName="Nav")
            btn.setCheckable(True)
            btn.setProperty("glyph", glyph)
            btn.setIconSize(QSize(18, 18))
            btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            btn.clicked.connect(lambda _=False, i=idx: self._goto(i))
            self.nav_buttons.append(btn)
            layout.addWidget(btn)
        self.nav_buttons[0].setChecked(True)
        self._refresh_nav_icons()
        layout.addStretch(1)

        self.lang_label = QLabel(objectName="Hint")
        layout.addWidget(self.lang_label)
        self.btn_lang_zh = QPushButton(objectName="Lang")
        self.btn_lang_en = QPushButton(objectName="Lang")
        self.btn_lang_zh.setCheckable(True)
        self.btn_lang_en.setCheckable(True)
        self.btn_lang_zh.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_lang_en.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        group = QButtonGroup(self)
        group.setExclusive(True)
        group.addButton(self.btn_lang_zh)
        group.addButton(self.btn_lang_en)
        self.btn_lang_zh.clicked.connect(lambda: self._change_language("zh"))
        self.btn_lang_en.clicked.connect(lambda: self._change_language("en"))
        layout.addWidget(self.btn_lang_zh)
        layout.addWidget(self.btn_lang_en)

        self.side_state = QLabel(objectName="Pill")
        self.side_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addSpacing(10)
        layout.addWidget(self.side_state)
        return side

    def _goto(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)
        self._refresh_nav_icons()
        if index == 2:
            self._reload_open_apps()

    def _refresh_nav_icons(self) -> None:
        for btn in self.nav_buttons:
            glyph = str(btn.property("glyph") or "")
            color = COLOR_ACTIVE if btn.isChecked() else COLOR_MUTED
            btn.setIcon(glyph_icon(glyph, 18, color))

    def _sync_monitor_button(self) -> None:
        if self._armed:
            self.btn_monitor.setText(t("btn.stop"))
            self.btn_monitor.setIcon(glyph_icon("stop", 16, COLOR_ON_PRIMARY))
        else:
            self.btn_monitor.setText(t("btn.start"))
            self.btn_monitor.setIcon(glyph_icon("play", 16, COLOR_ON_PRIMARY))

    def _refresh_tray_icon(self) -> None:
        if self._tray is None:
            return
        self._tray.setIcon(app_icon(tray_status(self._armed, self.store.get().dev_mode)))

    def _card(self) -> QFrame:
        return QFrame(objectName="Card")

    def _page(self) -> QWidget:
        page = QWidget(objectName="Page")
        apply_dark_surface(page)
        return page

    def _page_header(self, parent: QVBoxLayout) -> tuple[QLabel, QLabel, QLabel, QHBoxLayout]:
        head = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(4)
        eyebrow = QLabel(objectName="Eyebrow")
        title = QLabel(objectName="Title")
        hint = QLabel(objectName="Hint")
        hint.setWordWrap(True)
        titles.addWidget(eyebrow)
        titles.addWidget(title)
        titles.addWidget(hint)
        head.addLayout(titles, 1)
        parent.addLayout(head)
        return eyebrow, title, hint, head

    def _build_monitor(self) -> QWidget:
        page = self._page()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(26, 22, 26, 22)
        layout.setSpacing(14)
        self.mon_eyebrow, self.mon_title, self.mon_hint, head = self._page_header(layout)
        self.btn_monitor = QPushButton(objectName="Primary")
        self.btn_monitor.setIconSize(QSize(16, 16))
        self.btn_monitor.clicked.connect(self.toggle_monitor)
        self.btn_trigger = QPushButton()
        self.btn_trigger.setIconSize(QSize(16, 16))
        self.btn_trigger.setIcon(glyph_icon("bolt", 16))
        self.btn_trigger.clicked.connect(self.manual_trigger)
        head.addWidget(self.btn_trigger)
        head.addWidget(self.btn_monitor)

        shell = QFrame(objectName="PreviewShell")
        shell_box = QVBoxLayout(shell)
        shell_box.setContentsMargins(10, 10, 10, 10)
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(380)
        self.preview.setStyleSheet("background:#080c12; border-radius:12px;")
        shell_box.addWidget(self.preview)
        layout.addWidget(shell, 1)

        pills = QHBoxLayout()
        pills.setSpacing(8)
        self.pill_fps = QLabel(objectName="Pill")
        self.pill_match = QLabel(objectName="Pill")
        self.pill_faces = QLabel(objectName="Pill")
        pills.addWidget(self.pill_fps)
        pills.addWidget(self.pill_match)
        pills.addWidget(self.pill_faces)
        pills.addStretch(1)
        self.btn_export_log = QPushButton()
        self.btn_export_log.clicked.connect(self.export_log)
        pills.addWidget(self.btn_export_log)
        layout.addLayout(pills)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(132)
        layout.addWidget(self.log)
        return page

    def _build_faces(self) -> QWidget:
        page = self._page()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(26, 22, 26, 22)
        layout.setSpacing(14)
        self.faces_eyebrow, self.faces_title, self.faces_hint, head = self._page_header(layout)
        self.btn_upload = QPushButton(objectName="Primary")
        self.btn_upload.setIconSize(QSize(16, 16))
        self.btn_upload.setIcon(glyph_icon("upload", 16, COLOR_ON_PRIMARY))
        self.btn_upload.clicked.connect(self.add_person_from_file)
        self.btn_capture = QPushButton()
        self.btn_capture.setIconSize(QSize(16, 16))
        self.btn_capture.setIcon(glyph_icon("camera", 16))
        self.btn_capture.clicked.connect(self.add_person_from_camera)
        head.addWidget(self.btn_capture)
        head.addWidget(self.btn_upload)
        self.face_host = QWidget(objectName="FaceHost")
        apply_dark_surface(self.face_host)
        self.face_grid = QVBoxLayout(self.face_host)
        self.face_grid.setContentsMargins(0, 0, 0, 0)
        self.face_grid.setSpacing(12)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        apply_dark_surface(scroll)
        apply_dark_surface(scroll.viewport())
        scroll.setWidget(self.face_host)
        layout.addWidget(scroll, 1)
        return page

    def _build_work(self) -> QWidget:
        page = self._page()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(26, 22, 26, 22)
        layout.setSpacing(12)
        self.work_eyebrow, self.work_title, self.work_hint, _head = self._page_header(layout)
        quick = QHBoxLayout()
        self.work_quick_label = QLabel(objectName="Section")
        quick.addWidget(self.work_quick_label)
        for name, path in common_work_apps():
            btn = QPushButton(name)
            btn.clicked.connect(lambda _=False, n=name, p=path: self._add_work_app(n, p))
            quick.addWidget(btn)
        quick.addStretch(1)
        layout.addLayout(quick)
        pick_row = QHBoxLayout()
        self.btn_browse = QPushButton()
        self.btn_browse.clicked.connect(self.browse_work_app)
        self.btn_pick_open = QPushButton()
        self.btn_pick_open.clicked.connect(self.pick_open_work_app)
        pick_row.addWidget(self.btn_browse)
        pick_row.addWidget(self.btn_pick_open)
        pick_row.addStretch(1)
        layout.addLayout(pick_row)
        self.work_selected_label = QLabel(objectName="Section")
        layout.addWidget(self.work_selected_label)
        self.work_list = QListWidget()
        layout.addWidget(self.work_list, 1)
        row = QHBoxLayout()
        self.btn_work_remove = QPushButton(objectName="Danger")
        self.btn_work_remove.clicked.connect(self.remove_work_app)
        self.btn_work_up = QPushButton()
        self.btn_work_down = QPushButton()
        self.btn_work_up.clicked.connect(lambda: self._move_work(-1))
        self.btn_work_down.clicked.connect(lambda: self._move_work(1))
        row.addWidget(self.btn_work_remove)
        row.addWidget(self.btn_work_up)
        row.addWidget(self.btn_work_down)
        row.addStretch(1)
        layout.addLayout(row)

        open_head = QHBoxLayout()
        self.work_open_label = QLabel(objectName="Section")
        open_head.addWidget(self.work_open_label)
        self.btn_open_refresh = QPushButton()
        self.btn_open_refresh.clicked.connect(self._reload_open_apps)
        self.btn_open_add = QPushButton()
        self.btn_open_add.clicked.connect(self.add_selected_open_apps)
        open_head.addStretch(1)
        open_head.addWidget(self.btn_open_refresh)
        open_head.addWidget(self.btn_open_add)
        layout.addLayout(open_head)
        self.open_app_list = QListWidget()
        self.open_app_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.open_app_list.itemDoubleClicked.connect(lambda item: self._add_open_app_item(item))
        layout.addWidget(self.open_app_list, 1)
        return page

    def _build_hide(self) -> QWidget:
        page = self._page()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(26, 22, 26, 22)
        layout.setSpacing(12)
        self.hide_eyebrow, self.hide_title, self.hide_hint, _head = self._page_header(layout)
        card = QFrame(objectName="SettingsCard")
        form = QVBoxLayout(card)
        form.setContentsMargins(16, 14, 16, 14)
        form.setSpacing(8)
        self.chk_foreground = QCheckBox()
        self.chk_others = QCheckBox()
        self.chk_fullscreen = QCheckBox()
        for box in (self.chk_foreground, self.chk_others, self.chk_fullscreen):
            box.stateChanged.connect(self._save_from_ui)
            form.addWidget(box)
        layout.addWidget(card)
        row = QHBoxLayout()
        self.btn_ent_add = QPushButton()
        self.btn_ent_add.clicked.connect(self.add_entertainment)
        self.btn_ent_pick = QPushButton()
        self.btn_ent_pick.clicked.connect(self.pick_entertainment)
        row.addWidget(self.btn_ent_add)
        row.addWidget(self.btn_ent_pick)
        row.addStretch(1)
        layout.addLayout(row)
        self.ent_list = QListWidget()
        layout.addWidget(self.ent_list, 1)
        self.btn_ent_remove = QPushButton(objectName="Danger")
        self.btn_ent_remove.clicked.connect(self.remove_entertainment)
        layout.addWidget(self.btn_ent_remove, alignment=Qt.AlignmentFlag.AlignLeft)
        return page

    def _build_notify(self) -> QWidget:
        page = self._page()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(26, 22, 26, 22)
        layout.setSpacing(12)
        self.notify_eyebrow, self.notify_title, self.notify_hint, _head = self._page_header(layout)
        add_row = QHBoxLayout()
        self.btn_ch_ding_group = QPushButton()
        self.btn_ch_ding_app = QPushButton()
        self.btn_ch_feishu = QPushButton()
        self.btn_ch_webhook = QPushButton()
        for kind, btn in (
            ("dingtalk_group", self.btn_ch_ding_group),
            ("dingtalk_app", self.btn_ch_ding_app),
            ("feishu", self.btn_ch_feishu),
            ("webhook", self.btn_ch_webhook),
        ):
            btn.clicked.connect(lambda _=False, k=kind: self._edit_channel(k))
            add_row.addWidget(btn)
        add_row.addStretch(1)
        layout.addLayout(add_row)
        self.channel_list = QListWidget()
        self.channel_list.itemDoubleClicked.connect(lambda _item: self._edit_channel())
        layout.addWidget(self.channel_list, 1)
        row = QHBoxLayout()
        self.btn_ch_edit = QPushButton()
        self.btn_ch_edit.clicked.connect(lambda: self._edit_channel())
        self.btn_ch_test = QPushButton()
        self.btn_ch_test.clicked.connect(self._test_channel)
        self.btn_ch_remove = QPushButton(objectName="Danger")
        self.btn_ch_remove.clicked.connect(self._remove_channel)
        row.addWidget(self.btn_ch_edit)
        row.addWidget(self.btn_ch_test)
        row.addWidget(self.btn_ch_remove)
        row.addStretch(1)
        layout.addLayout(row)
        return page

    def _build_settings(self) -> QWidget:
        page = self._page()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(26, 22, 26, 22)
        layout.setSpacing(14)
        self.set_eyebrow, self.set_title, self.set_hint, _head = self._page_header(layout)
        self.set_hint.hide()
        card = QFrame(objectName="SettingsCard")
        form = QVBoxLayout(card)
        form.setContentsMargins(18, 18, 18, 18)
        form.setSpacing(12)
        cam_row = QHBoxLayout()
        self.set_camera_label = QLabel()
        cam_row.addWidget(self.set_camera_label)
        self.camera_box = QComboBox()
        for index in range(4):
            self.camera_box.addItem(str(index), index)
        self.camera_box.currentIndexChanged.connect(self._save_from_ui)
        cam_row.addWidget(self.camera_box, 1)
        form.addLayout(cam_row)

        self.lbl_threshold = QLabel()
        self.threshold = QSlider(Qt.Orientation.Horizontal)
        self.threshold.setRange(25, 70)
        self.threshold.valueChanged.connect(self._sync_slider_labels)
        self.threshold.sliderReleased.connect(self._save_from_ui)
        form.addWidget(self.lbl_threshold)
        form.addWidget(self.threshold)

        self.lbl_confirm = QLabel()
        self.confirm = QSlider(Qt.Orientation.Horizontal)
        self.confirm.setRange(1, 12)
        self.confirm.valueChanged.connect(self._sync_slider_labels)
        self.confirm.sliderReleased.connect(self._save_from_ui)
        form.addWidget(self.lbl_confirm)
        form.addWidget(self.confirm)

        self.lbl_cooldown = QLabel()
        self.cooldown = QSlider(Qt.Orientation.Horizontal)
        self.cooldown.setRange(3, 60)
        self.cooldown.valueChanged.connect(self._sync_slider_labels)
        self.cooldown.sliderReleased.connect(self._save_from_ui)
        form.addWidget(self.lbl_cooldown)
        form.addWidget(self.cooldown)

        self.chk_autolink = QCheckBox()
        self.chk_autolink.stateChanged.connect(self._save_from_ui)
        form.addWidget(self.chk_autolink)
        self.chk_dev = QCheckBox()
        self.chk_dev.stateChanged.connect(self._save_from_ui)
        form.addWidget(self.chk_dev)
        self.chk_autostart = QCheckBox()
        self.chk_autostart.stateChanged.connect(self._save_from_ui)
        form.addWidget(self.chk_autostart)
        self.chk_start_min = QCheckBox()
        self.chk_start_min.stateChanged.connect(self._save_from_ui)
        form.addWidget(self.chk_start_min)
        self.chk_boot = QCheckBox()
        self.chk_boot.stateChanged.connect(self._save_from_ui)
        form.addWidget(self.chk_boot)
        layout.addWidget(card)
        layout.addStretch(1)
        return page

    def _change_language(self, lang: str) -> None:
        chosen = set_language(lang)
        settings = self.store.get()
        if settings.language != chosen:
            settings.language = chosen
            self.store.replace(settings)
        self._apply_language()

    def _apply_language(self) -> None:
        lang = current_language()
        app = QApplication.instance()
        if app is not None:
            app.setApplicationName(t("app.name"))
        self.brand_mark.setText(t("app.eyebrow"))
        self.brand.setText(t("app.name"))
        self.brand_sub.setText(t("app.tagline"))
        self.lang_label.setText(t("lang.label"))
        self.btn_lang_zh.setText(t("lang.zh"))
        self.btn_lang_en.setText(t("lang.en"))
        self.btn_lang_zh.blockSignals(True)
        self.btn_lang_en.blockSignals(True)
        self.btn_lang_zh.setChecked(lang == "zh")
        self.btn_lang_en.setChecked(lang == "en")
        self.btn_lang_zh.blockSignals(False)
        self.btn_lang_en.blockSignals(False)
        for btn, key in zip(self.nav_buttons, NAV_KEYS, strict=True):
            btn.setText(t(key))
        self._refresh_nav_icons()

        self.mon_eyebrow.setText(t("app.eyebrow"))
        self.mon_title.setText(t("monitor.title"))
        self.mon_hint.setText(t("monitor.hint"))
        self.btn_trigger.setText(t("btn.trigger"))
        self.btn_export_log.setText(t("btn.export_excel"))
        self._sync_monitor_button()
        pixmap = self.preview.pixmap()
        if pixmap is None or pixmap.isNull():
            self.preview.setText(t("preview.opening"))
        self.pill_fps.setText(t("pill.fps", fps=0))
        self.pill_match.setText(t("pill.unmatched"))
        self.pill_faces.setText(t("pill.faces", count=0))

        self.faces_eyebrow.setText(t("app.eyebrow"))
        self.faces_title.setText(t("faces.title"))
        self.faces_hint.setText(t("faces.hint"))
        self.btn_upload.setText(t("btn.upload"))
        self.btn_capture.setText(t("btn.capture"))

        self.work_eyebrow.setText(t("app.eyebrow"))
        self.work_title.setText(t("work.title"))
        self.work_hint.setText(t("work.hint"))
        self.work_quick_label.setText(t("work.quick"))
        self.work_selected_label.setText(t("work.selected"))
        self.work_open_label.setText(t("work.open"))
        self.btn_browse.setText(t("work.browse"))
        self.btn_pick_open.setText(t("work.pick_open"))
        self.btn_work_remove.setText(t("work.remove"))
        self.btn_work_up.setText(t("work.up"))
        self.btn_work_down.setText(t("work.down"))
        self.btn_open_refresh.setText(t("work.refresh"))
        self.btn_open_add.setText(t("work.add_selected"))

        self.hide_eyebrow.setText(t("app.eyebrow"))
        self.hide_title.setText(t("hide.title"))
        self.hide_hint.setText(t("hide.hint"))
        self.chk_foreground.setText(t("hide.foreground"))
        self.chk_others.setText(t("hide.others"))
        self.chk_fullscreen.setText(t("hide.fullscreen"))
        self.btn_ent_add.setText(t("hide.add"))
        self.btn_ent_pick.setText(t("hide.pick"))
        self.btn_ent_remove.setText(t("hide.remove"))

        self.notify_eyebrow.setText(t("app.eyebrow"))
        self.notify_title.setText(t("notify.title"))
        self.notify_hint.setText(t("notify.hint"))
        self.btn_ch_ding_group.setText(t("channel.add_dingtalk_group"))
        self.btn_ch_ding_app.setText(t("channel.add_dingtalk_app"))
        self.btn_ch_feishu.setText(t("channel.add_feishu"))
        self.btn_ch_webhook.setText(t("channel.add_webhook"))
        self.btn_ch_edit.setText(t("channel.edit"))
        self.btn_ch_test.setText(t("channel.test"))
        self.btn_ch_remove.setText(t("channel.remove"))

        self.set_eyebrow.setText(t("app.eyebrow"))
        self.set_title.setText(t("settings.title"))
        self.set_camera_label.setText(t("settings.camera"))
        self.chk_autolink.setText(t("settings.autolink"))
        self.chk_autostart.setText(t("settings.autostart"))
        self.chk_start_min.setText(t("settings.start_minimized"))
        self.chk_boot.setText(t("settings.start_on_boot"))
        self.chk_dev.setText(t("settings.dev"))

        self._apply_dev_chrome()
        self._sync_slider_labels()
        self._refresh_camera_labels()
        if hasattr(self, "work_list"):
            self._reload_work(self.store.get())
            self._reload_open_apps()
            self._reload_faces()
            self._reload_channels(self.store.get())
        self._apply_tray_language()

    def _apply_tray_language(self) -> None:
        if self._tray is not None:
            self._tray.setToolTip(t("app.name"))
        if self._tray_show is not None:
            self._tray_show.setText(t("tray.open"))
        if self._tray_toggle is not None:
            self._tray_toggle.setText(t("tray.toggle"))
        if self._tray_quit is not None:
            self._tray_quit.setText(t("tray.quit"))

    def reload_all(self) -> None:
        settings = self.store.get()
        self._loading = True
        idx = self.camera_box.findData(settings.camera_index)
        self.camera_box.setCurrentIndex(idx if idx >= 0 else 0)
        self.threshold.setValue(int(round(settings.match_threshold * 100)))
        self.confirm.setValue(settings.confirm_frames)
        self.cooldown.setValue(int(settings.cooldown_seconds))
        self.chk_autostart.setChecked(settings.auto_start_monitor)
        self.chk_start_min.setChecked(settings.start_minimized)
        self.chk_boot.setChecked(settings.start_on_boot)
        self.chk_dev.setChecked(settings.dev_mode)
        self.chk_autolink.setChecked(settings.auto_link_same_person)
        self._apply_dev_chrome(settings)
        self.chk_foreground.setChecked(settings.hide_foreground)
        self.chk_others.setChecked(settings.minimize_other_windows)
        self.chk_fullscreen.setChecked(settings.break_fullscreen)
        self._sync_slider_labels()
        self._reload_work(settings)
        self._reload_entertainment(settings)
        self._reload_channels(settings)
        self._reload_faces()
        self._reload_open_apps()
        self._loading = False

    def _sync_slider_labels(self) -> None:
        self.lbl_threshold.setText(t("settings.threshold", value=self.threshold.value() / 100))
        self.lbl_confirm.setText(t("settings.confirm", value=self.confirm.value()))
        self.lbl_cooldown.setText(t("settings.cooldown", value=self.cooldown.value()))

    def _collect_settings(self) -> Settings:
        settings = self.store.get()
        data = self.camera_box.currentData()
        settings.camera_index = int(data) if data is not None else 0
        settings.match_threshold = self.threshold.value() / 100.0
        settings.confirm_frames = self.confirm.value()
        settings.cooldown_seconds = float(self.cooldown.value())
        settings.auto_start_monitor = self.chk_autostart.isChecked()
        settings.start_minimized = self.chk_start_min.isChecked()
        settings.start_on_boot = self.chk_boot.isChecked()
        settings.dev_mode = self.chk_dev.isChecked()
        settings.auto_link_same_person = self.chk_autolink.isChecked()
        settings.hide_foreground = self.chk_foreground.isChecked()
        settings.minimize_other_windows = self.chk_others.isChecked()
        settings.break_fullscreen = self.chk_fullscreen.isChecked()
        settings.language = current_language()
        return settings

    def _save_from_ui(self) -> None:
        if getattr(self, "_loading", False):
            return
        settings = self._collect_settings()
        try:
            sync_startup(settings.start_on_boot)
        except OSError as exc:
            QMessageBox.warning(self, t("settings.title"), t("settings.boot_fail", error=exc))
            self.chk_boot.blockSignals(True)
            self.chk_boot.setChecked(not settings.start_on_boot)
            self.chk_boot.blockSignals(False)
            settings.start_on_boot = self.chk_boot.isChecked()
        self.store.replace(settings)
        self._apply_dev_chrome(settings)

    def _apply_dev_chrome(self, settings: Settings | None = None) -> None:
        current = settings or self.store.get()
        self.setWindowTitle(t("app.name_dev") if current.dev_mode else t("app.name"))
        if self._armed:
            self.side_state.setText(t("status.drill") if current.dev_mode else t("status.watching"))
            self.side_state.setProperty("state", "warn" if current.dev_mode else "on")
        elif current.dev_mode:
            self.side_state.setText(t("status.dev"))
            self.side_state.setProperty("state", "warn")
        else:
            self.side_state.setText(t("status.idle"))
            self.side_state.setProperty("state", "")
        self.side_state.style().unpolish(self.side_state)
        self.side_state.style().polish(self.side_state)
        self._refresh_tray_icon()

    def _reload_work(self, settings: Settings) -> None:
        self.work_list.clear()
        if not settings.work_apps:
            self.work_list.addItem(t("work.empty"))
            return
        for app in settings.work_apps:
            item = QListWidgetItem(f"{app.name}\n{app.path}")
            item.setData(Qt.ItemDataRole.UserRole, app.id)
            self.work_list.addItem(item)

    def _reload_open_apps(self) -> None:
        if not hasattr(self, "open_app_list"):
            return
        apps = list_open_apps(exclude_pids={os.getpid()})
        self.open_app_list.clear()
        if not apps:
            self.open_app_list.addItem(t("work.no_open"))
            return
        for app in apps:
            item = QListWidgetItem(f"{app.name}  ·  {app.title}\n{app.path}")
            item.setData(Qt.ItemDataRole.UserRole, app)
            self.open_app_list.addItem(item)

    def _add_open_app_item(self, item: QListWidgetItem) -> None:
        app = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(app, OpenApp):
            self._add_work_app(app.name, app.path)

    def add_selected_open_apps(self) -> None:
        items = self.open_app_list.selectedItems()
        if not items:
            QMessageBox.information(self, t("msg.work_title"), t("work.pick_first"))
            return
        for item in items:
            self._add_open_app_item(item)

    def pick_open_work_app(self) -> None:
        apps = list_open_apps(exclude_pids={os.getpid()})
        if not apps:
            QMessageBox.information(self, t("msg.work_title"), t("work.none_detected"))
            return
        dialog = OpenAppPicker(self, apps)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        for app in dialog.selected():
            self._add_work_app(app.name, app.path)
        self._reload_open_apps()

    def _reload_entertainment(self, settings: Settings) -> None:
        self.ent_list.clear()
        self.ent_list.addItems(settings.entertainment_processes)

    def _reload_channels(self, settings: Settings) -> None:
        if not hasattr(self, "channel_list"):
            return
        self.channel_list.clear()
        if not settings.channels:
            self.channel_list.addItem(t("channel.empty"))
            return
        for channel in settings.channels:
            item = QListWidgetItem(channel_summary(channel))
            item.setData(Qt.ItemDataRole.UserRole, channel.id)
            self.channel_list.addItem(item)

    def _current_channel(self) -> MessageChannel | None:
        item = self.channel_list.currentItem()
        if item is None:
            return None
        cid = item.data(Qt.ItemDataRole.UserRole)
        if not cid:
            return None
        for channel in self.store.get().channels:
            if channel.id == cid:
                return channel
        return None

    def _edit_channel(self, kind: str | None = None) -> None:
        existing = None if kind else self._current_channel()
        if kind is None and existing is None:
            QMessageBox.information(self, t("notify.title"), t("channel.pick_first"))
            return
        dialog = ChannelDialog(self, kind or existing.kind, existing)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        result = dialog.result_channel()
        settings = self.store.get()
        if existing is None:
            settings.channels.append(result)
        else:
            settings.channels = [result if item.id == existing.id else item for item in settings.channels]
        self.store.replace(settings)
        self._reload_channels(settings)

    def _remove_channel(self) -> None:
        channel = self._current_channel()
        if channel is None:
            QMessageBox.information(self, t("notify.title"), t("channel.pick_first"))
            return
        settings = self.store.get()
        settings.channels = [item for item in settings.channels if item.id != channel.id]
        self.store.replace(settings)
        self._reload_channels(settings)

    def _test_channel(self) -> None:
        channel = self._current_channel()
        if channel is None:
            QMessageBox.information(self, t("notify.title"), t("channel.pick_first"))
            return
        event = NotifyEvent(person=t("app.name"), score=1.0, when=datetime.now(), test=True)
        self._notify_worker.start(event, [channel])

    def _reload_faces(self) -> None:
        while self.face_grid.count():
            item = self.face_grid.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        people = self.gallery.people()
        if not people:
            empty = QFrame(objectName="EmptyFace")
            box = QVBoxLayout(empty)
            box.setContentsMargins(28, 48, 28, 48)
            icon = QLabel()
            icon.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            icon.setPixmap(glyph_pixmap("faces", 56, COLOR_EMPTY))
            title = QLabel(t("faces.empty_title"))
            title.setStyleSheet("font-size:16px; font-weight:600; background:transparent;")
            title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            hint = QLabel(t("faces.empty_hint"), objectName="Hint")
            hint.setWordWrap(True)
            hint.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            box.addWidget(icon)
            box.addWidget(title)
            box.addWidget(hint)
            self.face_grid.addWidget(empty)
            self.face_grid.addStretch(1)
            return
        for person in people:
            self.face_grid.addWidget(self._face_card(person))
        self.face_grid.addStretch(1)

    def _face_card(self, person: Person) -> QFrame:
        card = self._card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        thumbs = QHBoxLayout()
        thumbs.setSpacing(8)
        visible = person.samples[:6]
        for sample in visible:
            thumbs.addWidget(self._sample_thumb(person, sample))
        leftover = len(person.samples) - len(visible)
        if leftover > 0:
            more = QLabel(f"+{leftover}", objectName="Pill")
            more.setAlignment(Qt.AlignmentFlag.AlignCenter)
            more.setFixedSize(56, 56)
            thumbs.addWidget(more)
        thumbs.addStretch(1)
        layout.addLayout(thumbs)

        info = QHBoxLayout()
        name = QLabel(person.name)
        name.setStyleSheet("font-size:16px; font-weight:600; background:transparent;")
        info.addWidget(name)
        tag = t("faces.auto_tag") if person.auto_linked else t("faces.manual_tag")
        pill = QLabel(t("faces.same_count", count=len(person.samples), tag=tag), objectName="Pill")
        if person.auto_linked:
            pill.setProperty("state", "on")
        info.addWidget(pill)
        status = QLabel(t("faces.status_on") if person.enabled else t("faces.status_off"), objectName="Pill")
        status.setProperty("state", "on" if person.enabled else "warn")
        info.addWidget(status)
        if person.blacklisted:
            blocked = QLabel(t("faces.status_block"), objectName="Pill")
            blocked.setProperty("state", "alert")
            info.addWidget(blocked)
        info.addStretch(1)
        layout.addLayout(info)

        enable = QCheckBox(t("faces.enable"))
        enable.setChecked(person.enabled)
        enable.setToolTip(t("faces.enable_hint"))
        enable.toggled.connect(
            lambda checked, pid=person.id, host=card, badge=status: self._toggle_person_enabled(pid, checked, host, badge)
        )
        layout.addWidget(enable)
        block = QCheckBox(t("faces.blacklist"))
        block.setChecked(person.blacklisted)
        block.setToolTip(t("faces.blacklist_hint"))
        block.toggled.connect(lambda checked, pid=person.id: self._toggle_person_blacklist(pid, checked))
        layout.addWidget(block)
        if not person.enabled:
            card.setProperty("muted", "true")

        buttons = QHBoxLayout()
        add = QPushButton(t("faces.add_photo"))
        add.clicked.connect(lambda: self.add_sample(person.id))
        merge = QPushButton(t("faces.merge"))
        merge.clicked.connect(lambda: self.merge_person(person.id))
        merge.setEnabled(len(self.gallery.people()) > 1)
        rename = QPushButton(t("faces.rename"))
        rename.clicked.connect(lambda: self.rename_person(person.id, person.name))
        delete = QPushButton(t("faces.delete"), objectName="Danger")
        delete.clicked.connect(lambda: self.delete_person(person.id))
        buttons.addWidget(add)
        buttons.addWidget(merge)
        buttons.addWidget(rename)
        buttons.addWidget(delete)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        return card

    def _toggle_person_enabled(self, person_id: str, enabled: bool, card: QFrame, status: QLabel) -> None:
        try:
            self.gallery.set_enabled(person_id, enabled)
        except KeyError:
            return
        person = self.gallery.person(person_id)
        name = person.name if person is not None else person_id
        status.setText(t("faces.status_on") if enabled else t("faces.status_off"))
        status.setProperty("state", "on" if enabled else "warn")
        status.style().unpolish(status)
        status.style().polish(status)
        card.setProperty("muted", "true" if not enabled else "false")
        card.style().unpolish(card)
        card.style().polish(card)
        self._log(t("log.enabled" if enabled else "log.disabled", name=name), person=name)

    def _toggle_person_blacklist(self, person_id: str, blacklisted: bool) -> None:
        try:
            self.gallery.set_blacklisted(person_id, blacklisted)
        except KeyError:
            return
        person = self.gallery.person(person_id)
        name = person.name if person is not None else person_id
        self._log(t("log.blacklist_on" if blacklisted else "log.blacklist_off", name=name), person=name)
        self._reload_faces()

    def _sample_thumb(self, person: Person, sample: Sample) -> QLabel:
        thumb = QLabel(objectName="Thumb")
        thumb.setFixedSize(64, 64)
        thumb.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        if sample.thumb_path.exists():
            thumb.setPixmap(_thumb_pixmap(sample.thumb_path, 64))
        source_key = SAMPLE_SOURCE_KEYS.get(sample.source)
        source = t(source_key) if source_key else sample.source
        thumb.setToolTip(t("sample.tip", source=source))
        thumb.mousePressEvent = lambda event, pid=person.id, sid=sample.id: self._sample_menu(pid, sid, event)
        return thumb

    def _sample_menu(self, person_id: str, sample_id: str, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton and event.button() != Qt.MouseButton.RightButton:
            return
        person = self.gallery.person(person_id)
        if person is None:
            return
        menu = QMenu(self)
        split = menu.addAction(t("menu.split"))
        split.setEnabled(len(person.samples) > 1)
        delete = menu.addAction(t("menu.delete"))
        chosen = menu.exec(event.globalPosition().toPoint())
        if chosen == split:
            self.split_sample(person_id, sample_id)
        elif chosen == delete:
            self.delete_sample(person_id, sample_id)

    def _refresh_camera_labels(self) -> None:
        if not hasattr(self, "camera_box"):
            return
        current = self.camera_box.currentData()
        self.camera_box.blockSignals(True)
        self.camera_box.clear()
        if self._camera_infos:
            for info in self._camera_infos:
                mark = t("settings.placeholder") if info.placeholder else ""
                self.camera_box.addItem(f"{info.index}  {info.width}x{info.height}{mark}", info.index)
        else:
            for index in range(4):
                self.camera_box.addItem(str(index), index)
        idx = self.camera_box.findData(current)
        self.camera_box.setCurrentIndex(idx if idx >= 0 else 0)
        self.camera_box.blockSignals(False)

    def _boot_cameras(self) -> None:
        infos = describe_cameras()
        self._camera_infos = infos
        self._loading = True
        settings = self.store.get()
        chosen = pick_camera(infos, settings.camera_index)
        self._refresh_camera_labels()
        if infos and chosen != settings.camera_index:
            settings.camera_index = chosen
            self.store.replace(settings)
            self._log(t("log.camera_switched", index=chosen))
        idx = self.camera_box.findData(chosen)
        self.camera_box.blockSignals(True)
        self.camera_box.setCurrentIndex(idx if idx >= 0 else 0)
        self.camera_box.blockSignals(False)
        self._loading = False
        self._start_preview()
        if self.store.get().auto_start_monitor and self.gallery.people():
            self.start_monitor()

    def _start_preview(self) -> None:
        if not self.monitor.isRunning():
            self.monitor.start()

    def hwnd(self) -> int:
        return int(self.winId())

    def start_monitor(self) -> None:
        self._start_preview()
        self.monitor.set_protected_hwnds({self.hwnd()})
        self.monitor.set_armed(True)
        self._armed = True
        self._sync_monitor_button()
        self._apply_dev_chrome()
        if self.store.get().dev_mode:
            self._log(t("log.monitor_start_dev"))
        else:
            self._log(t("log.monitor_start"))

    def stop_monitor(self) -> None:
        self.monitor.set_armed(False)
        self._armed = False
        self._sync_monitor_button()
        self._apply_dev_chrome()
        self._log(t("log.monitor_stop"))

    def toggle_monitor(self) -> None:
        if self._armed:
            self.stop_monitor()
        else:
            if not self.gallery.people():
                QMessageBox.information(self, t("app.name"), t("msg.need_face"))
                self._goto(1)
                return
            self.start_monitor()

    def _on_frame(self, frame: PreviewFrame) -> None:
        if not frame.camera_ok:
            self.preview.setText(frame.message)
            return
        pix = _render_preview(frame)
        self.preview.setPixmap(
            pix.scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        )
        self.pill_fps.setText(t("pill.fps", fps=frame.fps))
        self.pill_faces.setText(t("pill.faces", count=len(frame.hits)))
        if frame.matched_name:
            if frame.matched_armed:
                self.pill_match.setText(t("pill.hit", name=frame.matched_name, streak=frame.streak))
                self.pill_match.setProperty("state", "alert")
            else:
                self.pill_match.setText(t("pill.seen", name=frame.matched_name))
                self.pill_match.setProperty("state", "on")
        else:
            self.pill_match.setText(t("pill.unmatched"))
            self.pill_match.setProperty("state", "")
        self.pill_match.style().unpolish(self.pill_match)
        self.pill_match.style().polish(self.pill_match)
        self._log_seen_faces(frame.seen)

    def _log_seen_faces(self, seen: list[SeenFace]) -> None:
        present: dict[str, SeenFace] = {}
        for item in seen:
            prev = present.get(item.name)
            if prev is None or item.score > prev.score:
                present[item.name] = item
        self._seen_active, newly = track_seen(self._seen_active, present, time.monotonic())
        for item in newly:
            key = "log.seen" if item.hide_enabled else "log.seen_off"
            self._log(t(key, name=item.name, score=item.score), person=item.name)
            if item.blacklisted:
                self._notify_blacklist(item)

    def _notify_blacklist(self, item: SeenFace) -> None:
        now = time.monotonic()
        until = self._notify_until.get(item.name, 0.0)
        if now < until:
            return
        cooldown = self.store.get().cooldown_seconds
        self._notify_until[item.name] = now + max(1.0, cooldown)
        self._log(t("log.blacklist", name=item.name, score=item.score), person=item.name)
        channels = [channel for channel in self.store.get().channels if channel.enabled]
        event = NotifyEvent(person=item.name, score=item.score, when=datetime.now())
        self._notify_worker.start(event, channels)

    def _on_triggered(self, event: TriggerEvent) -> None:
        if event.error:
            self._log(t("log.trigger_fail", name=event.person_name, error=event.error), person=event.person_name)
            return
        key = "log.drill" if event.dry_run else "log.hit"
        self._log(
            t(key, name=event.person_name, score=event.score, actions=_join_actions(event.actions)),
            person=event.person_name,
        )

    def _log(self, text: str, person: str | None = None) -> None:
        when = datetime.now()
        self._log_records.append(LogRecord(when=when, text=text, person=person))
        self.log.appendPlainText(format_log_line(when, text))

    def export_log(self) -> None:
        if not self._log_records:
            QMessageBox.information(self, t("file.export_log"), t("log.export_empty"))
            return
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        desktop = Path.home() / "Desktop"
        folder = desktop if desktop.is_dir() else Path.home()
        suggested = str(folder / f"FaceHide-{stamp}.xlsx")
        path, _ = QFileDialog.getSaveFileName(self, t("file.export_log"), suggested, t("file.excel"))
        if not path:
            return
        target = Path(path)
        if target.suffix.lower() != ".xlsx":
            target = target.with_suffix(".xlsx")
        try:
            write_xlsx(
                target,
                self._log_records,
                headers=(t("log.col_time"), t("log.col_person"), t("log.col_text")),
                sheet=t("log.sheet"),
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, t("file.export_log"), t("log.export_fail", error=exc))
            return
        self._log(t("log.export_ok", path=target))
        QMessageBox.information(self, t("file.export_log"), t("log.export_ok", path=target))

    def _ask_name(self, default: str = "") -> str | None:
        name, ok = QInputDialog.getText(self, t("ask.name_title"), t("ask.name_label"), text=default)
        if not ok:
            return None
        return name.strip() or t("dialog.unnamed")

    def _extract_faces(self, bgr: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]] | None:
        try:
            return self.engine.enroll_all(bgr)
        except NoFaceError as exc:
            QMessageBox.warning(self, t("msg.cannot_enroll"), str(exc))
            return None

    def _append_samples(
        self,
        person_id: str,
        faces: list[tuple[np.ndarray, np.ndarray]],
        source: str,
    ) -> None:
        for feature, thumb in faces:
            self.gallery.add_sample(person_id, feature, thumb, source=source)

    def _create_person(self, name: str, faces: list[tuple[np.ndarray, np.ndarray]], source: str) -> Person:
        first, rest = faces[0], faces[1:]
        person = self.gallery.add_person(name, first[0], first[1], source=source)
        for feature, thumb in rest:
            self.gallery.add_sample(person.id, feature, thumb, source=source)
        return person

    def _apply_face(
        self,
        face: tuple[np.ndarray, np.ndarray],
        *,
        person_id: str,
        source: str,
        message: str,
    ) -> None:
        self._append_samples(person_id, [face], source)
        self._log(message)

    def _register_new_face(self, face: tuple[np.ndarray, np.ndarray], name: str) -> None:
        person = self._create_person(name, [face], "enroll")
        self._log(t("log.enrolled", name=person.name), person=person.name)

    def _resolve_one_face(
        self,
        face: tuple[np.ndarray, np.ndarray],
        force_person_id: str | None,
        *,
        index: int,
        total: int,
    ) -> bool:
        settings = self.store.get()
        feature, thumb = face
        ranked = rank_people(feature, self.gallery.people())
        nearest = ranked[0] if ranked else None
        score = nearest.score if nearest else -1.0
        decision = decide_link(score, settings.match_threshold, settings.auto_link_same_person)
        forced = self.gallery.person(force_person_id) if force_person_id else None

        if total == 1 and forced is not None:
            self._apply_face(
                face,
                person_id=forced.id,
                source="manual",
                message=t("log.manual", name=forced.name),
            )
            return True

        if total == 1 and forced is None:
            if decision == "auto" and nearest is not None:
                self._apply_face(
                    face,
                    person_id=nearest.person.id,
                    source="auto",
                    message=t("log.auto", name=nearest.person.name, score=nearest.score),
                )
                return True
            if decision == "ask" and nearest is not None:
                dialog = SamePersonDialog(self, thumb, nearest)
                result = dialog.exec()
                if result == SamePersonDialog.SAME:
                    self._apply_face(
                        face,
                        person_id=nearest.person.id,
                        source="manual",
                        message=t("log.ask", name=nearest.person.name, score=nearest.score),
                    )
                    return True
                if result != SamePersonDialog.NEW:
                    return False
            default = nearest.person.name if nearest and nearest.score >= settings.match_threshold else ""
            name = self._ask_name(default)
            if name is None:
                return False
            self._register_new_face(face, name)
            return True

        show_match = nearest if nearest is not None and score >= settings.match_threshold else None
        if show_match is None and nearest is not None and decision == "ask":
            show_match = nearest
        default = nearest.person.name if nearest is not None and nearest.score >= settings.match_threshold else ""
        dialog = EnrollFaceDialog(
            self,
            thumb,
            index,
            total,
            match=None if forced is not None else show_match,
            forced_name=forced.name if forced is not None else None,
            default_name=default,
        )
        result = dialog.exec()
        if result == EnrollFaceDialog.SKIP:
            self._log(t("log.skip", index=index))
            return True
        if result == EnrollFaceDialog.SAME:
            if forced is not None:
                self._apply_face(
                    face,
                    person_id=forced.id,
                    source="manual",
                    message=t("log.added_forced", index=index, name=forced.name),
                )
                return True
            if show_match is not None:
                self._apply_face(
                    face,
                    person_id=show_match.person.id,
                    source="manual",
                    message=t("log.added_match", index=index, name=show_match.person.name, score=show_match.score),
                )
                return True
        if result == EnrollFaceDialog.NEW:
            self._register_new_face(face, dialog.chosen_name())
            return True
        self._log(t("log.cancel_rest"))
        return False

    def _enroll_image(self, bgr: np.ndarray, person_id: str | None = None) -> None:
        faces = self._extract_faces(bgr)
        if not faces:
            return
        total = len(faces)
        if total > 1:
            self._log(t("log.multi", total=total))
        for index, face in enumerate(faces, start=1):
            if not self._resolve_one_face(face, person_id, index=index, total=total):
                break
        self._reload_faces()

    def add_person_from_file(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, t("file.choose_face"), "", t("file.images"))
        if not paths:
            return
        for path in paths:
            image = _imread(path)
            if image is None:
                QMessageBox.warning(self, t("msg.cannot_enroll"), t("msg.cannot_read", name=Path(path).name))
                continue
            self._enroll_image(image)

    def add_person_from_camera(self) -> None:
        self._start_preview()
        dialog = CaptureDialog(self, self.monitor)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        frame = dialog.captured_bgr()
        if frame is None:
            QMessageBox.information(self, t("app.name"), t("msg.no_frame"))
            return
        self._enroll_image(frame)

    def add_sample(self, person_id: str) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, t("file.add_same"), "", t("file.images"))
        if not paths:
            return
        for path in paths:
            image = _imread(path)
            if image is None:
                QMessageBox.warning(self, t("msg.cannot_enroll"), t("msg.cannot_read", name=Path(path).name))
                continue
            self._enroll_image(image, person_id=person_id)

    def merge_person(self, person_id: str) -> None:
        others = [item for item in self.gallery.people() if item.id != person_id]
        if not others:
            QMessageBox.information(self, t("dialog.merge_title"), t("msg.merge_none"))
            return
        dialog = MergePersonDialog(self, self.gallery.people(), person_id)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        absorb_id = dialog.target_id()
        if not absorb_id:
            return
        keep = self.gallery.person(person_id)
        absorb = self.gallery.person(absorb_id)
        if keep is None or absorb is None:
            return
        self.gallery.merge_people(person_id, absorb_id)
        self._reload_faces()
        self._log(t("log.merged", absorb=absorb.name, keep=keep.name))

    def split_sample(self, person_id: str, sample_id: str) -> None:
        name = self._ask_name()
        if name is None:
            return
        try:
            created = self.gallery.split_sample(person_id, sample_id, name)
        except ValueError as exc:
            QMessageBox.information(self, t("msg.split_title"), str(exc))
            return
        self._reload_faces()
        self._log(t("log.split", name=created.name))

    def delete_sample(self, person_id: str, sample_id: str) -> None:
        person = self.gallery.person(person_id)
        if person is None:
            return
        last = len(person.samples) <= 1
        text = t("msg.delete_last") if last else t("msg.delete_photo_q")
        if QMessageBox.question(self, t("msg.delete_photo"), text) != QMessageBox.StandardButton.Yes:
            return
        self.gallery.remove_sample(person_id, sample_id)
        self._reload_faces()

    def rename_person(self, person_id: str, current: str) -> None:
        name = self._ask_name(current)
        if name is None:
            return
        self.gallery.rename(person_id, name)
        self._reload_faces()

    def delete_person(self, person_id: str) -> None:
        if QMessageBox.question(self, t("msg.delete_face"), t("msg.delete_face_q")) != QMessageBox.StandardButton.Yes:
            return
        self.gallery.remove_person(person_id)
        self._reload_faces()

    def browse_work_app(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, t("file.choose_app"), "", t("file.apps"))
        if not path:
            return
        self._add_work_app(Path(path).stem, path)

    def _add_work_app(self, name: str, path: str) -> None:
        settings = self.store.get()
        if any(os.path.normcase(app.path) == os.path.normcase(path) for app in settings.work_apps):
            return
        settings.work_apps.append(WorkApp(id=new_id(), name=name, path=path))
        self.store.replace(settings)
        self._reload_work(settings)

    def remove_work_app(self) -> None:
        item = self.work_list.currentItem()
        if not item:
            return
        app_id = item.data(Qt.ItemDataRole.UserRole)
        if not app_id:
            return
        settings = self.store.get()
        settings.work_apps = [app for app in settings.work_apps if app.id != app_id]
        self.store.replace(settings)
        self._reload_work(settings)

    def _move_work(self, delta: int) -> None:
        item = self.work_list.currentItem()
        if not item:
            return
        app_id = item.data(Qt.ItemDataRole.UserRole)
        settings = self.store.get()
        ids = [app.id for app in settings.work_apps]
        if app_id not in ids:
            return
        index = ids.index(app_id)
        target = index + delta
        if target < 0 or target >= len(settings.work_apps):
            return
        settings.work_apps[index], settings.work_apps[target] = settings.work_apps[target], settings.work_apps[index]
        self.store.replace(settings)
        self._reload_work(settings)
        self.work_list.setCurrentRow(target)

    def add_entertainment(self) -> None:
        name, ok = QInputDialog.getText(self, t("msg.ent_title"), t("msg.ent_prompt"))
        if not ok or not name.strip():
            return
        self._append_entertainment([name])

    def pick_entertainment(self) -> None:
        dialog = ProcessPicker(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._append_entertainment(dialog.selected())

    def _append_entertainment(self, names: list[str]) -> None:
        settings = self.store.get()
        merged = list(settings.entertainment_processes)
        for name in names:
            item = name.strip().lower()
            if item and item not in merged:
                merged.append(item)
        settings.entertainment_processes = merged
        self.store.replace(settings)
        self._reload_entertainment(settings)

    def remove_entertainment(self) -> None:
        item = self.ent_list.currentItem()
        if not item:
            return
        settings = self.store.get()
        settings.entertainment_processes = [name for name in settings.entertainment_processes if name != item.text()]
        self.store.replace(settings)
        self._reload_entertainment(settings)

    def manual_trigger(self) -> None:
        settings = self.store.get()
        try:
            actions = perform_switch(
                settings,
                protected_hwnds={self.hwnd()},
                protected_pids={os.getpid()},
                dry_run=settings.dev_mode,
            )
            key = "log.sim_dev" if settings.dev_mode else "log.sim"
            self._log(t(key, actions=_join_actions(actions)))
        except Exception as exc:  # noqa: BLE001
            self._log(t("log.sim_fail", error=exc))

    def reveal(self) -> None:
        self.setWindowState(self.windowState() & ~Qt.WindowState.WindowMinimized)
        self.showNormal()
        self.raise_()
        self.activateWindow()
        try:
            import ctypes

            hwnd = int(self.winId())
            ctypes.windll.user32.ShowWindow(hwnd, 9)
            ctypes.windll.user32.SetForegroundWindow(hwnd)
        except Exception:
            pass

    def _hide_to_tray(self, notify: bool = True) -> None:
        self.hide()
        if not notify or self._tray is None or not self._tray.isVisible() or self._tray_hint_shown:
            return
        self._tray_hint_shown = True
        self._tray.showMessage(
            t("app.name"),
            t("tray.hidden"),
            QSystemTrayIcon.MessageIcon.Information,
            2500,
        )

    def changeEvent(self, event: QEvent) -> None:
        if (
            event.type() == QEvent.Type.WindowStateChange
            and self.isMinimized()
            and not self.store.get().dev_mode
        ):
            QTimer.singleShot(0, self._hide_to_tray)
        super().changeEvent(event)

    def request_quit(self) -> None:
        self._allow_quit = True
        if self._tray is not None:
            menu = self._tray.contextMenu()
            if menu is not None:
                menu.close()
        QTimer.singleShot(0, self._quit_now)

    def shutdown(self) -> None:
        self.monitor.stop()
        self.monitor.wait(1500)

    def _quit_now(self) -> None:
        if self._quitting:
            return
        self._quitting = True
        self._allow_quit = True
        self.shutdown()
        if self._tray is not None:
            self._tray.hide()
            self._tray.setContextMenu(None)
        app = QApplication.instance()
        if app is not None:
            app.quit()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._quitting:
            event.accept()
            return
        if not self._allow_quit and not self.store.get().dev_mode:
            event.ignore()
            self._hide_to_tray()
            return
        event.accept()
        self._quit_now()
