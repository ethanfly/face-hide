from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QWidget

from facehide.runtime import asset_path

_CHECK_ICON = asset_path("check.png").as_posix()

PAGE_BG = "#0b0f16"
CARD_BG = "#141a24"
SIDE_BG = "#10151e"

_BASE_QSS = """
QWidget {
    color: #e8eef8;
    font-family: "Microsoft YaHei UI", "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}
QMainWindow, QDialog, QWidget#Root, QStackedWidget, QStackedWidget > QWidget, QWidget#Page {
    background: #0b0f16;
}
QDialog {
    background: #10151e;
}
QFrame#Sidebar {
    background: #10151e;
    border-right: 1px solid #232b39;
}
QLabel#BrandLogo {
    background: transparent;
    border: 0;
}
QLabel#BrandMark {
    color: #8eb6ff;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    background: transparent;
}
QLabel#Brand {
    color: #f6f8fc;
    font-size: 20px;
    font-weight: 700;
    background: transparent;
}
QLabel#BrandSub {
    color: #8b95a8;
    font-size: 11px;
    background: transparent;
}
QLabel#Eyebrow {
    color: #6f8cff;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.4px;
    background: transparent;
}
QPushButton#Nav {
    background: transparent;
    color: #aeb8c9;
    border: 0;
    border-radius: 10px;
    padding: 9px 12px 9px 10px;
    text-align: left;
}
QPushButton#Nav:hover {
    background: #1a2230;
    color: #ffffff;
}
QPushButton#Nav:checked {
    background: #1b2c48;
    color: #9ec3ff;
    border-left: 3px solid #6f8cff;
    padding-left: 7px;
}
QPushButton#Lang {
    background: #121826;
    color: #9aa4b5;
    border: 1px solid #2a3344;
    border-radius: 8px;
    padding: 7px 10px;
}
QPushButton#Lang:hover {
    color: #ffffff;
    border-color: #3d4b63;
}
QPushButton#Lang:checked {
    background: #1d2d4d;
    color: #d7e6ff;
    border: 1px solid #4d74c9;
}
QFrame#Card, QFrame#EmptyFace, QFrame#PreviewShell, QFrame#SettingsCard {
    background: #141a24;
    border: 1px solid #273143;
    border-radius: 16px;
}
QFrame#Card[muted="true"] {
    border: 1px solid #5a4a28;
    background: #121720;
}
QFrame#EmptyFace {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #141a24, stop:1 #182033);
}
QLabel#Title {
    font-size: 22px;
    font-weight: 700;
    color: #f6f8fc;
    background: transparent;
}
QLabel#Hint {
    color: #8b95a8;
    background: transparent;
}
QLabel#Section {
    color: #c5cede;
    font-weight: 600;
    background: transparent;
}
QLabel#Pill {
    padding: 4px 10px;
    border-radius: 999px;
    background: #1c2432;
    color: #c5cede;
}
QLabel#Pill[state="on"] {
    background: #143528;
    color: #62d89a;
}
QLabel#Pill[state="alert"] {
    background: #3a1d1b;
    color: #ff8b7b;
}
QLabel#Pill[state="warn"] {
    background: #3a2d16;
    color: #f3c16b;
}
QPushButton {
    background: #222a3a;
    color: #e8eef8;
    border: 1px solid #354257;
    border-radius: 10px;
    padding: 8px 14px;
}
QPushButton:hover {
    background: #2b3548;
    border-color: #45536b;
}
QPushButton:pressed {
    background: #1b2230;
}
QPushButton:disabled {
    color: #6d7687;
    background: #171c26;
    border-color: #262d3a;
}
QPushButton#Primary {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4d80f5, stop:1 #3a6ae0);
    border: 1px solid #5b8cff;
    color: white;
    font-weight: 600;
}
QPushButton#Primary:hover {
    background: #5b8cff;
}
QPushButton#Danger {
    background: #7d2d2b;
    border: 1px solid #b0433f;
    color: #ffe8e6;
}
QPushButton#Danger:hover {
    background: #9a3834;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget, QPlainTextEdit {
    background: #0e131c;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 8px 10px;
    selection-background-color: #3a6ae0;
    selection-color: #ffffff;
}
QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus, QListWidget:focus {
    border: 1px solid #4d74c9;
}
QComboBox::drop-down {
    border: 0;
    width: 22px;
}
QComboBox QAbstractItemView {
    background: #141a24;
    border: 1px solid #2d3748;
    selection-background-color: #1b2c48;
    outline: 0;
}
QListWidget {
    padding: 6px;
    outline: 0;
}
QListWidget::item {
    padding: 10px 12px;
    border-radius: 10px;
    margin: 2px 0;
}
QListWidget::item:hover {
    background: #1a2230;
}
QListWidget::item:selected {
    background: #1b2c48;
    color: #d7e6ff;
}
QPlainTextEdit {
    font-family: "Cascadia Mono", "Consolas", "Microsoft YaHei UI", monospace;
    font-size: 12px;
    color: #c5cede;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #243044;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #5b8cff;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 16px;
    margin: -6px 0;
    border-radius: 8px;
    background: #d7e6ff;
    border: 2px solid #5b8cff;
}
QSlider::handle:horizontal:hover {
    background: #ffffff;
}
QCheckBox {
    spacing: 10px;
    background: transparent;
    padding: 4px 0;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 5px;
    border: 1px solid #3a4558;
    background: #121826;
}
QCheckBox::indicator:hover {
    border: 1px solid #5b8cff;
}
QCheckBox::indicator:checked {
    background: #3a6ae0;
    border: 1px solid #5b8cff;
    image: url("__CHECK_URL__");
}
QCheckBox::indicator:checked:hover {
    background: #4d80f5;
}
QCheckBox::indicator:disabled {
    background: #171c26;
    border: 1px solid #262d3a;
}
QRadioButton {
    spacing: 8px;
    background: transparent;
}
QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border-radius: 8px;
    border: 1px solid #3a4558;
    background: #121826;
}
QRadioButton::indicator:hover {
    border: 1px solid #5b8cff;
}
QRadioButton::indicator:checked {
    background: #3a6ae0;
    border: 1px solid #5b8cff;
}
QScrollArea, QScrollArea > QWidget, QAbstractScrollArea::viewport, QWidget#FaceHost {
    border: 0;
    background: #0b0f16;
}
QLabel#Thumb {
    background: #0c1118;
    border: 1px solid #2a3140;
    border-radius: 10px;
}
QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 4px 2px;
    border: 0;
}
QScrollBar:horizontal {
    background: transparent;
    height: 8px;
    margin: 2px 4px;
    border: 0;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: rgba(126, 166, 220, 0.28);
    border-radius: 4px;
    min-height: 28px;
    min-width: 28px;
}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
    background: rgba(155, 190, 255, 0.55);
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
    height: 0;
    background: transparent;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: transparent;
}
QMenu {
    background: #141a24;
    border: 1px solid #2a3140;
    color: #e8eef8;
    padding: 8px;
    border-radius: 8px;
}
QMenu::item {
    padding: 8px 14px;
    border-radius: 4px;
}
QMenu::item:selected {
    background: #1b2c48;
}
QMenu::separator {
    height: 1px;
    background: #273143;
    margin: 4px 8px;
}
QMenu#TrayMenu {
    border-radius: 0;
    padding: 4px;
}
QMenu#TrayMenu::item {
    padding: 8px 14px 8px 10px;
    border-radius: 4px;
}
QToolTip {
    background: #141a24;
    color: #e8eef8;
    border: 1px solid #2a3140;
    padding: 6px 8px;
    border-radius: 6px;
}
QProgressDialog, QProgressBar {
    background: #10151e;
}
QProgressBar {
    border: 1px solid #2a3140;
    border-radius: 6px;
    text-align: center;
}
QProgressBar::chunk {
    background: #5b8cff;
    border-radius: 5px;
}
"""

APP_QSS = _BASE_QSS.replace("__CHECK_URL__", _CHECK_ICON)


def apply_dark_surface(widget: QWidget) -> None:
    widget.setAutoFillBackground(True)
    palette = widget.palette()
    color = QColor(PAGE_BG)
    palette.setColor(QPalette.ColorRole.Window, color)
    palette.setColor(QPalette.ColorRole.Base, color)
    palette.setColor(QPalette.ColorRole.Button, QColor(CARD_BG))
    palette.setColor(QPalette.ColorRole.Text, QColor("#e8eef8"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#e8eef8"))
    widget.setPalette(palette)
