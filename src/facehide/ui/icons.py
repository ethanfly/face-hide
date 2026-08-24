from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QApplication

from facehide.mark import (
    ICON_SIZES,
    STATUS_ALERT,
    STATUS_DEV,
    STATUS_IDLE,
    STATUS_WATCHING,
    render_mark,
)

COLOR_MUTED = "#aeb8c9"
COLOR_ACTIVE = "#9ec3ff"
COLOR_DEFAULT = "#e8eef8"
COLOR_ON_PRIMARY = "#ffffff"
COLOR_EMPTY = "#5a6780"

_icon_cache: dict[str, QIcon] = {}
_glyph_cache: dict[tuple[str, int, str, float], QPixmap] = {}


def app_icon(status: str = STATUS_IDLE) -> QIcon:
    if status not in _icon_cache:
        icon = QIcon()
        for size in ICON_SIZES:
            icon.addPixmap(_image_to_pixmap(render_mark(size, status)))
        _icon_cache[status] = icon
    return _icon_cache[status]


def app_pixmap(logical: int = 64, status: str = STATUS_IDLE) -> QPixmap:
    dpr = _device_pixel_ratio()
    px = max(1, int(round(logical * dpr)))
    pixmap = _image_to_pixmap(render_mark(px, status))
    pixmap.setDevicePixelRatio(dpr)
    return pixmap


def glyph_pixmap(name: str, size: int = 18, color: str = COLOR_DEFAULT) -> QPixmap:
    dpr = _device_pixel_ratio()
    key = (name, int(size), color, dpr)
    cached = _glyph_cache.get(key)
    if cached is not None:
        return cached
    drawer = _GLYPHS.get(name)
    if drawer is None:
        raise KeyError(f"unknown glyph: {name}")
    px = max(1, int(round(size * dpr)))
    pixmap = QPixmap(px, px)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    scale = px / 24.0
    painter.scale(scale, scale)
    pen = QPen(QColor(color), 1.85)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    drawer(painter)
    painter.end()
    pixmap.setDevicePixelRatio(dpr)
    _glyph_cache[key] = pixmap
    return pixmap


def glyph_icon(name: str, size: int = 18, color: str = COLOR_DEFAULT) -> QIcon:
    return QIcon(glyph_pixmap(name, size, color))


def tray_status(armed: bool, dev_mode: bool) -> str:
    if not armed:
        return STATUS_IDLE
    return STATUS_DEV if dev_mode else STATUS_WATCHING


def _device_pixel_ratio() -> float:
    app = QApplication.instance()
    if app is None:
        return 1.0
    return float(app.devicePixelRatio())


def _image_to_pixmap(image) -> QPixmap:
    rgba = image.convert("RGBA")
    qimage = QImage(
        rgba.tobytes("raw", "RGBA"),
        rgba.width,
        rgba.height,
        QImage.Format.Format_RGBA8888,
    )
    return QPixmap.fromImage(qimage.copy())


def _g_monitor(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(2.6, 7.4, 18.8, 13.0), 3.0, 3.0)
    p.drawEllipse(QRectF(8.6, 9.8, 6.8, 6.8))
    p.drawRoundedRect(QRectF(9.0, 4.3, 6.0, 3.4), 1.1, 1.1)


def _g_camera(p: QPainter) -> None:
    _g_monitor(p)


def _g_faces(p: QPainter) -> None:
    p.drawEllipse(QRectF(8.0, 3.2, 8.0, 8.0))
    path = QPainterPath()
    path.moveTo(4.8, 20.6)
    path.cubicTo(5.4, 14.6, 18.6, 14.6, 19.2, 20.6)
    p.drawPath(path)


def _g_work(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(3.2, 8.2, 17.6, 11.8), 2.2, 2.2)
    handle = QPainterPath()
    handle.moveTo(9.2, 8.2)
    handle.lineTo(9.2, 5.6)
    handle.lineTo(14.8, 5.6)
    handle.lineTo(14.8, 8.2)
    p.drawPath(handle)
    p.drawLine(QPointF(3.2, 13.2), QPointF(20.8, 13.2))


def _g_hide(p: QPainter) -> None:
    eye = QPainterPath()
    eye.moveTo(3.0, 12.0)
    eye.cubicTo(7.2, 6.2, 16.8, 6.2, 21.0, 12.0)
    eye.cubicTo(16.8, 17.8, 7.2, 17.8, 3.0, 12.0)
    p.drawPath(eye)
    p.drawEllipse(QRectF(9.6, 9.6, 4.8, 4.8))
    p.drawLine(QPointF(5.2, 18.8), QPointF(18.8, 5.2))


def _g_settings(p: QPainter) -> None:
    p.drawLine(QPointF(4.0, 7.0), QPointF(20.0, 7.0))
    p.drawLine(QPointF(4.0, 12.0), QPointF(20.0, 12.0))
    p.drawLine(QPointF(4.0, 17.0), QPointF(20.0, 17.0))
    color = p.pen().color()
    p.save()
    p.setBrush(color)
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(QRectF(13.4, 5.3, 3.4, 3.4))
    p.drawEllipse(QRectF(6.8, 10.3, 3.4, 3.4))
    p.drawEllipse(QRectF(11.4, 15.3, 3.4, 3.4))
    p.restore()


def _g_play(p: QPainter) -> None:
    path = QPainterPath()
    path.moveTo(8.0, 5.8)
    path.lineTo(8.0, 18.2)
    path.lineTo(18.4, 12.0)
    path.closeSubpath()
    p.drawPath(path)


def _g_stop(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(7.0, 7.0, 10.0, 10.0), 1.8, 1.8)


def _g_bolt(p: QPainter) -> None:
    path = QPainterPath()
    path.moveTo(13.4, 3.6)
    path.lineTo(7.4, 13.0)
    path.lineTo(11.8, 13.0)
    path.lineTo(10.4, 20.4)
    path.lineTo(16.8, 10.2)
    path.lineTo(12.4, 10.2)
    path.closeSubpath()
    p.drawPath(path)


def _g_upload(p: QPainter) -> None:
    p.drawLine(QPointF(12.0, 16.2), QPointF(12.0, 4.8))
    p.drawLine(QPointF(7.6, 9.0), QPointF(12.0, 4.8))
    p.drawLine(QPointF(16.4, 9.0), QPointF(12.0, 4.8))
    tray = QPainterPath()
    tray.moveTo(5.0, 16.4)
    tray.lineTo(5.0, 19.2)
    tray.lineTo(19.0, 19.2)
    tray.lineTo(19.0, 16.4)
    p.drawPath(tray)


def _g_window(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(4.0, 5.0, 16.0, 14.0), 2.2, 2.2)
    p.drawLine(QPointF(4.0, 9.0), QPointF(20.0, 9.0))
    p.drawEllipse(QRectF(6.2, 6.3, 1.7, 1.7))
    p.drawEllipse(QRectF(8.6, 6.3, 1.7, 1.7))


def _g_power(p: QPainter) -> None:
    p.drawArc(QRectF(5.0, 5.2, 14.0, 14.0), 135 * 16, 270 * 16)
    p.drawLine(QPointF(12.0, 4.4), QPointF(12.0, 12.0))


_GLYPHS: dict[str, Callable[[QPainter], None]] = {
    "monitor": _g_monitor,
    "camera": _g_camera,
    "faces": _g_faces,
    "work": _g_work,
    "hide": _g_hide,
    "settings": _g_settings,
    "play": _g_play,
    "stop": _g_stop,
    "bolt": _g_bolt,
    "upload": _g_upload,
    "window": _g_window,
    "power": _g_power,
}

__all__ = [
    "COLOR_ACTIVE",
    "COLOR_DEFAULT",
    "COLOR_EMPTY",
    "COLOR_MUTED",
    "COLOR_ON_PRIMARY",
    "STATUS_ALERT",
    "STATUS_DEV",
    "STATUS_IDLE",
    "STATUS_WATCHING",
    "app_icon",
    "app_pixmap",
    "glyph_icon",
    "glyph_pixmap",
    "tray_status",
]
