from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPixmap


def app_icon(size: int = 64) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor("#2f6fed"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRoundedRect(0, 0, size, size, size * 0.22, size * 0.22)

    painter.setBrush(QColor("#10131a"))
    screen = QRectF(size * 0.18, size * 0.2, size * 0.64, size * 0.42)
    painter.drawRoundedRect(screen, 4, 4)
    painter.setBrush(QColor("#7eb6ff"))
    painter.drawEllipse(QRectF(size * 0.36, size * 0.28, size * 0.28, size * 0.28))
    painter.setBrush(QColor("#10131a"))
    painter.drawEllipse(QRectF(size * 0.44, size * 0.36, size * 0.12, size * 0.12))

    slash = QPainterPath()
    slash.moveTo(size * 0.22, size * 0.78)
    slash.lineTo(size * 0.32, size * 0.70)
    slash.lineTo(size * 0.80, size * 0.22)
    slash.lineTo(size * 0.70, size * 0.30)
    slash.closeSubpath()
    painter.setBrush(QColor("#ff8b7b"))
    painter.drawPath(slash)
    painter.end()
    return QIcon(pixmap)
