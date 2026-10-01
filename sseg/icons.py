"""Small vector-drawn toolbar icons (no external image files or emoji fonts needed)."""

from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QBrush, QColor, QIcon, QPainter, QPixmap


def make_hand_icon(size=32):
    """A simple solid black open-hand silhouette used for the Pan tool."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.scale(size / 32.0, size / 32.0)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(QColor(0, 0, 0)))

    # Four fingers (left to right), different lengths
    for x, top in ((8.0, 6.0), (11.7, 3.0), (15.4, 2.5), (19.1, 5.0)):
        painter.drawRoundedRect(QRectF(x, top, 3.2, 20.0 - top), 1.6, 1.6)

    # Thumb, angled out from the palm
    painter.save()
    painter.translate(11.0, 24.0)
    painter.rotate(45)
    painter.drawRoundedRect(QRectF(-8.5, -2.4, 10.0, 4.8), 2.4, 2.4)
    painter.restore()

    # Palm
    painter.drawRoundedRect(QRectF(8.0, 14.0, 15.0, 14.0), 5.0, 5.0)
    painter.end()
    return QIcon(pixmap)
