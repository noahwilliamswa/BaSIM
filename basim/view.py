from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QBrush
from PySide6.QtWidgets import QWidget

from .attractor import CompetitiveAttractorLayer
from .domain import SimulationSnapshot


class SpectrogramView(QWidget):
    """Compact time × frequency display of the exact spectrum driving the model."""

    def __init__(self, frequency_labels: tuple[str, ...], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.frequency_labels = frequency_labels
        self.history: list[SimulationSnapshot] = []
        self.total_s = 5.0
        self.setMinimumHeight(150)
        self.setMaximumHeight(190)

    def set_history(self, history: list[SimulationSnapshot], total_s: float) -> None:
        self.history = history
        self.total_s = max(float(total_s), 1e-6)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        p.fillRect(self.rect(), QColor("#111318"))

        title = QFont()
        title.setPointSize(10)
        title.setBold(True)
        p.setFont(title)
        p.setPen(QColor("#f0f2f5"))
        p.drawText(12, 20, "Input spectrogram · model frequency bands")

        left, top, right, bottom = 64.0, 30.0, 14.0, 24.0
        plot = QRectF(left, top, max(20.0, self.width() - left - right), max(20.0, self.height() - top - bottom))
        p.fillRect(plot, QColor("#090a0d"))

        n_bands = len(self.frequency_labels)
        if n_bands:
            band_h = plot.height() / n_bands
            for band in range(n_bands):
                label_i = n_bands - 1 - band
                y = plot.top() + band * band_h
                p.setPen(QColor(165, 173, 185))
                small = QFont(); small.setPointSize(7); p.setFont(small)
                p.drawText(QRectF(3, y, left - 8, band_h), Qt.AlignRight | Qt.AlignVCenter, self.frequency_labels[label_i])

            for snap in self.history:
                x0 = plot.left() + (snap.time_s / self.total_s) * plot.width()
                x1 = plot.left() + ((snap.time_s + 0.049) / self.total_s) * plot.width()
                cell_w = max(1.0, x1 - x0 + 0.5)
                for band, value in enumerate(snap.spectrum):
                    row = n_bands - 1 - band
                    y = plot.top() + row * band_h
                    v = float(np.clip(value, 0.0, 1.0))
                    color = QColor(int(22 + 220 * v), int(38 + 145 * v), int(65 + 95 * (1.0 - v)))
                    p.fillRect(QRectF(x0, y, cell_w, band_h + 0.5), color)

        p.setPen(QPen(QColor(255, 255, 255, 55), 1))
        p.drawRect(plot)
        p.setPen(QColor(150, 158, 170))
        small = QFont(); small.setPointSize(7); p.setFont(small)
        p.drawText(int(plot.left()), self.height() - 6, "0 s")
        p.drawText(int(plot.right() - 26), self.height() - 6, f"{self.total_s:.0f} s")


class AttractorView(QWidget):
    """2D view of a layer's changing potential field and leading candidates."""

    def __init__(self, layer: CompetitiveAttractorLayer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.layer = layer
        self.setMinimumSize(390, 430)
        self._rgb: np.ndarray | None = None

    def set_layer(self, layer: CompetitiveAttractorLayer) -> None:
        self.layer = layer
        self.update()

    def _heat_image(self) -> QImage:
        potential = self.layer.potential_grid(112)
        lo, hi = float(np.min(potential)), float(np.max(potential))
        depth = np.zeros_like(potential) if hi - lo < 1e-12 else 1.0 - (potential - lo) / (hi - lo)
        rgb = np.empty((*depth.shape, 3), dtype=np.uint8)
        rgb[..., 0] = np.clip(28 + 212 * depth, 0, 255)
        rgb[..., 1] = np.clip(42 + 145 * (1.0 - np.abs(depth - 0.55)), 0, 255)
        rgb[..., 2] = np.clip(72 + 150 * (1.0 - depth), 0, 255)
        self._rgb = np.ascontiguousarray(rgb)
        h, w, _ = self._rgb.shape
        return QImage(self._rgb.data, w, h, 3 * w, QImage.Format_RGB888)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor("#111318"))
        margin, title_h = 16, 34
        size = min(self.width() - 2 * margin, self.height() - 150)
        rect = QRectF(margin, title_h + 4, size, size)
        font = QFont(); font.setPointSize(11); font.setBold(True)
        p.setFont(font)
        p.setPen(QColor("#f0f2f5"))
        p.drawText(margin, 24, self.layer.name)
        p.drawImage(rect, self._heat_image())
        p.setPen(QPen(QColor(255, 255, 255, 55), 1))
        p.drawRect(rect)

        max_depth = max(float(np.max(self.layer.depths, initial=1.0)), 1e-9)
        for i, (center, label) in enumerate(zip(self.layer.centers, self.layer.labels, strict=True)):
            x = rect.left() + center[0] * rect.width()
            y = rect.top() + center[1] * rect.height()
            activation = float(self.layer.activation[i])
            radius = 4.0 + 7.0 * float(self.layer.depths[i]) / max_depth
            p.setBrush(QBrush(QColor(250, 250, 250, int(70 + 170 * activation))))
            p.setPen(QPen(QColor(10, 10, 12, 190), 1))
            p.drawEllipse(QRectF(x - radius, y - radius, 2 * radius, 2 * radius))
            small = QFont(); small.setPointSize(7); p.setFont(small)
            p.setPen(QColor(245, 245, 245, 220))
            p.drawText(QRectF(x - 38, y + radius + 2, 76, 16), Qt.AlignHCenter, label)

        px = rect.left() + self.layer.position[0] * rect.width()
        py = rect.top() + self.layer.position[1] * rect.height()
        p.setBrush(QBrush(QColor("#ffffff")))
        p.setPen(QPen(QColor("#0a0a0c"), 2))
        p.drawEllipse(QRectF(px - 5, py - 5, 10, 10))
        self._draw_rankings(p, rect.bottom() + 22, margin)

    def _draw_rankings(self, p: QPainter, y0: float, margin: int) -> None:
        bar_left = margin + 74
        bar_width = max(80, self.width() - bar_left - margin - 34)
        tiny = QFont(); tiny.setPointSize(8); p.setFont(tiny)
        for row, idx in enumerate(np.argsort(self.layer.activation)[::-1][:3]):
            value = float(self.layer.activation[int(idx)])
            y = y0 + row * 24
            p.setPen(QColor("#d9dde4"))
            p.drawText(margin, int(y + 12), self.layer.labels[int(idx)])
            p.fillRect(QRectF(bar_left, y, bar_width, 10), QColor("#292d35"))
            p.fillRect(QRectF(bar_left, y, bar_width * value, 10), QColor("#e8edf5"))
            p.drawText(int(bar_left + bar_width + 4), int(y + 10), f"{value:.2f}")
