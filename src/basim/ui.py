from __future__ import annotations

import sys

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .model import AdaptiveAttractorLayer, FrameRecord
from .presets import cat_demo


class LayerView(QWidget):
    def __init__(self, layer: AdaptiveAttractorLayer, title: str, parent=None):
        super().__init__(parent)
        self.layer = layer
        self.title = title
        self.setMinimumSize(330, 330)

    def _map(self, x: float, y: float) -> QPointF:
        margin = 48.0
        w = max(1.0, self.width() - margin * 2.0)
        h = max(1.0, self.height() - margin * 2.0)
        px = margin + ((x + 2.5) / 5.0) * w
        py = margin + ((2.5 - y) / 5.0) * h
        return QPointF(px, py)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor("#f7f7f4"))

        p.setPen(QPen(QColor("#1f2328"), 1))
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        p.setFont(font)
        p.drawText(16, 24, self.title)

        centers = [a.center for a in self.layer.config.attractors]
        depths = self.layer.depths
        activations = self.layer.activations
        max_depth = max(float(depths.max()), 1e-6)
        max_activation = float(max(activations))

        for i, (spec, center) in enumerate(
            zip(self.layer.config.attractors, centers)
        ):
            pt = self._map(center[0], center[1])
            depth_n = float(depths[i]) / max_depth
            act = float(activations[i])
            radius = 22.0 + 25.0 * depth_n
            alpha = int(65 + 150 * act)
            p.setBrush(QColor(83, 128, 196, alpha))
            p.setPen(QPen(QColor(45, 70, 105, 180), 1.2))
            p.drawEllipse(pt, radius, radius)

            label_font = QFont()
            label_font.setPointSize(9)
            label_font.setBold(abs(act - max_activation) < 1e-9)
            p.setFont(label_font)
            p.setPen(QPen(QColor("#20242a"), 1))
            p.drawText(
                QRectF(pt.x() - 55, pt.y() - 9, 110, 18),
                Qt.AlignCenter,
                spec.label,
            )

        state = self._map(float(self.layer.state[0]), float(self.layer.state[1]))
        p.setBrush(QColor("#111111"))
        p.setPen(Qt.NoPen)
        p.drawEllipse(state, 6.0, 6.0)

        p.setPen(QPen(QColor(0, 0, 0, 40), 1))
        p.drawRect(self.rect().adjusted(8, 36, -8, -8))


class CompetitionView(QWidget):
    def __init__(self, labels: tuple[str, ...], parent=None):
        super().__init__(parent)
        self.labels = labels
        self.values = [1.0 / len(labels)] * len(labels)
        self.setMinimumWidth(240)

    def set_values(self, values) -> None:
        self.values = list(values)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor("#ffffff"))

        title_font = QFont()
        title_font.setPointSize(11)
        title_font.setBold(True)
        p.setFont(title_font)
        p.setPen(QPen(QColor("#1f2328"), 1))
        p.drawText(16, 24, "Lexical competition")

        top = 54
        row_h = 46
        left = 54
        right = self.width() - 18
        bar_w = max(20, right - left)
        for i, (label, value) in enumerate(zip(self.labels, self.values)):
            y = top + i * row_h
            p.setPen(QPen(QColor("#3b4048"), 1))
            p.drawText(14, y + 17, label)
            p.setBrush(QColor("#eef1f4"))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(QRectF(left, y, bar_w, 22), 5, 5)
            p.setBrush(QColor("#5b7fb3"))
            p.drawRoundedRect(
                QRectF(left, y, bar_w * float(value), 22),
                5,
                5,
            )
            p.setPen(QPen(QColor("#21262d"), 1))
            p.drawText(left + 6, y + 16, f"{float(value):.2f}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BaSIM — adaptive speech attractor demo")
        self.resize(1280, 760)

        self.bundle = cat_demo()
        self.sim = self.bundle.simulation
        self.timer = QTimer(self)
        self.timer.setInterval(int(self.sim.dt_ms))
        self.timer.timeout.connect(self._tick)

        self.time_label = QLabel()
        self.time_label.setMinimumWidth(170)
        self.observed_label = QLabel("phonemes: —")

        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self._toggle)
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._reset)
        step_btn = QPushButton("Step 49 ms")
        step_btn.clicked.connect(self._step_once)

        controls = QHBoxLayout()
        controls.addWidget(self.run_btn)
        controls.addWidget(step_btn)
        controls.addWidget(reset_btn)
        controls.addSpacing(18)
        controls.addWidget(self.time_label)
        controls.addWidget(self.observed_label)
        controls.addStretch(1)

        self.acoustic_view = LayerView(
            self.sim.acoustic,
            "1 · acoustic feature field",
        )
        self.phoneme_view = LayerView(
            self.sim.phoneme,
            "2 · phoneme field",
        )
        self.lexical_view = LayerView(
            self.sim.lexical,
            "3 · lexical field",
        )
        self.competition = CompetitionView(self.sim.lexical.labels)

        views = QHBoxLayout()
        views.setSpacing(8)
        views.addWidget(self.acoustic_view, 1)
        views.addWidget(self.phoneme_view, 1)
        views.addWidget(self.lexical_view, 1)
        views.addWidget(self.competition, 0)

        note = QLabel(
            "Synthetic 1 s /k æ t/ input. The network continues evolving to 5 s. "
            "Circle size reflects current basin depth; the black point is layer state. "
            "Coordinates are latent explanatory coordinates, not anatomy."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#555; padding:4px 2px 8px 2px;")

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addLayout(controls)
        layout.addLayout(views, 1)
        layout.addWidget(note)
        self.setCentralWidget(root)

        self.setStyleSheet(
            "QMainWindow, QWidget { background:#f7f7f4; color:#1f2328; }"
            "QPushButton { background:white; border:1px solid #c9cdd2; "
            "border-radius:5px; padding:6px 12px; }"
            "QPushButton:hover { border-color:#7d8792; }"
        )
        self._refresh_status(None)

    def _toggle(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.run_btn.setText("Run")
            return
        if self.sim.finished:
            self._reset()
        self.timer.start()
        self.run_btn.setText("Pause")

    def _reset(self) -> None:
        self.timer.stop()
        self.run_btn.setText("Run")
        self.sim.reset()
        self.competition.set_values(self.sim.lexical.activations)
        self._refresh_status(None)
        self._update_views()

    def _step_once(self) -> None:
        if not self.sim.finished:
            record = self.sim.step()
            self._refresh_status(record)
            self._update_views()

    def _tick(self) -> None:
        if self.sim.finished:
            self.timer.stop()
            self.run_btn.setText("Run")
            self._refresh_status(
                self.sim.records[-1] if self.sim.records else None
            )
            return
        record = self.sim.step()
        self._refresh_status(record)
        self._update_views()

    def _update_views(self) -> None:
        self.acoustic_view.update()
        self.phoneme_view.update()
        self.lexical_view.update()
        self.competition.set_values(self.sim.lexical.activations)

    def _refresh_status(self, record: FrameRecord | None) -> None:
        t = 0.0 if record is None else record.t_ms
        self.time_label.setText(
            f"t = {t:6.0f} ms / {self.sim.total_ms:.0f} ms"
        )
        observed = (
            "—"
            if record is None or not record.observed_phonemes
            else " → ".join(record.observed_phonemes)
        )
        self.observed_label.setText(f"phonemes: {observed}")


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    raise SystemExit(app.exec())
