from __future__ import annotations

import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QSlider, QVBoxLayout, QWidget,
)

from .engine import BaSimEngine
from .stimuli import ambiguous_onset_stimulus, default_stimulus
from .types import SimulationConfig
from .view import AttractorView


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BaSIM — layered attractor speech explainer")
        self.resize(1360, 610)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        self.preset = QComboBox()
        self.preset.addItems(["BAT", "Ambiguous B/P onset"])
        self.preset.currentIndexChanged.connect(self._reset)
        self.feedback = QCheckBox("lexical feedback")
        self.feedback.setChecked(True)
        self.feedback.stateChanged.connect(self._reset)
        self.feedback_slider = QSlider(Qt.Horizontal)
        self.feedback_slider.setRange(0, 35)
        self.feedback_slider.setValue(16)
        self.feedback_slider.setFixedWidth(120)
        self.feedback_slider.valueChanged.connect(self._feedback_changed)
        self.feedback_value = QLabel("0.16")
        self.feedback_value.setFixedWidth(34)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("stimulus"))
        controls.addWidget(self.preset)
        controls.addSpacing(12)
        controls.addWidget(self.feedback)
        controls.addWidget(self.feedback_slider)
        controls.addWidget(self.feedback_value)
        controls.addStretch(1)
        for text, fn in [("Play", self._play), ("Pause", self._pause), ("Step", self._tick), ("Reset", self._reset)]:
            button = QPushButton(text)
            button.clicked.connect(fn)
            controls.addWidget(button)

        self.status = QLabel()
        self.status.setStyleSheet("font-family: monospace; color: #cdd3dc;")
        self.note = QLabel("1.0 s acoustic event · 5.0 s simulation · 49 ms model step · continuous propagation")
        self.note.setStyleSheet("color: #8f98a6;")
        self.engine = self._make_engine()
        self.views = [
            AttractorView(self.engine.feature),
            AttractorView(self.engine.phoneme),
            AttractorView(self.engine.lexical),
        ]
        row = QHBoxLayout()
        row.setSpacing(8)
        for view in self.views:
            row.addWidget(view, 1)

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)
        layout.addLayout(controls)
        layout.addWidget(self.status)
        layout.addLayout(row, 1)
        layout.addWidget(self.note)
        self.setCentralWidget(body)
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #0d0f13; color: #e8ebef; }
            QPushButton, QComboBox {
                background: #1a1e25;
                border: 1px solid #363d48;
                border-radius: 5px;
                padding: 5px 9px;
            }
            QPushButton:hover, QComboBox:hover { border-color: #697484; }
        """)
        self._refresh()

    def _make_engine(self) -> BaSimEngine:
        stimulus = default_stimulus() if self.preset.currentIndex() == 0 else ambiguous_onset_stimulus()
        gain = self.feedback_slider.value() / 100.0 if self.feedback.isChecked() else 0.0
        return BaSimEngine(SimulationConfig(0.049, 5.0, gain), stimulus)

    def _feedback_changed(self, value: int) -> None:
        self.feedback_value.setText(f"{value / 100.0:.2f}")
        self._reset()

    def _play(self) -> None:
        if self.engine.done:
            self._reset()
        self.timer.start(int(round(self.engine.config.dt_s * 1000.0)))

    def _pause(self) -> None:
        self.timer.stop()

    def _reset(self, *_args) -> None:
        self.timer.stop()
        self.engine = self._make_engine()
        for view, layer in zip(
            self.views,
            [self.engine.feature, self.engine.phoneme, self.engine.lexical],
            strict=True,
        ):
            view.set_layer(layer)
        self._refresh()

    def _tick(self) -> None:
        if self.engine.done:
            self.timer.stop()
        else:
            self.engine.step()
        self._refresh()

    def _refresh(self) -> None:
        snap = self.engine.snapshot()
        ph = self.engine.top_candidates("phoneme", 1)[0][0]
        word = self.engine.top_candidates("lexical", 1)[0][0]
        self.status.setText(
            f"t={snap.time_s:0.3f}s / 5.0s   input={snap.stimulus_label:<7}   "
            f"phoneme≈{ph:<3}   word≈{word}"
        )
        for view in self.views:
            view.update()


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
