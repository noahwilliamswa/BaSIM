from __future__ import annotations

import math
from collections import deque

import numpy as np

from .domain import ConnectionConfig


class LeakyDelayConnection:
    """Delay a population vector, then integrate it with a first-order low-pass filter."""

    def __init__(self, width: int, dt_s: float, config: ConnectionConfig) -> None:
        if width <= 0:
            raise ValueError("width must be > 0")
        if dt_s <= 0.0:
            raise ValueError("dt_s must be > 0")
        if config.delay_s < 0.0 or config.tau_s < 0.0:
            raise ValueError("connection delay_s and tau_s must be >= 0")
        self.width = int(width)
        self.dt_s = float(dt_s)
        self.config = config
        self.delay_steps = max(0, int(round(config.delay_s / dt_s)))
        self._buffer: deque[np.ndarray] = deque(maxlen=self.delay_steps + 1)
        self.state = np.zeros(self.width, dtype=np.float64)

    def reset(self) -> None:
        self._buffer.clear()
        self.state.fill(0.0)

    def step(self, source: np.ndarray) -> np.ndarray:
        source = np.asarray(source, dtype=np.float64)
        if source.shape != self.state.shape:
            raise ValueError(f"source must have shape {self.state.shape}, got {source.shape}")
        self._buffer.append(source.copy())
        if len(self._buffer) <= self.delay_steps:
            delayed = np.zeros_like(self.state)
        else:
            delayed = self._buffer[0]

        target = self.config.gain * delayed
        if self.config.tau_s <= 0.0:
            self.state = target.copy()
        else:
            alpha = 1.0 - math.exp(-self.dt_s / self.config.tau_s)
            self.state += alpha * (target - self.state)
        return self.state.copy()
