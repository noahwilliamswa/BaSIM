"""Public model API. Implementation is split across focused modules."""

from .connections import LeakyDelayConnection
from .engine import BaSimEngine
from .stimuli import ambiguous_onset_stimulus, default_stimulus
from .domain import (
    AcousticFrame,
    ConnectionConfig,
    SimulationConfig,
    SpectrumFrame,
    StimulusProgram,
    StimulusSegment,
)

__all__ = [
    "AcousticFrame",
    "BaSimEngine",
    "ConnectionConfig",
    "LeakyDelayConnection",
    "SimulationConfig",
    "SpectrumFrame",
    "StimulusProgram",
    "StimulusSegment",
    "ambiguous_onset_stimulus",
    "default_stimulus",
]
