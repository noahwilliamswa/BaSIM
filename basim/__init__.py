"""BaSIM: layered attractor dynamics for a speech-perception explainer."""

from .engine import BaSimEngine
from .stimuli import ambiguous_onset_stimulus, default_stimulus
from .types import AcousticFrame, SimulationConfig, StimulusProgram, StimulusSegment

__all__ = [
    "AcousticFrame", "BaSimEngine", "SimulationConfig", "StimulusProgram", "StimulusSegment",
    "ambiguous_onset_stimulus", "default_stimulus",
]
