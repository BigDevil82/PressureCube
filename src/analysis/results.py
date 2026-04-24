from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class AnalysisResults:
    """单次静力分析的原始结果集合。"""

    displacements: Dict[int, np.ndarray] = field(default_factory=dict)
    beam_forces: Dict[int, np.ndarray] = field(default_factory=dict)
    plate_stresses: Dict[int, np.ndarray] = field(default_factory=dict)
    reactions: Dict[int, np.ndarray] = field(default_factory=dict)

    def __add__(self, other: object) -> "AnalysisResults | None":
        if not isinstance(other, AnalysisResults):
            return None

        result = AnalysisResults()
        for key in set(self.displacements) & set(other.displacements):
            result.displacements[key] = self.displacements[key] + other.displacements[key]
        for key in set(self.beam_forces) & set(other.beam_forces):
            result.beam_forces[key] = self.beam_forces[key] + other.beam_forces[key]
        for key in set(self.plate_stresses) & set(other.plate_stresses):
            result.plate_stresses[key] = self.plate_stresses[key] + other.plate_stresses[key]
        for key in set(self.reactions) & set(other.reactions):
            result.reactions[key] = self.reactions[key] + other.reactions[key]
        return result

    def scale(self, factor: float) -> "AnalysisResults":
        result = AnalysisResults()
        result.displacements = {key: value * factor for key, value in self.displacements.items()}
        result.beam_forces = {key: value * factor for key, value in self.beam_forces.items()}
        result.plate_stresses = {key: value * factor for key, value in self.plate_stresses.items()}
        result.reactions = {key: value * factor for key, value in self.reactions.items()}
        return result


@dataclass
class ModalAnalysisResults:
    """模态分析结果。"""

    frequencies_hz: list[float]
    periods_s: list[float]
    mode_shapes: Dict[int, Dict[int, np.ndarray]] = field(default_factory=dict)
