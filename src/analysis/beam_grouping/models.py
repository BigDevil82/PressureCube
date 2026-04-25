from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.structure.enums import BeamType


@dataclass(frozen=True)
class BeamGroupConfig:
    n_groups_by_type: Dict[BeamType, int]
    stress_weight: float = 1.0
    displacement_weight: float = 0.7
    spatial_weight: float = 0.35
    symmetry_round_decimals: int = 3
    mirror_x: bool = True
    mirror_y: bool = False
    mirror_z: bool = True


@dataclass(frozen=True)
class BeamFeature:
    beam_id: int
    beam_type: BeamType
    orientation: str
    face: str
    midpoint: np.ndarray
    normalized_midpoint: np.ndarray
    length_norm: float
    stress_ratio: float
    stress_mpa: float
    displacement_max_mm: float
    displacement_mean_mm: float
    relative_displacement_mm: float
    symmetry_key: tuple


@dataclass(frozen=True)
class BeamGroupSummary:
    group_id: str
    beam_type: BeamType
    beam_count: int
    stress_ratio_mean: float
    stress_ratio_p90: float
    stress_ratio_max: float
    displacement_mean_mm: float
    displacement_p90_mm: float
    displacement_max_mm: float
    representative_center: tuple[float, float, float]
    symmetry_orbit_count: int
    control_factor: str


@dataclass(frozen=True)
class BeamGroupingResult:
    beam_to_group: Dict[int, str]
    group_summaries: list[BeamGroupSummary]
    features: Dict[int, BeamFeature]
    feature_weights: Dict[str, float]
    n_groups_by_type: Dict[BeamType, int]
