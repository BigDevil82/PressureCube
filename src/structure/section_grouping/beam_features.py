from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.analysis.metrics import beam_displacement_metric, beam_stress_metric
from src.analysis.results import AnalysisResults
from src.structure.enums import BeamType
from src.structure.structural_model import StructuralModel


@dataclass(frozen=True)
class BeamFeature:
    """单根梁用于发现稳定截面分组的特征。"""

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


def extract_beam_features(
    model: StructuralModel,
    strength_results: AnalysisResults,
    service_results: AnalysisResults,
    *,
    mirror_x: bool,
    mirror_y: bool,
    mirror_z: bool,
    symmetry_round_decimals: int,
) -> Dict[int, BeamFeature]:
    """提取发现阶段使用的梁特征。

    应力和位移来自 FEA 结果，只用于离线发现分组；优化内层使用的组别分配
    只依赖几何原型，不依赖这些结果。
    """

    bounds_min, bounds_max = _model_bounds(model)
    span = np.maximum(bounds_max - bounds_min, 1e-9)
    length_ref = float(np.linalg.norm(span))

    features: Dict[int, BeamFeature] = {}
    for beam_id, beam in model.beams.items():
        if beam.is_virtual:
            continue

        p1 = model.points[beam.start_node].coordinates()
        p2 = model.points[beam.end_node].coordinates()
        midpoint = (p1 + p2) / 2.0
        normalized_midpoint = (midpoint - bounds_min) / span
        length_norm = float(np.linalg.norm(p2 - p1) / length_ref)
        orientation = beam_orientation(p1, p2)
        stress = beam_stress_metric(beam, strength_results.beam_forces.get(beam_id))
        displacement = beam_displacement_metric(beam, service_results)
        symmetry_key = _symmetry_key(
            beam.beam_type,
            orientation,
            beam.on_face,
            normalized_midpoint,
            mirror_x=mirror_x,
            mirror_y=mirror_y,
            mirror_z=mirror_z,
            decimals=symmetry_round_decimals,
        )

        features[beam_id] = BeamFeature(
            beam_id=beam_id,
            beam_type=beam.beam_type,
            orientation=orientation,
            face=beam.on_face,
            midpoint=midpoint,
            normalized_midpoint=normalized_midpoint,
            length_norm=length_norm,
            stress_ratio=stress.ratio,
            stress_mpa=stress.stress_mpa,
            displacement_max_mm=displacement.max_mm,
            displacement_mean_mm=displacement.mean_mm,
            relative_displacement_mm=displacement.relative_mm,
            symmetry_key=symmetry_key,
        )
    return features


def group_features_by_symmetry(features: Dict[int, BeamFeature]) -> Dict[tuple, list[BeamFeature]]:
    grouped = defaultdict(list)
    for feature in features.values():
        grouped[feature.symmetry_key].append(feature)
    return dict(grouped)


def beam_orientation(p1: np.ndarray, p2: np.ndarray) -> str:
    diff = np.abs(p2 - p1)
    axis = int(np.argmax(diff))
    return ("x", "y", "z")[axis]


def _model_bounds(model: StructuralModel) -> tuple[np.ndarray, np.ndarray]:
    coords = np.array([point.coordinates() for point in model.points.values()])
    return coords.min(axis=0), coords.max(axis=0)


def _symmetry_key(
    beam_type: BeamType,
    orientation: str,
    face: str,
    normalized_midpoint: np.ndarray,
    *,
    mirror_x: bool,
    mirror_y: bool,
    mirror_z: bool,
    decimals: int,
) -> tuple:
    x, y, z = normalized_midpoint
    if mirror_x:
        x = min(x, 1.0 - x)
    if mirror_y:
        y = min(y, 1.0 - y)
    if mirror_z:
        z = min(z, 1.0 - z)
    face_family = _face_family(face)
    return (
        beam_type.name,
        orientation,
        face_family,
        round(float(x), decimals),
        round(float(y), decimals),
        round(float(z), decimals),
    )


def _face_family(face: str) -> str:
    parts = set(str(face).split("/"))
    if "top" in parts or "bottom" in parts:
        return "horizontal"
    if "front" in parts or "back" in parts:
        return "longitudinal_side"
    if "left" in parts or "right" in parts:
        return "end_side"
    return str(face)
