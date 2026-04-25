from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict

import numpy as np

from src.analysis.beam_grouping.models import BeamFeature
from src.analysis.post_processor import DESIGN_SHEAR_FV, DESIGN_STRENGTH_F
from src.analysis.results import AnalysisResults
from src.analysis.unit import M2_TO_MM2, M4_TO_MM4, M_TO_MM
from src.structure.enums import BeamType
from src.structure.structural_model import Beam, StructuralModel


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
        orientation = _beam_orientation(p1, p2)

        stress_ratio, stress_mpa = compute_beam_stress_metrics(model, beam_id, strength_results.beam_forces.get(beam_id))
        disp_max, disp_mean, rel_disp = _beam_displacement_metrics(beam, service_results)
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
            stress_ratio=stress_ratio,
            stress_mpa=stress_mpa,
            displacement_max_mm=disp_max,
            displacement_mean_mm=disp_mean,
            relative_displacement_mm=rel_disp,
            symmetry_key=symmetry_key,
        )
    return features


def group_features_by_symmetry(features: Dict[int, BeamFeature]) -> Dict[tuple, list[BeamFeature]]:
    grouped = defaultdict(list)
    for feature in features.values():
        grouped[feature.symmetry_key].append(feature)
    return dict(grouped)


def compute_beam_stress_metrics(
    model: StructuralModel, beam_id: int, forces: np.ndarray | None
) -> tuple[float, float]:
    beam = model.beams[beam_id]
    if forces is None or len(forces) < 12:
        return 0.0, 0.0

    sec = beam.section
    area = sec.area * M2_TO_MM2
    if hasattr(sec, "H") and sec.H > 0:
        wz = (sec.Ix * M4_TO_MM4) / (sec.H * M_TO_MM / 2.0)
    else:
        wz = (sec.Ix * M4_TO_MM4) / math.sqrt(sec.area * M2_TO_MM2)

    if hasattr(sec, "B") and sec.B > 0:
        wy = (sec.Iy * M4_TO_MM4) / (sec.B * M_TO_MM / 2.0)
    else:
        wy = (sec.Iy * M4_TO_MM4) / math.sqrt(sec.area * M2_TO_MM2)

    max_ratio = 0.0
    max_stress = 0.0
    for offset in (0, 6):
        axial = forces[offset + 0]
        vy = forces[offset + 1]
        vz = forces[offset + 2]
        my = forces[offset + 4]
        mz = forces[offset + 5]

        sigma = abs(axial / area) + abs(mz / wz) + abs(my / wy)
        tau = math.sqrt(vy**2 + vz**2) / (area * 0.5)
        ratio = max(sigma / DESIGN_STRENGTH_F, tau / DESIGN_SHEAR_FV)
        stress = max(sigma, tau)
        if ratio > max_ratio:
            max_ratio = float(ratio)
            max_stress = float(stress)
    return max_ratio, max_stress


def _beam_displacement_metrics(beam: Beam, results: AnalysisResults) -> tuple[float, float, float]:
    d1 = results.displacements.get(beam.start_node, np.zeros(6))[:3]
    d2 = results.displacements.get(beam.end_node, np.zeros(6))[:3]
    norm1 = float(np.linalg.norm(d1))
    norm2 = float(np.linalg.norm(d2))
    return max(norm1, norm2), (norm1 + norm2) / 2.0, float(np.linalg.norm(d2 - d1))


def _beam_orientation(p1: np.ndarray, p2: np.ndarray) -> str:
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
