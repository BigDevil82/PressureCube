from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.structure.section_grouping.beam_features import BeamFeature, beam_orientation
from src.structure.enums import BeamType
from src.structure.structural_model import Beam, StructuralModel

ASSIGNMENT_FEATURE_SCHEMA = [
    "x_norm",
    "y_norm",
    "z_norm",
    "length_norm",
    "ori_x",
    "ori_y",
    "ori_z",
    "face_horizontal",
    "face_longitudinal_side",
    "face_end_side",
    "face_other",
]


@dataclass(frozen=True)
class BeamGeometryContext:
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    span: np.ndarray
    length_ref: float


def make_geometry_context(model: StructuralModel) -> BeamGeometryContext:
    coords = np.array([point.coordinates() for point in model.points.values()])
    bounds_min = coords.min(axis=0)
    bounds_max = coords.max(axis=0)
    span = np.maximum(bounds_max - bounds_min, 1e-9)
    return BeamGeometryContext(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        span=span,
        length_ref=float(np.linalg.norm(span)),
    )


def beam_assignment_vector(
    model: StructuralModel,
    beam: Beam,
    context: BeamGeometryContext | None = None,
) -> np.ndarray:
    """生成新模型分配梁组时可用的几何向量。

    这里不能使用应力/位移，因为优化时需要在 FEA 前确定每根梁使用哪个截面。
    """

    context = context or make_geometry_context(model)
    p1 = model.points[beam.start_node].coordinates()
    p2 = model.points[beam.end_node].coordinates()
    midpoint = (p1 + p2) / 2.0
    normalized_midpoint = (midpoint - context.bounds_min) / context.span
    length_norm = float(np.linalg.norm(p2 - p1) / max(context.length_ref, 1e-9))
    orientation = beam_orientation(p1, p2)
    face_family = normalize_face_family(beam.on_face)
    return assignment_vector_from_parts(normalized_midpoint, length_norm, orientation, face_family)


def feature_assignment_vector(feature: BeamFeature) -> np.ndarray:
    face_family = normalize_face_family(feature.face)
    return assignment_vector_from_parts(
        feature.normalized_midpoint,
        feature.length_norm,
        feature.orientation,
        face_family,
    )


def assignment_vector_from_parts(
    normalized_midpoint: np.ndarray,
    length_norm: float,
    orientation: str,
    face_family: str,
) -> np.ndarray:
    orientation_one_hot = [float(orientation == value) for value in ("x", "y", "z")]
    face_one_hot = [
        float(face_family == "horizontal"),
        float(face_family == "longitudinal_side"),
        float(face_family == "end_side"),
        float(face_family == "other"),
    ]
    return np.array(
        [
            float(normalized_midpoint[0]),
            float(normalized_midpoint[1]),
            float(normalized_midpoint[2]),
            float(length_norm),
            *orientation_one_hot,
            *face_one_hot,
        ],
        dtype=float,
    )


def normalize_face_family(face: str) -> str:
    parts = set(str(face).split("/"))
    if "top" in parts or "bottom" in parts:
        return "horizontal"
    if "front" in parts or "back" in parts:
        return "longitudinal_side"
    if "left" in parts or "right" in parts:
        return "end_side"
    return "other"


def beam_type_sort_key(beam_type: BeamType) -> int:
    return int(beam_type.value)
