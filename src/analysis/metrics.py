from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.analysis.results import AnalysisResults
from src.analysis.unit import M2_TO_MM2, M4_TO_MM4, M_TO_MM
from src.structure.structural_model import Beam, Plate, StructuralModel


DESIGN_STRENGTH_F = 314.0  # MPa (N/mm2), Q345 example
DESIGN_SHEAR_FV = 175.0  # MPa (N/mm2)


@dataclass(frozen=True)
class BeamStressMetric:
    ratio: float
    stress_mpa: float
    mode: str


@dataclass(frozen=True)
class BeamDisplacementMetric:
    max_mm: float
    mean_mm: float
    relative_mm: float


@dataclass(frozen=True)
class PlateStressState:
    sigma_x: float
    sigma_y: float
    tau_xy: float
    von_mises: float


@dataclass(frozen=True)
class PlateStressMetric:
    ratio: float
    stress_mpa: float
    component_values: dict[str, float]


def beam_stress_metric(beam: Beam, forces: np.ndarray | None) -> BeamStressMetric:
    """Return the controlling simplified GB50017-like stress metric for one beam."""

    if forces is None or len(forces) < 12 or beam.is_virtual:
        return BeamStressMetric(ratio=0.0, stress_mpa=0.0, mode="OK")

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
    mode = "OK"
    for offset in (0, 6):
        axial = forces[offset + 0]
        vy = forces[offset + 1]
        vz = forces[offset + 2]
        my = forces[offset + 4]
        mz = forces[offset + 5]

        sigma = abs(axial / area) + abs(mz / wz) + abs(my / wy)
        tau = math.sqrt(vy**2 + vz**2) / (area * 0.5)
        ratio_sigma = sigma / DESIGN_STRENGTH_F
        ratio_tau = tau / DESIGN_SHEAR_FV
        ratio = max(ratio_sigma, ratio_tau)
        stress = max(sigma, tau)
        if ratio > max_ratio:
            max_ratio = float(ratio)
            max_stress = float(stress)
            mode = "Bending" if ratio_sigma > ratio_tau else "Shear"
    return BeamStressMetric(ratio=max_ratio, stress_mpa=max_stress, mode=mode)


def beam_displacement_metric(beam: Beam, results: AnalysisResults) -> BeamDisplacementMetric:
    d1 = results.displacements.get(beam.start_node, np.zeros(6))[:3]
    d2 = results.displacements.get(beam.end_node, np.zeros(6))[:3]
    norm1 = float(np.linalg.norm(d1))
    norm2 = float(np.linalg.norm(d2))
    return BeamDisplacementMetric(
        max_mm=max(norm1, norm2),
        mean_mm=(norm1 + norm2) / 2.0,
        relative_mm=float(np.linalg.norm(d2 - d1)),
    )


def plate_thickness_mm(model: StructuralModel, plate: Plate) -> float:
    thickness = (
        model.params.opening_panel.thickness if plate.panel_type == "opening" else model.params.plate_thickness
    )
    return thickness * M_TO_MM


def plate_design_strength_mpa(model: StructuralModel, plate: Plate) -> float:
    if plate.panel_type == "opening":
        return model.params.opening_panel.design_strength_mpa
    return DESIGN_STRENGTH_F


def plate_stress_state_at_gauss(
    stress_raw: np.ndarray,
    gauss_index: int,
    thickness_mm: float,
    sign: float,
) -> PlateStressState:
    offset = gauss_index * 8
    nxx = float(stress_raw[offset + 0])
    nyy = float(stress_raw[offset + 1])
    nxy = float(stress_raw[offset + 2])
    mxx = float(stress_raw[offset + 3])
    myy = float(stress_raw[offset + 4])
    mxy = float(stress_raw[offset + 5])

    c_n = 1.0 / thickness_mm
    c_m = 6.0 / (thickness_mm**2)
    sigma_x = nxx * c_n + sign * mxx * c_m
    sigma_y = nyy * c_n + sign * myy * c_m
    tau_xy = nxy * c_n + sign * mxy * c_m
    von_mises = math.sqrt(sigma_x**2 + sigma_y**2 - sigma_x * sigma_y + 3.0 * tau_xy**2)
    return PlateStressState(sigma_x=sigma_x, sigma_y=sigma_y, tau_xy=tau_xy, von_mises=von_mises)


def plate_stress_metric(model: StructuralModel, plate_id: int, stress_raw: np.ndarray | None) -> PlateStressMetric:
    plate = model.plates.get(plate_id)
    if plate is None or stress_raw is None or len(stress_raw) < 32:
        return PlateStressMetric(
            ratio=0.0,
            stress_mpa=0.0,
            component_values={"von_mises": 0.0, "sigma_x": 0.0, "sigma_y": 0.0, "tau_xy": 0.0},
        )

    thickness_mm = plate_thickness_mm(model, plate)
    design_strength = plate_design_strength_mpa(model, plate)
    max_values = {"von_mises": 0.0, "sigma_x": 0.0, "sigma_y": 0.0, "tau_xy": 0.0}
    for gauss_index in range(4):
        for sign in (1.0, -1.0):
            state = plate_stress_state_at_gauss(stress_raw, gauss_index, thickness_mm, sign)
            max_values["von_mises"] = max(max_values["von_mises"], abs(state.von_mises))
            max_values["sigma_x"] = max(max_values["sigma_x"], abs(state.sigma_x))
            max_values["sigma_y"] = max(max_values["sigma_y"], abs(state.sigma_y))
            max_values["tau_xy"] = max(max_values["tau_xy"], abs(state.tau_xy))
    return PlateStressMetric(
        ratio=max_values["von_mises"] / design_strength,
        stress_mpa=max_values["von_mises"],
        component_values=max_values,
    )


def displacement_field_values(model: StructuralModel, results: AnalysisResults, component: str) -> Dict[int, float]:
    values: Dict[int, float] = {}
    for node_id in model.points:
        disp = results.displacements.get(node_id)
        if disp is None:
            values[node_id] = 0.0
        elif component == "magnitude":
            values[node_id] = float(np.linalg.norm(disp[:3]))
        elif component == "ux":
            values[node_id] = float(disp[0])
        elif component == "uy":
            values[node_id] = float(disp[1])
        else:
            values[node_id] = float(disp[2])
    return values
