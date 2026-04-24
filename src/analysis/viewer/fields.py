from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.analysis.results import AnalysisResults, ModalAnalysisResults
from src.analysis.unit import M_TO_MM
from src.structure.structural_model import StructuralModel


@dataclass(frozen=True)
class ScalarField:
    label: str
    values: Dict[int, float]
    unit: str


def get_displacement_field(model: StructuralModel, results: AnalysisResults, component: str) -> ScalarField:
    labels = {
        "magnitude": ("总位移", "mm"),
        "ux": ("Ux", "mm"),
        "uy": ("Uy", "mm"),
        "uz": ("Uz", "mm"),
    }
    label, unit = labels[component]
    values: Dict[int, float] = {}
    for node_id in model.points:
        disp = results.displacements.get(node_id)
        if disp is None:
            values[node_id] = 0.0
            continue
        if component == "magnitude":
            values[node_id] = float(np.linalg.norm(disp[:3]))
        elif component == "ux":
            values[node_id] = float(disp[0])
        elif component == "uy":
            values[node_id] = float(disp[1])
        else:
            values[node_id] = float(disp[2])
    return ScalarField(label=label, values=values, unit=unit)


def get_beam_force_field(model: StructuralModel, results: AnalysisResults, component: str) -> ScalarField:
    index_map = {"N": 0, "Vy": 1, "Vz": 2, "T": 3, "My": 4, "Mz": 5}
    unit_map = {"N": "N", "Vy": "N", "Vz": "N", "T": "N·mm", "My": "N·mm", "Mz": "N·mm"}
    values: Dict[int, float] = {}
    response_index = index_map[component]
    for beam_id in model.beams:
        force = results.beam_forces.get(beam_id)
        if force is None or len(force) < 12:
            values[beam_id] = 0.0
            continue
        values[beam_id] = max(abs(float(force[response_index])), abs(float(force[response_index + 6])))
    return ScalarField(label=component, values=values, unit=unit_map[component])


def _plate_scalar_at_gauss(stress_raw: np.ndarray, offset: int, component: str, thickness_mm: float) -> float:
    nxx = float(stress_raw[offset + 0])
    nyy = float(stress_raw[offset + 1])
    nxy = float(stress_raw[offset + 2])
    mxx = float(stress_raw[offset + 3])
    myy = float(stress_raw[offset + 4])
    mxy = float(stress_raw[offset + 5])

    c_n = 1.0 / thickness_mm
    c_m = 6.0 / (thickness_mm**2)

    top_sig_x = nxx * c_n + mxx * c_m
    top_sig_y = nyy * c_n + myy * c_m
    top_tau_xy = nxy * c_n + mxy * c_m
    bot_sig_x = nxx * c_n - mxx * c_m
    bot_sig_y = nyy * c_n - myy * c_m
    bot_tau_xy = nxy * c_n - mxy * c_m

    if component == "sigma_x":
        return max(abs(top_sig_x), abs(bot_sig_x))
    if component == "sigma_y":
        return max(abs(top_sig_y), abs(bot_sig_y))
    if component == "tau_xy":
        return max(abs(top_tau_xy), abs(bot_tau_xy))

    top_vm = math.sqrt(top_sig_x**2 + top_sig_y**2 - top_sig_x * top_sig_y + 3.0 * top_tau_xy**2)
    bot_vm = math.sqrt(bot_sig_x**2 + bot_sig_y**2 - bot_sig_x * bot_sig_y + 3.0 * bot_tau_xy**2)
    return max(top_vm, bot_vm)


def get_plate_stress_field(model: StructuralModel, results: AnalysisResults, component: str) -> ScalarField:
    labels = {
        "von_mises": "Von Mises",
        "sigma_x": "σx",
        "sigma_y": "σy",
        "tau_xy": "τxy",
    }
    values: Dict[int, float] = {}
    for plate_id, plate in model.plates.items():
        stress_raw = results.plate_stresses.get(plate_id)
        if stress_raw is None or len(stress_raw) < 32:
            values[plate_id] = 0.0
            continue
        thickness_mm = (
            model.params.opening_panel.thickness if plate.panel_type == "opening" else model.params.plate_thickness
        ) * M_TO_MM
        gauss_values = [
            _plate_scalar_at_gauss(stress_raw, gauss_index * 8, component, thickness_mm) for gauss_index in range(4)
        ]
        values[plate_id] = max(gauss_values)
    return ScalarField(label=labels[component], values=values, unit="MPa")


def get_plate_nodal_stress_field(model: StructuralModel, results: AnalysisResults, component: str) -> ScalarField:
    """将板应力投影到节点并做平均，用于连续云图显示。"""
    element_field = get_plate_stress_field(model, results, component)
    node_sums: Dict[int, float] = {node_id: 0.0 for node_id in model.points}
    node_counts: Dict[int, int] = {node_id: 0 for node_id in model.points}

    for plate_id, plate in model.plates.items():
        value = element_field.values.get(plate_id, 0.0)
        for node_id in plate.nodes:
            node_sums[node_id] += value
            node_counts[node_id] += 1

    nodal_values = {
        node_id: (node_sums[node_id] / node_counts[node_id] if node_counts[node_id] > 0 else 0.0)
        for node_id in model.points
    }
    return ScalarField(label=element_field.label, values=nodal_values, unit=element_field.unit)


def get_mode_shape_field(modal_results: ModalAnalysisResults, mode: int) -> ScalarField:
    mode_shape = modal_results.mode_shapes.get(mode, {})
    values: Dict[int, float] = {}
    for node_id, vector in mode_shape.items():
        values[node_id] = float(np.linalg.norm(vector[:3]))
    return ScalarField(label=f"Mode {mode}", values=values, unit="相对值")


def get_extreme_location(model: StructuralModel, results: AnalysisResults, view: str, component: str):
    if view == "deformation":
        field = get_displacement_field(model, results, component)
        node_id = max(field.values, key=lambda key: abs(field.values[key]))
        point = model.points[node_id]
        return {
            "kind": "node",
            "id": node_id,
            "value": field.values[node_id],
            "coords": point.coordinates(),
            "label": field.label,
            "unit": field.unit,
        }

    if view == "plate_stress":
        field = get_plate_stress_field(model, results, component)
        plate_id = max(field.values, key=lambda key: abs(field.values[key]))
        plate = model.plates[plate_id]
        coords = np.mean([model.points[node_id].coordinates() for node_id in plate.nodes], axis=0)
        return {
            "kind": "plate",
            "id": plate_id,
            "value": field.values[plate_id],
            "coords": coords,
            "label": field.label,
            "unit": field.unit,
        }

    field = get_beam_force_field(model, results, component)
    beam_id = max(field.values, key=lambda key: abs(field.values[key]))
    beam = model.beams[beam_id]
    coords = (model.points[beam.start_node].coordinates() + model.points[beam.end_node].coordinates()) / 2.0
    return {
        "kind": "beam",
        "id": beam_id,
        "value": field.values[beam_id],
        "coords": coords,
        "label": field.label,
        "unit": field.unit,
    }
