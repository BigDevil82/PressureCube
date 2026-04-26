from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.analysis.metrics import displacement_field_values, plate_stress_metric
from src.analysis.results import AnalysisResults, ModalAnalysisResults
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
    return ScalarField(label=label, values=displacement_field_values(model, results, component), unit=unit)


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


def get_plate_stress_field(model: StructuralModel, results: AnalysisResults, component: str) -> ScalarField:
    labels = {
        "von_mises": "Von Mises",
        "sigma_x": "σx",
        "sigma_y": "σy",
        "tau_xy": "τxy",
    }
    values: Dict[int, float] = {}
    for plate_id in model.plates:
        metric = plate_stress_metric(model, plate_id, results.plate_stresses.get(plate_id))
        values[plate_id] = metric.component_values[component]
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
