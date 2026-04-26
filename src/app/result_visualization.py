from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from src.analysis.results import AnalysisResults, ModalAnalysisResults
from src.app.result_fields import (
    ScalarField,
    get_beam_force_field,
    get_displacement_field,
    get_extreme_location,
    get_mode_shape_field,
    get_plate_nodal_stress_field,
    get_plate_stress_field,
)
from src.app.model_visualization import build_structure_figure, get_beam_style
from src.structure.structural_model import StructuralModel


def _value_range(values: Iterable[float]) -> tuple[float, float]:
    values = list(values)
    if not values:
        return 0.0, 1.0
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1.0
    return vmin, vmax


def _sample_color(value: float, vmin: float, vmax: float, colorscale: str = "Turbo") -> str:
    if vmax <= vmin:
        ratio = 0.5
    else:
        ratio = (value - vmin) / (vmax - vmin)
    ratio = min(max(ratio, 0.0), 1.0)
    return sample_colorscale(colorscale, [ratio])[0]


def _add_scalar_bar(fig: go.Figure, field: ScalarField, vmin: float, vmax: float):
    fig.add_trace(
        go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode="markers",
            marker=dict(
                size=0.1,
                color=[vmin, vmax],
                cmin=vmin,
                cmax=vmax,
                colorscale="Turbo",
                showscale=True,
                colorbar=dict(title=f"{field.label} ({field.unit})"),
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )


def _deformed_node_positions(
    model: StructuralModel, nodal_vectors: Dict[int, np.ndarray], scale: float
) -> Dict[int, np.ndarray]:
    coords: Dict[int, np.ndarray] = {}
    for node_id, point in model.points.items():
        vector = nodal_vectors.get(node_id, np.zeros(6))
        coords[node_id] = point.coordinates() + vector[:3] * scale / 1000.0
    return coords


def build_static_result_figure(
    model: StructuralModel,
    results: AnalysisResults,
    view: str,
    component: str,
    deformation_scale: float = 1.0,
) -> go.Figure:
    if view == "deformation":
        field = get_displacement_field(model, results, component)
        deformed_coords = _deformed_node_positions(model, results.displacements, deformation_scale)
        fig = _build_deformation_surface_figure(model, field, deformed_coords)
        fig.update_layout(title=f"变形图: {field.label}")
        _add_extreme_marker(
            fig, get_extreme_location(model, results, view, component), deformed_coords=deformed_coords
        )
        return fig

    if view == "plate_stress":
        field = get_plate_stress_field(model, results, component)
        nodal_field = get_plate_nodal_stress_field(model, results, component)
        fig = _build_plate_scalar_figure(model, nodal_field)
        _add_extreme_marker(fig, get_extreme_location(model, results, view, component))
        return fig

    field = get_beam_force_field(model, results, component)
    fig = _build_beam_scalar_figure(model, field)
    _add_extreme_marker(fig, get_extreme_location(model, results, view, component))
    return fig


def build_modal_result_figure(
    model: StructuralModel,
    modal_results: ModalAnalysisResults,
    mode: int,
    deformation_scale: float = 1.0,
) -> go.Figure:
    mode_shape = modal_results.mode_shapes.get(mode, {})
    nodal_vectors = {node_id: vector for node_id, vector in mode_shape.items()}
    field = get_mode_shape_field(modal_results, mode)
    deformed_coords = _deformed_node_positions(model, nodal_vectors, deformation_scale)
    fig = build_structure_figure(model, show_plates=False, show_nodes=False, beam_coordinates=deformed_coords)
    fig.update_layout(title=f"振型 {mode}: {modal_results.frequencies_hz[mode - 1]:.3f} Hz")

    node_ids = sorted(model.points)
    values = [field.values.get(node_id, 0.0) for node_id in node_ids]
    x = [deformed_coords[node_id][0] for node_id in node_ids]
    y = [deformed_coords[node_id][1] for node_id in node_ids]
    z = [deformed_coords[node_id][2] for node_id in node_ids]
    vmin, vmax = _value_range(values)
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(size=2.5, color=values, colorscale="Turbo", cmin=vmin, cmax=vmax, opacity=0.9),
            name="振型节点",
            customdata=np.array(values),
            hovertemplate="模态幅值: %{customdata:.4f}<extra></extra>",
        )
    )
    _add_scalar_bar(fig, field, vmin, vmax)
    return fig


def _build_deformation_surface_figure(
    model: StructuralModel, field: ScalarField, deformed_coords: Dict[int, np.ndarray]
) -> go.Figure:
    fig = build_structure_figure(
        model,
        show_plates=False,
        show_nodes=False,
        show_beams=True,
        beam_coordinates=deformed_coords,
        plate_opacity=0.12,
    )
    all_node_ids = sorted(model.points)
    node_to_idx = {node_id: idx for idx, node_id in enumerate(all_node_ids)}
    x = [deformed_coords[node_id][0] for node_id in all_node_ids]
    y = [deformed_coords[node_id][1] for node_id in all_node_ids]
    z = [deformed_coords[node_id][2] for node_id in all_node_ids]
    intensity = [field.values.get(node_id, 0.0) for node_id in all_node_ids]
    tri_i = []
    tri_j = []
    tri_k = []
    for plate in model.plates.values():
        node_idx = [node_to_idx[node_id] for node_id in plate.nodes]
        tri_i.extend([node_idx[0], node_idx[0]])
        tri_j.extend([node_idx[1], node_idx[2]])
        tri_k.extend([node_idx[2], node_idx[3]])
    vmin, vmax = _value_range(intensity)
    fig.add_trace(
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=tri_i,
            j=tri_j,
            k=tri_k,
            intensity=intensity,
            intensitymode="vertex",
            colorscale="Turbo",
            cmin=vmin,
            cmax=vmax,
            opacity=1.0,
            colorbar=dict(title=f"{field.label} ({field.unit})"),
            name=field.label,
            hoverinfo="skip",
            showscale=True,
        )
    )
    return fig


def _build_plate_scalar_figure(model: StructuralModel, field: ScalarField) -> go.Figure:
    fig = build_structure_figure(model, show_plates=False, show_nodes=False, show_beams=True)
    all_node_ids = sorted(model.points)
    node_to_idx = {node_id: idx for idx, node_id in enumerate(all_node_ids)}
    mesh_x = [model.points[node_id].x for node_id in all_node_ids]
    mesh_y = [model.points[node_id].y for node_id in all_node_ids]
    mesh_z = [model.points[node_id].z for node_id in all_node_ids]
    intensities = [field.values.get(node_id, 0.0) for node_id in all_node_ids]
    tri_i = []
    tri_j = []
    tri_k = []

    for _, plate in model.plates.items():
        node_idx = [node_to_idx[node_id] for node_id in plate.nodes]
        tri_i.extend([node_idx[0], node_idx[0]])
        tri_j.extend([node_idx[1], node_idx[2]])
        tri_k.extend([node_idx[2], node_idx[3]])

    vmin, vmax = _value_range(field.values.values())
    fig.add_trace(
        go.Mesh3d(
            x=mesh_x,
            y=mesh_y,
            z=mesh_z,
            i=tri_i,
            j=tri_j,
            k=tri_k,
            intensity=intensities,
            intensitymode="vertex",
            colorscale="Turbo",
            cmin=vmin,
            cmax=vmax,
            opacity=1.0,
            flatshading=False,
            colorbar=dict(title=f"{field.label} ({field.unit})"),
            name=field.label,
            hoverinfo="skip",
            showscale=True,
        )
    )
    fig.update_layout(title=f"板结果云图: {field.label}")
    return fig


def _build_beam_scalar_figure(model: StructuralModel, field: ScalarField) -> go.Figure:
    fig = build_structure_figure(model, show_plates=True, show_nodes=False, plate_opacity=0.15)
    vmin, vmax = _value_range(field.values.values())
    buckets: Dict[str, Dict[str, list[float]]] = {}
    customdata_map: Dict[str, list[float]] = {}
    color_bins = 24
    for beam_id, beam in model.beams.items():
        if beam.is_virtual:
            continue
        start = model.points[beam.start_node]
        end = model.points[beam.end_node]
        value = field.values.get(beam_id, 0.0)
        if vmax <= vmin:
            binned_ratio = 0.5
        else:
            raw_ratio = (value - vmin) / (vmax - vmin)
            raw_ratio = min(max(raw_ratio, 0.0), 1.0)
            binned_ratio = round(raw_ratio * (color_bins - 1)) / max(color_bins - 1, 1)
        color = _sample_color(vmin + binned_ratio * (vmax - vmin), vmin, vmax)
        _, width = get_beam_style(beam.beam_type)
        key = f"{color}|{max(width + 2, 5)}|{binned_ratio:.4f}"
        buckets.setdefault(key, {"x": [], "y": [], "z": []})
        customdata_map.setdefault(key, [])
        buckets[key]["x"].extend([start.x, end.x, None])
        buckets[key]["y"].extend([start.y, end.y, None])
        buckets[key]["z"].extend([start.z, end.z, None])
        customdata_map[key].extend([value, value, None])

    for key, coords in buckets.items():
        color, width, _ = key.split("|")
        fig.add_trace(
            go.Scatter3d(
                x=coords["x"],
                y=coords["y"],
                z=coords["z"],
                mode="lines",
                line=dict(color=color, width=float(width)),
                customdata=customdata_map[key],
                hovertemplate=f"{field.label}: %{{customdata:.3f}} {field.unit}<extra></extra>",
                showlegend=False,
            )
        )

    _add_scalar_bar(fig, field, vmin, vmax)
    fig.update_layout(title=f"梁结果云图: {field.label}")
    return fig


def _add_extreme_marker(fig: go.Figure, extreme: dict, deformed_coords: Dict[int, np.ndarray] | None = None):
    coords = extreme["coords"]
    if deformed_coords is not None and extreme["kind"] == "node":
        coords = deformed_coords[extreme["id"]]
    fig.add_trace(
        go.Scatter3d(
            x=[coords[0]],
            y=[coords[1]],
            z=[coords[2]],
            mode="markers+text",
            marker=dict(size=8, color="#111111", symbol="diamond"),
            text=[f"Max {extreme['label']}"],
            textposition="top center",
            name="Extreme",
            hovertemplate=(
                f"ID: {extreme['id']}<br>{extreme['label']}: {extreme['value']:.3f} {extreme['unit']}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )
