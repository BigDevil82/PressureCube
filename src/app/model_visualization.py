from __future__ import annotations

from typing import Dict

import numpy as np
import plotly.graph_objects as go

from src.structure.enums import BeamType
from src.structure.structural_model import StructuralModel


def get_beam_style(beam_type: BeamType) -> tuple[str, int]:
    if beam_type == BeamType.FRAME:
        return "#FF4B4B", 10
    if beam_type == BeamType.MAIN:
        return "#1E90FF", 7
    if beam_type == BeamType.SECONDARY:
        return "#FFA500", 4
    if beam_type == BeamType.BRACE:
        return "#27C344", 2
    return "#888888", 1


def build_structure_figure(
    model: StructuralModel,
    show_plates: bool = True,
    show_nodes: bool = False,
    show_beams: bool = True,
    mesh_debug_view: bool = False,
    original_model: StructuralModel | None = None,
    beam_coordinates: Dict[int, np.ndarray] | None = None,
    plate_opacity: float | None = None,
) -> go.Figure:
    traces = []
    beam_source_model = original_model if mesh_debug_view and original_model is not None else model
    coords = beam_coordinates or {node_id: point.coordinates() for node_id, point in model.points.items()}
    beam_coords = (
        {node_id: point.coordinates() for node_id, point in beam_source_model.points.items()}
        if beam_coordinates is None or beam_source_model is not model
        else coords
    )

    if show_beams:
        beams_by_type: Dict[BeamType, Dict[str, list[float]]] = {}
        for beam in beam_source_model.beams.values():
            if beam.is_virtual:
                continue
            beams_by_type.setdefault(beam.beam_type, {"x": [], "y": [], "z": []})
            start = beam_coords[beam.start_node]
            end = beam_coords[beam.end_node]
            beams_by_type[beam.beam_type]["x"].extend([start[0], end[0], None])
            beams_by_type[beam.beam_type]["y"].extend([start[1], end[1], None])
            beams_by_type[beam.beam_type]["z"].extend([start[2], end[2], None])

        for beam_type, data in beams_by_type.items():
            color, width = get_beam_style(beam_type)
            traces.append(
                go.Scatter3d(
                    x=data["x"],
                    y=data["y"],
                    z=data["z"],
                    mode="lines",
                    line=dict(color=color, width=width, dash="solid"),
                    name=str(beam_type).split(".")[-1],
                )
            )

    if mesh_debug_view and original_model is not None:
        traces.extend(_build_mesh_debug_traces(model, original_model))

    if show_plates and model.plates:
        traces.extend(_build_plate_traces(model, coords, plate_opacity))

    if show_nodes:
        traces.append(
            go.Scatter3d(
                x=[coords[node_id][0] for node_id in model.points],
                y=[coords[node_id][1] for node_id in model.points],
                z=[coords[node_id][2] for node_id in model.points],
                mode="markers",
                marker=dict(
                    size=1 if mesh_debug_view else 2, color="black", opacity=0.45 if mesh_debug_view else 0.8
                ),
                showlegend=False,
            )
        )

    layout = go.Layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            bgcolor="#ffffff",
        ),
        paper_bgcolor="#ffffff",
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.05, font=dict(color="black")),
        height=760,
    )
    fig = go.Figure(data=traces, layout=layout)
    fig.frames = _build_rotation_frames()
    fig.update_layout(updatemenus=[_build_rotation_menu()])
    return fig


def _build_mesh_debug_traces(model: StructuralModel, original_model: StructuralModel) -> list[go.Scatter3d]:
    original_edges = set()
    for beam in original_model.beams.values():
        if beam.is_virtual:
            continue
        p1 = original_model.points[beam.start_node]
        p2 = original_model.points[beam.end_node]
        edge = tuple(
            sorted(
                (
                    (round(p1.x, 6), round(p1.y, 6), round(p1.z, 6)),
                    (round(p2.x, 6), round(p2.y, 6), round(p2.z, 6)),
                )
            )
        )
        original_edges.add(edge)

    debug_coords = {"x": [], "y": [], "z": []}
    seen_debug_edges = set()

    def add_edge(pa, pb):
        edge = tuple(
            sorted(
                (
                    (round(pa.x, 6), round(pa.y, 6), round(pa.z, 6)),
                    (round(pb.x, 6), round(pb.y, 6), round(pb.z, 6)),
                )
            )
        )
        if edge in original_edges or edge in seen_debug_edges:
            return
        seen_debug_edges.add(edge)
        debug_coords["x"].extend([pa.x, pb.x, None])
        debug_coords["y"].extend([pa.y, pb.y, None])
        debug_coords["z"].extend([pa.z, pb.z, None])

    for beam in model.beams.values():
        if beam.is_virtual:
            continue
        add_edge(model.points[beam.start_node], model.points[beam.end_node])

    for plate in model.plates.values():
        for index in range(4):
            add_edge(model.points[plate.nodes[index]], model.points[plate.nodes[(index + 1) % 4]])

    if not debug_coords["x"]:
        return []

    return [
        go.Scatter3d(
            x=debug_coords["x"],
            y=debug_coords["y"],
            z=debug_coords["z"],
            mode="lines",
            line=dict(color="#666666", width=2, dash="dash"),
            name="Mesh Debug",
            opacity=0.75,
        )
    ]


def _build_plate_traces(
    model: StructuralModel, coords: Dict[int, np.ndarray], plate_opacity: float | None
) -> list[go.Mesh3d]:
    all_node_ids = sorted(model.points)
    node_to_idx = {node_id: idx for idx, node_id in enumerate(all_node_ids)}
    mesh_x = [coords[node_id][0] for node_id in all_node_ids]
    mesh_y = [coords[node_id][1] for node_id in all_node_ids]
    mesh_z = [coords[node_id][2] for node_id in all_node_ids]
    plate_groups = {
        "steel": {
            "i": [],
            "j": [],
            "k": [],
            "color": "#A9A9A9",
            "name": "Steel Plate",
            "opacity": plate_opacity or 0.42,
        },
        "opening": {
            "i": [],
            "j": [],
            "k": [],
            "color": "#00B7C2",
            "name": "Opening Panel",
            "opacity": plate_opacity or 0.75,
        },
    }
    for plate in model.plates.values():
        plate_idx = [node_to_idx[node_id] for node_id in plate.nodes]
        group = plate_groups["opening" if plate.panel_type == "opening" else "steel"]
        group["i"].extend([plate_idx[0], plate_idx[0]])
        group["j"].extend([plate_idx[1], plate_idx[2]])
        group["k"].extend([plate_idx[2], plate_idx[3]])

    traces = []
    for group in plate_groups.values():
        if not group["i"]:
            continue
        traces.append(
            go.Mesh3d(
                x=mesh_x,
                y=mesh_y,
                z=mesh_z,
                i=group["i"],
                j=group["j"],
                k=group["k"],
                opacity=group["opacity"],
                color=group["color"],
                alphahull=-1,
                lighting=dict(ambient=0.5, diffuse=0.8),
                name=group["name"],
                showlegend=True,
                hoverinfo="skip",
            )
        )
    return traces


def _build_rotation_frames() -> list[go.Frame]:
    frames = []
    for step in range(180):
        theta = (2 * np.pi / 180) * step
        frames.append(
            go.Frame(
                layout=dict(
                    scene=dict(camera=dict(eye=dict(x=2 * np.cos(theta), y=2 * np.sin(theta), z=1.25)))
                )
            )
        )
    return frames


def _build_rotation_menu() -> dict:
    return dict(
        type="buttons",
        showactive=False,
        y=0.1,
        x=0.0,
        xanchor="left",
        yanchor="bottom",
        pad=dict(t=0, r=10),
        buttons=[
            dict(
                label="▶ 开始旋转",
                method="animate",
                args=[
                    None,
                    dict(
                        frame=dict(duration=50, redraw=True),
                        fromcurrent=True,
                        transition=dict(duration=0),
                        mode="immediate",
                    ),
                ],
            ),
            dict(
                label="⏸ 停止",
                method="animate",
                args=[
                    [None],
                    dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0)),
                ],
            ),
        ],
    )
