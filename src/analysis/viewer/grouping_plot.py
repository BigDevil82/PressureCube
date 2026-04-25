from __future__ import annotations

import plotly.graph_objects as go
from plotly.colors import qualitative

from src.analysis.beam_grouping.models import BeamGroupingResult
from src.app.model_visualization import build_structure_figure
from src.structure.structural_model import StructuralModel


def build_beam_grouping_figure(model: StructuralModel, grouping: BeamGroupingResult) -> go.Figure:
    fig = build_structure_figure(model, show_plates=True, show_nodes=False, show_beams=False, plate_opacity=0.12)
    palette = qualitative.Safe + qualitative.Bold + qualitative.Dark24
    group_ids = sorted(set(grouping.beam_to_group.values()))
    color_by_group = {group_id: palette[idx % len(palette)] for idx, group_id in enumerate(group_ids)}

    for group_id in group_ids:
        coords = {"x": [], "y": [], "z": []}
        hover = []
        for beam_id, assigned_group in grouping.beam_to_group.items():
            if assigned_group != group_id:
                continue
            beam = model.beams[beam_id]
            p1 = model.points[beam.start_node]
            p2 = model.points[beam.end_node]
            feature = grouping.features[beam_id]
            coords["x"].extend([p1.x, p2.x, None])
            coords["y"].extend([p1.y, p2.y, None])
            coords["z"].extend([p1.z, p2.z, None])
            text = (
                f"beam={beam_id}<br>group={group_id}<br>type={beam.beam_type.name}"
                f"<br>stress_ratio={feature.stress_ratio:.3f}"
                f"<br>disp_max={feature.displacement_max_mm:.3f} mm"
                f"<br>symmetry={feature.symmetry_key}"
            )
            hover.extend([text, text, None])
        fig.add_trace(
            go.Scatter3d(
                x=coords["x"],
                y=coords["y"],
                z=coords["z"],
                mode="lines",
                line=dict(color=color_by_group[group_id], width=7),
                name=group_id,
                text=hover,
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig.update_layout(title="梁分组结果")
    return fig
