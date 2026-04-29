from __future__ import annotations

import plotly.graph_objects as go
from plotly.colors import qualitative

from src.app.model_visualization import build_structure_figure
from src.structure.section_grouping.application import assign_beam_groups
from src.structure.section_grouping.models import SectionGroupingRuleset
from src.structure.structural_model import StructuralModel


def build_section_grouping_figure(
    model: StructuralModel,
    ruleset: SectionGroupingRuleset,
    beam_to_group: dict[int, str] | None = None,
) -> go.Figure:
    """显示梁组分配结果。"""

    assignments = beam_to_group or assign_beam_groups(model, ruleset)
    fig = build_structure_figure(
        model, show_plates=True, show_nodes=False, show_beams=False, plate_opacity=0.12
    )
    palette = qualitative.Safe + qualitative.Bold + qualitative.Dark24
    assigned_groups = set(assignments.values())
    group_ids = [group_id for group_id in ruleset.group_ids() if group_id in assigned_groups]
    color_by_group = {group_id: palette[idx % len(palette)] for idx, group_id in enumerate(group_ids)}

    for group_id in group_ids:
        coords = {"x": [], "y": [], "z": []}
        hover = []
        for beam_id, assigned_group in assignments.items():
            if assigned_group != group_id:
                continue
            beam = model.beams[beam_id]
            p1 = model.points[beam.start_node]
            p2 = model.points[beam.end_node]
            coords["x"].extend([p1.x, p2.x, None])
            coords["y"].extend([p1.y, p2.y, None])
            coords["z"].extend([p1.z, p2.z, None])
            text = f"beam={beam_id}<br>group={group_id}<br>type={beam.beam_type.name}<br>face={beam.on_face}"
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

    fig.update_layout(title="稳定截面分组规则在当前模型上的分配结果")
    return fig
