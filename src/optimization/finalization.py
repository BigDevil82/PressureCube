from __future__ import annotations

from src.analysis.evaluator import StructuralEvaluator
from src.optimization.design_space import get_layout_adapter
from src.optimization.section_constraints import apply_ungrouped_section_constraints
from src.structure.section_grouping.models import SectionGroupingRuleset


def finalize_ungrouped_design(
    *,
    params,
    pressure: float,
    topology: dict,
    sections: list[int],
    mesh_size: float,
) -> dict:
    """Apply section minimums and return the final real-FEA result."""

    sections = apply_ungrouped_section_constraints(sections)
    adapter = get_layout_adapter(params.layout_strategy)
    rules = SectionGroupingRuleset.by_beam_type(params.layout_strategy)
    design = adapter.build_design(
        geometry={"L": params.length, "W": params.width, "H": params.height},
        pressure=pressure,
        topology=topology,
        section_group_rules=rules,
        section_indices_by_group=dict(zip(rules.group_ids(), sections)),
        base_params=params,
    )
    result = StructuralEvaluator(mode="fea", mesh_size=mesh_size).evaluate(design)
    return {
        "sections": sections,
        "topology": dict(topology),
        "mass": result.mass,
        "u_sigma": result.u_sigma,
        "u_delta": result.u_delta,
        "feasible": result.feasible,
    }
