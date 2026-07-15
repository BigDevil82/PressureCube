from __future__ import annotations

from dataclasses import dataclass

from src.analysis.evaluator import EvalResult, StructuralEvaluator
from src.optimization.design_space import get_layout_adapter, get_section_lib
from src.optimization.parallel_ga import GAResult
from src.structure.enums import BeamType
from src.structure.section_assignment import BEAM_SECTION_TYPES
from src.structure.section_grouping.models import SectionGroupingRuleset
from src.structure.structural_design import StructuralDesign
from src.structure.structural_model import StructuralModel


MIN_FRAME_SECTION_INDEX = 2  # BOX-150x6


@dataclass(frozen=True)
class RepairStep:
    group_id: str
    section_index: int
    u_sigma: float
    u_delta: float


@dataclass(frozen=True)
class GroupedOptimizationResult:
    ga_result: GAResult
    initial_fea_result: EvalResult
    fea_result: EvalResult
    topology: dict
    section_indices: dict[str, int]
    design: StructuralDesign
    model: StructuralModel
    repair_steps: list[RepairStep]

    @property
    def was_repaired(self) -> bool:
        return bool(self.repair_steps)


def validate_and_repair(
    ga_result: GAResult,
    ruleset: SectionGroupingRuleset,
    base_params,
    pressure: float,
    mesh_size: float,
    max_steps: int = 24,
) -> GroupedOptimizationResult:
    """Validate a grouped GA candidate and enlarge sections until FEA passes."""

    topology = dict(ga_result.optimal_topology)
    sections = apply_final_section_constraints(
        ga_result.optimal_section_indices_by_group,
        ruleset,
    )
    evaluator = StructuralEvaluator(mode="fea", mesh_size=mesh_size)
    design, model = _build_design(topology, sections, ruleset, base_params, pressure)
    fea = evaluator.evaluate(design)
    initial_fea = fea
    steps = []

    if not fea.feasible:
        order = _repair_order(ruleset, fea)
        section_counts = _section_counts(ruleset)
        cursor = 0
        for _ in range(max_steps):
            group_id = _next_adjustable_group(order, sections, section_counts, cursor)
            if group_id is None:
                break
            cursor = order.index(group_id) + 1
            sections[group_id] += 1
            design, model = _build_design(topology, sections, ruleset, base_params, pressure)
            fea = evaluator.evaluate(design)
            steps.append(RepairStep(group_id, sections[group_id], fea.u_sigma, fea.u_delta))
            if fea.feasible:
                break

    return GroupedOptimizationResult(
        ga_result=ga_result,
        initial_fea_result=initial_fea,
        fea_result=fea,
        topology=topology,
        section_indices=sections,
        design=design,
        model=model,
        repair_steps=steps,
    )


def apply_final_section_constraints(
    sections: dict[str, int],
    ruleset: SectionGroupingRuleset,
) -> dict[str, int]:
    """Apply engineering minimum sections before final FEA verification."""

    constrained = dict(sections)
    for group_id, beam_type in ruleset.group_beam_types().items():
        if beam_type == BeamType.FRAME:
            constrained[group_id] = max(constrained[group_id], MIN_FRAME_SECTION_INDEX)
    return constrained


def section_result_rows(result: GroupedOptimizationResult) -> list[dict]:
    lib = get_section_lib(result.design.layout_strategy)
    group_types = result.design.section_assignment.groups
    type_names = {
        "FRAME": "框架梁",
        "MAIN": "主梁",
        "SECONDARY": "次梁",
        "BRACE": "支撑梁",
    }
    rows = []
    for group_id in result.design.section_assignment.groups:
        group = group_types[group_id]
        index = result.section_indices[group_id]
        section = lib.get_section(BEAM_SECTION_TYPES[group.beam_type], index)
        rows.append(
            {
                "分组": group_id,
                "构件类型": type_names[group.beam_type.name],
                # Streamlit right-aligns numeric columns; display indices as labels.
                "截面索引": str(index),
                "截面名称": section.name,
            }
        )
    return rows


def _build_design(
    topology: dict,
    sections: dict[str, int],
    ruleset: SectionGroupingRuleset,
    base_params,
    pressure: float,
) -> tuple[StructuralDesign, StructuralModel]:
    adapter = get_layout_adapter(ruleset.layout_strategy)
    geometry = {"L": base_params.length, "W": base_params.width, "H": base_params.height}
    return adapter.build_design_with_model(
        geometry=geometry,
        pressure=pressure,
        topology=topology,
        section_group_rules=ruleset,
        section_indices_by_group=sections,
        base_params=base_params,
    )


def _section_counts(ruleset: SectionGroupingRuleset) -> dict[str, int]:
    lib = get_section_lib(ruleset.layout_strategy)
    return {
        group_id: lib.section_counts[BEAM_SECTION_TYPES[beam_type]]
        for group_id, beam_type in ruleset.group_beam_types().items()
    }


def _repair_order(ruleset: SectionGroupingRuleset, fea: EvalResult) -> list[str]:
    use_stress = fea.u_sigma >= fea.u_delta
    score = {
        item.group_id: item.stress_ratio_p90 if use_stress else item.displacement_p90_mm
        for item in ruleset.prototypes
    }
    return sorted(ruleset.group_ids(), key=lambda group_id: score.get(group_id, 0.0), reverse=True)


def _next_adjustable_group(
    order: list[str],
    sections: dict[str, int],
    counts: dict[str, int],
    cursor: int,
) -> str | None:
    if not order:
        return None
    for offset in range(len(order)):
        group_id = order[(cursor + offset) % len(order)]
        if sections[group_id] < counts[group_id] - 1:
            return group_id
    return None
