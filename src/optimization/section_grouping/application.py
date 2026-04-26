from __future__ import annotations

import numpy as np

from src.optimization.section_grouping.features import beam_assignment_vector, make_geometry_context
from src.optimization.section_grouping.models import SectionGroupingRuleset
from src.structure.section_library import SectionLibrary
from src.structure.structural_model import BEAM_TYPE_MAP, StructuralModel


def assign_beam_groups(model: StructuralModel, ruleset: SectionGroupingRuleset) -> dict[int, str]:
    """按稳定规则为当前模型中的梁动态分组。

    分配只依赖梁类型和几何向量，不依赖 beam_id，也不依赖当前模型的 FEA 结果。
    """

    prototypes_by_type = ruleset.prototypes_by_type()
    context = make_geometry_context(model)
    assignments: dict[int, str] = {}
    for beam_id, beam in model.beams.items():
        if beam.is_virtual:
            continue
        prototypes = prototypes_by_type.get(beam.beam_type, [])
        if not prototypes and ruleset.n_groups_by_type.get(beam.beam_type, 0) == 1:
            assignments[beam_id] = f"{beam.beam_type.name}-SG1"
            continue
        if not prototypes:
            continue
        vector = beam_assignment_vector(model, beam, context)
        distances = [
            float(np.linalg.norm(vector - np.array(prototype.prototype_centroid, dtype=float)))
            for prototype in prototypes
        ]
        assignments[beam_id] = prototypes[int(np.argmin(distances))].group_id
    return assignments


def apply_group_sections(
    model: StructuralModel,
    section_lib: SectionLibrary,
    ruleset: SectionGroupingRuleset,
    section_indices_by_group: dict[str, int],
) -> dict[int, str]:
    """按分组规则更新模型中每根梁的截面。

    返回 beam_id -> group_id，便于调试和可视化。没有匹配到 prototype 的梁保持
    原截面不变。
    """

    assignments = assign_beam_groups(model, ruleset)
    group_types = ruleset.group_beam_types()
    for beam_id, group_id in assignments.items():
        if group_id not in section_indices_by_group:
            continue
        beam_type = group_types[group_id]
        section_type = BEAM_TYPE_MAP[beam_type]
        model.beams[beam_id].section = section_lib.get_section(
            section_type,
            int(section_indices_by_group[group_id]),
        )
    return assignments


def section_counts_for_groups(ruleset: SectionGroupingRuleset, section_lib: SectionLibrary) -> list[int]:
    group_types = ruleset.group_beam_types()
    return [
        section_lib.section_counts[BEAM_TYPE_MAP[group_types[group_id]]]
        for group_id in ruleset.group_ids()
    ]
