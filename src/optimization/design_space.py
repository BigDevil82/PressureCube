from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from src.structure.section_grouping.application import assign_beam_groups
from src.structure.section_grouping.models import SectionGroupingRuleset
from src.structure.enums import BeamType
from src.structure.layout.params import (
    BaseLayoutParameters,
    EngineeringLayoutParameters,
    ResearchLayoutParameters,
)
from src.structure.section_assignment import BEAM_SECTION_TYPES, SectionAssignment, zero_assignment
from src.structure.section_library import SectionLibrary
from src.structure.structural_design import StructuralDesign
from src.structure.structural_model import StructuralModel


@dataclass(frozen=True)
class VariableSpec:
    name: str
    kind: Literal["categorical", "int", "float", "bool"]
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    choices: Sequence[Any] | None = None
    default: Any = None


class LayoutDesignAdapter:
    layout_strategy: str
    params_cls: type[BaseLayoutParameters]
    default_section_lib_path: str

    def topology_specs(self) -> list[VariableSpec]:
        raise NotImplementedError

    def topology_feature_names(self) -> list[str]:
        raise NotImplementedError

    def topology_feature_values(self, params: BaseLayoutParameters) -> list[float]:
        raise NotImplementedError

    def default_topology(self) -> dict[str, Any]:
        return {spec.name: spec.default for spec in self.topology_specs()}

    def build_params(self, *, geometry: dict[str, float], topology: dict[str, Any]) -> BaseLayoutParameters:
        return self.params_cls(
            length=geometry["L"],
            width=geometry["W"],
            height=geometry["H"],
            main_beam_spacing=topology["main_beam_spacing"],
            add_support_beam=topology["add_support_beam"],
            **self._extra_param_kwargs(topology),
        )

    def build_design(
        self,
        *,
        geometry: dict[str, float],
        pressure: float,
        topology: dict[str, Any],
        section_group_rules: SectionGroupingRuleset,
        section_indices_by_group: dict[str, int],
    ) -> StructuralDesign:
        params = self.build_params(geometry=geometry, topology=topology)
        assignment = build_assignment_for_params(
            params,
            section_group_rules,
            section_indices_by_group,
            get_section_lib(params.layout_strategy),
        )
        return StructuralDesign(
            params=params,
            pressure=pressure,
            section_assignment=assignment,
        )

    def build_design_from_sections(
        self,
        *,
        geometry: dict[str, float],
        pressure: float,
        topology: dict[str, Any],
        sections: list[int],
    ) -> StructuralDesign:
        rules = SectionGroupingRuleset.by_beam_type(self.layout_strategy)
        return self.build_design(
            geometry=geometry,
            pressure=pressure,
            topology=topology,
            section_group_rules=rules,
            section_indices_by_group={
                group_id: index for group_id, index in zip(rules.group_ids(), sections)
            },
        )

    def _extra_param_kwargs(self, topology: dict[str, Any]) -> dict[str, Any]:
        return {}


class ResearchLayoutAdapter(LayoutDesignAdapter):
    layout_strategy = "research"
    params_cls = ResearchLayoutParameters
    default_section_lib_path = "src/data/steel_sections_large.json"

    def topology_specs(self) -> list[VariableSpec]:
        return [
            VariableSpec(
                "main_beam_spacing",
                "categorical",
                # choices=[0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.30],
                choices=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
                default=0.75,
            ),
            VariableSpec("num_secondary_width", "int", low=2, high=6, default=4),
            VariableSpec("num_secondary_height", "int", low=2, high=6, default=4),
            VariableSpec("add_support_beam", "bool", choices=[True, False], default=True),
        ]

    def topology_feature_names(self) -> list[str]:
        return ["num_secondary_width", "num_secondary_height", "add_support_beam"]

    def topology_feature_values(self, params: BaseLayoutParameters) -> list[float]:
        assert isinstance(params, ResearchLayoutParameters)
        return [
            float(params.num_secondary_width),
            float(params.num_secondary_height),
            float(params.add_support_beam),
        ]

    def _extra_param_kwargs(self, topology: dict[str, Any]) -> dict[str, Any]:
        return {
            "num_secondary_width": topology["num_secondary_width"],
            "num_secondary_height": topology["num_secondary_height"],
        }


class EngineeringLayoutAdapter(LayoutDesignAdapter):
    layout_strategy = "engineering"
    params_cls = EngineeringLayoutParameters
    default_section_lib_path = "src/data/steel_sections.json"

    def topology_specs(self) -> list[VariableSpec]:
        return [
            VariableSpec(
                "main_beam_spacing",
                "categorical",
                choices=[0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.30],
                default=0.95,
            ),
            VariableSpec(
                "support_beam_max_spacing",
                "categorical",
                choices=[0.3, 0.35, 0.40, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
                default=0.5,
            ),
            VariableSpec("add_support_beam", "bool", choices=[True, False], default=True),
        ]

    def topology_feature_names(self) -> list[str]:
        return ["support_beam_max_spacing", "add_support_beam"]

    def topology_feature_values(self, params: BaseLayoutParameters) -> list[float]:
        assert isinstance(params, EngineeringLayoutParameters)
        return [float(params.support_beam_max_spacing), float(params.add_support_beam)]

    def _extra_param_kwargs(self, topology: dict[str, Any]) -> dict[str, Any]:
        return {"support_beam_max_spacing": topology["support_beam_max_spacing"]}


_ADAPTERS: dict[str, LayoutDesignAdapter] = {
    "research": ResearchLayoutAdapter(),
    "engineering": EngineeringLayoutAdapter(),
}

_SECTION_LIB_CACHE: dict[str, SectionLibrary] = {}


def get_section_lib(layout_strategy: str = "research", file_path: str | None = None) -> SectionLibrary:
    resolved_path = file_path or get_layout_adapter(layout_strategy).default_section_lib_path
    if resolved_path not in _SECTION_LIB_CACHE:
        _SECTION_LIB_CACHE[resolved_path] = SectionLibrary(resolved_path)
    return _SECTION_LIB_CACHE[resolved_path]


def get_layout_adapter(layout_strategy: str) -> LayoutDesignAdapter:
    return _ADAPTERS[layout_strategy]


def build_assignment_for_params(
    params: BaseLayoutParameters,
    ruleset: SectionGroupingRuleset,
    section_indices_by_group: dict[str, int],
    section_lib: SectionLibrary,
) -> SectionAssignment:
    model = StructuralModel(params, section_lib, section_assignment=zero_assignment())
    model.generate()
    return SectionAssignment.from_group_indices(
        ruleset.group_beam_types(),
        section_indices_by_group,
        beam_groups=assign_beam_groups(model, ruleset),
    )


def design_to_feature_vector(design: StructuralDesign) -> np.ndarray:
    p = design.params
    lib = get_section_lib(p.layout_strategy)
    adapter = get_layout_adapter(p.layout_strategy)
    section_stats = _section_feature_stats(design.section_assignment, lib)

    volume = p.length * p.width * p.height
    floor_area = p.length * p.width
    aspect_ratio = p.length / p.width if p.width > 0 else 1.0
    pressure_kpa = design.pressure / 1000.0
    total_load = design.pressure * floor_area
    n_main_beams = int(p.length / p.main_beam_spacing) + 1

    areas = section_stats["areas"]
    ixs = section_stats["ixs"]
    iys = section_stats["iys"]

    total_area = sum(areas)
    avg_ix = float(np.mean(ixs))
    min_ix = min(ixs)
    max_ix = max(ixs)
    ix_ratio = min_ix / (max_ix + 1e-10)
    stiffness_load_ratio = avg_ix / (total_load / 1e6 + 1e-10)
    strength_load_ratio = total_area / (total_load / 1e6 + 1e-10)

    features = [
        p.length,
        p.width,
        p.height,
        volume,
        floor_area,
        aspect_ratio,
        pressure_kpa,
        total_load / 1e6,
        p.main_beam_spacing,
        *adapter.topology_feature_values(p),
        float(n_main_beams),
        *areas,
        *ixs,
        *iys,
        total_area,
        avg_ix,
        min_ix,
        ix_ratio,
        stiffness_load_ratio,
        strength_load_ratio,
    ]
    return np.array(features, dtype=np.float32)


def feature_names(layout_strategy: str) -> list[str]:
    adapter = get_layout_adapter(layout_strategy)
    return [
        "L",
        "W",
        "H",
        "volume",
        "floor_area",
        "aspect_ratio",
        "pressure_kPa",
        "total_load_MN",
        "main_beam_spacing",
        *adapter.topology_feature_names(),
        "n_main_beams",
        "frame_area_mean",
        "main_area_mean",
        "sec_area_mean",
        "brace_area_mean",
        "frame_Ix_mean",
        "main_Ix_mean",
        "sec_Ix_mean",
        "brace_Ix_mean",
        "frame_Iy_mean",
        "main_Iy_mean",
        "sec_Iy_mean",
        "brace_Iy_mean",
        "total_area",
        "avg_Ix",
        "min_Ix",
        "Ix_ratio",
        "stiffness_load_ratio",
        "strength_load_ratio",
    ]


def _section_feature_stats(assignment: SectionAssignment, lib: SectionLibrary) -> dict[str, list[float]]:
    groups_by_type: dict[BeamType, list[str]] = {beam_type: [] for beam_type in BEAM_SECTION_TYPES}
    for group_id, group in assignment.groups.items():
        groups_by_type[group.beam_type].append(group_id)

    areas = []
    ixs = []
    iys = []
    for beam_type in sorted(BEAM_SECTION_TYPES, key=lambda item: int(item.value)):
        sections = [
            lib.get_section(BEAM_SECTION_TYPES[beam_type], assignment.section_indices[group_id])
            for group_id in groups_by_type[beam_type]
        ]
        areas.append(float(np.mean([section.area * 1e4 for section in sections])))
        ixs.append(float(np.mean([section.Ix * 1e8 for section in sections])))
        iys.append(float(np.mean([section.Iy * 1e8 for section in sections])))

    return {"areas": areas, "ixs": ixs, "iys": iys}


def sample_spec_from_unit(spec: VariableSpec, value: float) -> Any:
    value = min(max(value, 0.0), 0.999999)
    if spec.kind in {"categorical", "bool"}:
        choices = list(spec.choices or [])
        idx = min(int(value * len(choices)), len(choices) - 1)
        return choices[idx]
    if spec.kind == "int":
        assert spec.low is not None and spec.high is not None
        span = int(spec.high) - int(spec.low) + 1
        return min(int(spec.low) + int(value * span), int(spec.high))
    if spec.kind == "float":
        assert spec.low is not None and spec.high is not None
        raw = float(spec.low) + value * (float(spec.high) - float(spec.low))
        if spec.step:
            steps = round((raw - float(spec.low)) / float(spec.step))
            raw = float(spec.low) + steps * float(spec.step)
        return raw
    raise ValueError(f"Unsupported variable spec kind: {spec.kind}")


def suggest_spec_with_optuna(trial: Any, spec: VariableSpec) -> Any:
    if spec.kind in {"categorical", "bool"}:
        return trial.suggest_categorical(spec.name, list(spec.choices or []))
    if spec.kind == "int":
        return trial.suggest_int(spec.name, int(spec.low), int(spec.high))
    if spec.kind == "float":
        kwargs = {}
        if spec.step is not None:
            kwargs["step"] = spec.step
        return trial.suggest_float(spec.name, float(spec.low), float(spec.high), **kwargs)
    raise ValueError(f"Unsupported variable spec kind: {spec.kind}")
