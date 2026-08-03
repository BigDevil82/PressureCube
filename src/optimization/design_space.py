from __future__ import annotations

from dataclasses import dataclass, replace
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

    def build_params(
        self,
        *,
        geometry: dict[str, float],
        topology: dict[str, Any],
        base_params: BaseLayoutParameters | None = None,
    ) -> BaseLayoutParameters:
        if base_params is not None:
            return replace(
                base_params,
                main_beam_spacing=topology["main_beam_spacing"],
                add_support_beam=topology["add_support_beam"],
                **self._extra_param_kwargs(topology),
            )
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
        base_params: BaseLayoutParameters | None = None,
    ) -> StructuralDesign:
        params = self.build_params(geometry=geometry, topology=topology, base_params=base_params)
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

    def build_design_with_model(
        self,
        *,
        geometry: dict[str, float],
        pressure: float,
        topology: dict[str, Any],
        section_group_rules: SectionGroupingRuleset,
        section_indices_by_group: dict[str, int],
        base_params: BaseLayoutParameters | None = None,
    ) -> tuple[StructuralDesign, StructuralModel]:
        params = self.build_params(geometry=geometry, topology=topology, base_params=base_params)
        return build_design_with_model(
            params,
            pressure,
            section_group_rules,
            section_indices_by_group,
            get_section_lib(params.layout_strategy),
        )

    def build_design_from_sections(
        self,
        *,
        geometry: dict[str, float],
        pressure: float,
        topology: dict[str, Any],
        sections: list[int],
        base_params: BaseLayoutParameters | None = None,
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
            base_params=base_params,
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


def build_design_with_model(
    params: BaseLayoutParameters,
    pressure: float,
    ruleset: SectionGroupingRuleset,
    section_indices_by_group: dict[str, int],
    section_lib: SectionLibrary,
) -> tuple[StructuralDesign, StructuralModel]:
    model = StructuralModel(params, section_lib, section_assignment=zero_assignment())
    model.generate()
    assignment = SectionAssignment.from_group_indices(
        ruleset.group_beam_types(),
        section_indices_by_group,
        beam_groups=assign_beam_groups(model, ruleset),
    )
    model.apply_section_assignment(assignment)
    design = StructuralDesign(
        params=params,
        pressure=pressure,
        section_assignment=assignment,
    )
    return design, model


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


def design_to_group_feature_vector(
    design: StructuralDesign,
    max_groups_per_type: int = 3,
) -> np.ndarray:
    """Feature vector for grouped-section surrogate datasets.

    The legacy feature vector collapses all groups of the same BeamType to a
    mean section property. That loses the section distribution when group count
    varies. This feature keeps a fixed number of per-group slots for every beam
    type, then adds per-type distribution statistics and load-demand
    interactions. It is intended for newly generated grouped datasets and does
    not replace design_to_feature_vector used by existing surrogate models.
    """

    p = design.params
    lib = get_section_lib(p.layout_strategy)
    adapter = get_layout_adapter(p.layout_strategy)

    volume = p.length * p.width * p.height
    floor_area = p.length * p.width
    aspect_ratio = p.length / p.width if p.width > 0 else 1.0
    pressure_kpa = design.pressure / 1000.0
    total_load_mn = design.pressure * floor_area / 1e6
    n_main_beams = int(p.length / p.main_beam_spacing) + 1

    features = [
        p.length,
        p.width,
        p.height,
        volume,
        floor_area,
        aspect_ratio,
        pressure_kpa,
        total_load_mn,
        p.main_beam_spacing,
        *adapter.topology_feature_values(p),
        float(n_main_beams),
    ]

    for beam_type in sorted(BEAM_SECTION_TYPES, key=lambda item: int(item.value)):
        group_features = _group_section_features(
            design.section_assignment,
            lib,
            beam_type,
            max_groups_per_type,
            total_load_mn,
            pressure_kpa,
            floor_area,
        )
        features.extend(group_features)

    return np.array(features, dtype=np.float32)


def group_feature_names(layout_strategy: str, max_groups_per_type: int = 3) -> list[str]:
    adapter = get_layout_adapter(layout_strategy)
    names = [
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
    ]

    for beam_type in sorted(BEAM_SECTION_TYPES, key=lambda item: int(item.value)):
        prefix = _beam_type_prefix(beam_type)
        names.extend([f"{prefix}_n_groups", f"{prefix}_n_groups_norm"])
        for slot in range(1, max_groups_per_type + 1):
            names.extend(
                [
                    f"{prefix}_g{slot}_active",
                    f"{prefix}_g{slot}_section_index_norm",
                    f"{prefix}_g{slot}_area",
                    f"{prefix}_g{slot}_Ix",
                    f"{prefix}_g{slot}_Iy",
                    f"{prefix}_g{slot}_log_area",
                    f"{prefix}_g{slot}_log_Ix",
                    f"{prefix}_g{slot}_log_Iy",
                    f"{prefix}_g{slot}_load_per_area",
                    f"{prefix}_g{slot}_load_per_Ix",
                    f"{prefix}_g{slot}_pressure_area_demand",
                ]
            )
        for prop in ["area", "Ix", "Iy"]:
            names.extend(
                [
                    f"{prefix}_{prop}_mean",
                    f"{prefix}_{prop}_min",
                    f"{prefix}_{prop}_max",
                    f"{prefix}_{prop}_std",
                    f"{prefix}_{prop}_cv",
                    f"{prefix}_{prop}_min_to_max",
                ]
            )
        names.extend(
            [
                f"{prefix}_weakest_load_per_area",
                f"{prefix}_weakest_load_per_Ix",
                f"{prefix}_weakest_pressure_area_demand",
            ]
        )
    return names


def design_to_group_feature_vector_v2(
    design: StructuralDesign,
    max_groups_per_type: int = 3,
) -> np.ndarray:
    """Grouped feature vector with section-group geometry aggregates.

    v1 describes which sections each group uses, but not which beams are
    assigned to that group. Stress is often controlled by local members, so v2
    appends fixed-slot geometric summaries for every section group.
    """

    lib = get_section_lib(design.layout_strategy)
    model = StructuralModel(design.params, lib, section_assignment=design.section_assignment)
    model.generate_beams()
    base = design_to_group_feature_vector(design, max_groups_per_type)
    geom = []
    for beam_type in sorted(BEAM_SECTION_TYPES, key=lambda item: int(item.value)):
        geom.extend(_group_geometry_features(model, beam_type, max_groups_per_type, design.pressure))
    return np.concatenate([base, np.array(geom, dtype=np.float32)]).astype(np.float32)


def group_feature_names_v2(layout_strategy: str, max_groups_per_type: int = 3) -> list[str]:
    names = group_feature_names(layout_strategy, max_groups_per_type)
    geom_names = [
        "beam_count",
        "beam_count_norm",
        "length_sum_norm",
        "length_mean_norm",
        "length_max_norm",
        "x_mean",
        "y_mean",
        "z_mean",
        "x_std",
        "y_std",
        "z_std",
        "edge_proximity_mean",
        "edge_proximity_min",
        "corner_proximity_mean",
        "corner_proximity_min",
        "ori_x_frac",
        "ori_y_frac",
        "ori_z_frac",
        "face_horizontal_frac",
        "face_longitudinal_frac",
        "face_end_frac",
        "face_other_frac",
        "length_per_area",
        "length_per_Ix",
        "pressure_length_per_area",
    ]
    for beam_type in sorted(BEAM_SECTION_TYPES, key=lambda item: int(item.value)):
        prefix = _beam_type_prefix(beam_type)
        for slot in range(1, max_groups_per_type + 1):
            names.extend([f"{prefix}_g{slot}_{name}" for name in geom_names])
    return names


def _section_feature_stats(assignment: SectionAssignment, lib: SectionLibrary) -> dict[str, list[float]]:
    groups_by_type: dict[BeamType, list[str]] = {beam_type: [] for beam_type in BEAM_SECTION_TYPES}
    for group_id, group in assignment.groups.items():
        groups_by_type[group.beam_type].append(group_id)

    areas = []
    ixs = []
    iys = []
    for beam_type in sorted(BEAM_SECTION_TYPES, key=lambda item: int(item.value)):
        sections = [
            lib.get_section(BEAM_SECTION_TYPES[beam_type], assignment.groups[group_id].section_index)
            for group_id in groups_by_type[beam_type]
        ]
        areas.append(float(np.mean([section.area * 1e4 for section in sections])))
        ixs.append(float(np.mean([section.Ix * 1e8 for section in sections])))
        iys.append(float(np.mean([section.Iy * 1e8 for section in sections])))

    return {"areas": areas, "ixs": ixs, "iys": iys}


def _group_section_features(
    assignment: SectionAssignment,
    lib: SectionLibrary,
    beam_type: BeamType,
    max_groups_per_type: int,
    total_load_mn: float,
    pressure_kpa: float,
    floor_area: float,
) -> list[float]:
    section_type = BEAM_SECTION_TYPES[beam_type]
    groups = [
        group
        for group in assignment.groups.values()
        if group.beam_type == beam_type
    ]
    groups.sort(key=lambda group: group.group_id)

    values = [
        lib.get_section(section_type, group.section_index)
        for group in groups
    ]
    section_count = lib.section_counts[section_type]

    areas = np.array([section.area * 1e4 for section in values], dtype=float)
    ixs = np.array([section.Ix * 1e8 for section in values], dtype=float)
    iys = np.array([section.Iy * 1e8 for section in values], dtype=float)

    features = [float(len(groups)), float(len(groups) / max_groups_per_type)]
    for slot in range(max_groups_per_type):
        if slot < len(groups):
            group = groups[slot]
            section = values[slot]
            area = section.area * 1e4
            ix = section.Ix * 1e8
            iy = section.Iy * 1e8
            features.extend(
                [
                    1.0,
                    float(group.section_index / max(section_count - 1, 1)),
                    float(area),
                    float(ix),
                    float(iy),
                    float(np.log1p(area)),
                    float(np.log1p(ix)),
                    float(np.log1p(iy)),
                    float(total_load_mn / (area + 1e-10)),
                    float(total_load_mn / (ix + 1e-10)),
                    float(pressure_kpa * floor_area / (area + 1e-10)),
                ]
            )
        else:
            features.extend([0.0] * 11)

    for prop_values in [areas, ixs, iys]:
        features.extend(_distribution_features(prop_values))

    features.extend(
        [
            float(total_load_mn / (np.min(areas) + 1e-10)),
            float(total_load_mn / (np.min(ixs) + 1e-10)),
            float(pressure_kpa * floor_area / (np.min(areas) + 1e-10)),
        ]
    )
    return features


def _distribution_features(values: np.ndarray) -> list[float]:
    mean = float(np.mean(values))
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    std = float(np.std(values))
    return [
        mean,
        min_value,
        max_value,
        std,
        std / (mean + 1e-10),
        min_value / (max_value + 1e-10),
    ]


def _group_geometry_features(
    model: StructuralModel,
    beam_type: BeamType,
    max_groups_per_type: int,
    pressure: float,
) -> list[float]:
    from src.structure.section_grouping.features import make_geometry_context

    context = make_geometry_context(model)
    groups = [
        group
        for group in model.section_assignment.groups.values()
        if group.beam_type == beam_type
    ]
    groups.sort(key=lambda group: group.group_id)

    beam_ids_by_group = {group.group_id: [] for group in groups}
    for beam_id, group_id in model.section_assignment.beam_groups.items():
        beam = model.beams.get(beam_id)
        if beam is not None and not beam.is_virtual and beam.beam_type == beam_type:
            beam_ids_by_group.setdefault(group_id, []).append(beam_id)

    type_beam_count = sum(len(ids) for ids in beam_ids_by_group.values())
    features: list[float] = []
    for slot in range(max_groups_per_type):
        if slot >= len(groups):
            features.extend([0.0] * 25)
            continue

        group = groups[slot]
        beam_ids = beam_ids_by_group.get(group.group_id, [])
        features.extend(_single_group_geometry_features(model, beam_ids, group, context, type_beam_count, pressure))
    return features


def _single_group_geometry_features(
    model: StructuralModel,
    beam_ids: list[int],
    group,
    context,
    type_beam_count: int,
    pressure: float,
) -> list[float]:
    from src.structure.section_grouping.beam_features import beam_orientation
    from src.structure.section_grouping.features import normalize_face_family

    if not beam_ids:
        return [0.0] * 25

    midpoints = []
    lengths = []
    orientations = []
    faces = []
    for beam_id in beam_ids:
        beam = model.beams[beam_id]
        p1 = model.points[beam.start_node].coordinates()
        p2 = model.points[beam.end_node].coordinates()
        midpoint = ((p1 + p2) / 2.0 - context.bounds_min) / context.span
        length = np.linalg.norm(p2 - p1) / max(context.length_ref, 1e-10)
        midpoints.append(midpoint)
        lengths.append(float(length))
        orientations.append(beam_orientation(p1, p2))
        faces.append(normalize_face_family(beam.on_face))

    midpoint_arr = np.array(midpoints, dtype=float)
    length_arr = np.array(lengths, dtype=float)
    boundary_dist = np.minimum(midpoint_arr, 1.0 - midpoint_arr)
    edge_proximity = np.min(boundary_dist, axis=1)
    corner_proximity = np.sum(boundary_dist, axis=1)
    section = model.section_lib.get_section(group.section_type, group.section_index)
    area = section.area * 1e4
    ix = section.Ix * 1e8
    length_sum = float(length_arr.sum())
    pressure_kpa = pressure / 1000.0

    return [
        float(len(beam_ids)),
        float(len(beam_ids) / max(type_beam_count, 1)),
        length_sum,
        float(length_arr.mean()),
        float(length_arr.max()),
        *[float(value) for value in midpoint_arr.mean(axis=0)],
        *[float(value) for value in midpoint_arr.std(axis=0)],
        float(edge_proximity.mean()),
        float(edge_proximity.min()),
        float(corner_proximity.mean()),
        float(corner_proximity.min()),
        *[float(np.mean([orientation == value for orientation in orientations])) for value in ("x", "y", "z")],
        *[float(np.mean([face == value for face in faces])) for value in ("horizontal", "longitudinal_side", "end_side", "other")],
        float(length_sum / (area + 1e-10)),
        float(length_sum / (ix + 1e-10)),
        float(pressure_kpa * length_sum / (area + 1e-10)),
    ]


def _beam_type_prefix(beam_type: BeamType) -> str:
    if beam_type == BeamType.SECONDARY:
        return "sec"
    return beam_type.name.lower()


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
