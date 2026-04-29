from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import RobustScaler

from src.analysis.opensees_analysis import OpenSeesSolver
from src.analysis.results import AnalysisResults
from src.optimization.design_space import get_layout_adapter, get_section_lib
from src.structure.section_grouping.beam_features import (
    BeamFeature,
    extract_beam_features,
    group_features_by_symmetry,
)
from src.structure.section_grouping.features import (
    ASSIGNMENT_FEATURE_SCHEMA,
    beam_type_sort_key,
    feature_assignment_vector,
)
from src.structure.section_grouping.models import BeamGroupPrototype, SectionGroupingRuleset
from src.structure.enums import BeamType
from src.structure.mesh_generator import MeshGenerator
from src.structure.structural_design import StructuralDesign
from src.structure.structural_model import StructuralModel


@dataclass(frozen=True)
class RepresentativeAnalysis:
    """一个用于发现分组规律的代表模型及分析结果。"""

    model: StructuralModel
    strength_results: AnalysisResults
    service_results: AnalysisResults


@dataclass(frozen=True)
class SectionGroupingDiscoveryResult:
    ruleset: SectionGroupingRuleset
    sample_beam_to_group: list[dict[int, str]]


@dataclass(frozen=True)
class SectionGroupingDiscoveryConfig:
    """稳定截面分组规则发现配置。"""

    n_groups_by_type: dict[BeamType, int]
    stress_weight: float = 1.0
    displacement_weight: float = 0.7
    spatial_weight: float = 0.35
    mirror_x: bool = True
    mirror_y: bool = False
    mirror_z: bool = True
    symmetry_round_decimals: int = 3
    description: str = ""
    strength_combo: dict[str, float] = field(default_factory=lambda: {"DL": 1.3, "LL": 1.5, "PL": 1.25})
    service_combo: dict[str, float] = field(default_factory=lambda: {"DL": 1.0, "LL": 1.0, "PL": 1.0})


@dataclass(frozen=True)
class _OrbitRecord:
    sample_index: int
    beam_type: BeamType
    features: list[BeamFeature]
    assignment_vector: np.ndarray
    discovery_vector: np.ndarray


def discover_section_group_rules(
    samples: Iterable[RepresentativeAnalysis],
    config: SectionGroupingDiscoveryConfig,
    *,
    layout_strategy: str,
) -> SectionGroupingRuleset:
    """从多个代表性已分析模型中发现稳定梁分组规则。

    发现阶段使用应力、位移和几何共同聚类；生成的 ruleset 只保留几何原型，
    供后续不同拓扑的新模型在 FEA 前动态分配截面组。
    """

    return discover_section_group_rules_with_assignments(
        samples,
        config,
        layout_strategy=layout_strategy,
    ).ruleset


def discover_section_group_rules_with_assignments(
    samples: Iterable[RepresentativeAnalysis],
    config: SectionGroupingDiscoveryConfig,
    *,
    layout_strategy: str,
) -> SectionGroupingDiscoveryResult:
    """发现稳定分组规则，并保留代表样本上的原始聚类分组。"""

    sample_list = list(samples)
    records = _collect_orbit_records(sample_list, config)
    records_by_type: dict[BeamType, list[_OrbitRecord]] = defaultdict(list)
    for record in records:
        records_by_type[record.beam_type].append(record)

    prototypes: list[BeamGroupPrototype] = []
    sample_beam_to_group: list[dict[int, str]] = [dict() for _ in sample_list]
    for beam_type in sorted(records_by_type, key=beam_type_sort_key):
        type_records = records_by_type[beam_type]
        requested_groups = int(config.n_groups_by_type.get(beam_type, 1))
        n_groups = max(1, min(requested_groups, len(type_records)))
        labels = _cluster_records(type_records, n_groups)
        type_prototypes, label_to_group_id = _build_type_prototypes(beam_type, type_records, labels)
        prototypes.extend(type_prototypes)
        for record, label in zip(type_records, labels):
            group_id = label_to_group_id[int(label)]
            for feature in record.features:
                sample_beam_to_group[record.sample_index][feature.beam_id] = group_id

    return SectionGroupingDiscoveryResult(
        ruleset=SectionGroupingRuleset(
            layout_strategy=layout_strategy,
            n_groups_by_type=config.n_groups_by_type,
            prototypes=prototypes,
            feature_schema=ASSIGNMENT_FEATURE_SCHEMA,
            created_from_samples=len(sample_list),
            description=config.description,
        ),
        sample_beam_to_group=sample_beam_to_group,
    )


def build_rules_from_designs(
    designs: Iterable[StructuralDesign],
    discovery_config: SectionGroupingDiscoveryConfig,
    *,
    mesh_size: float = 0.5,
    section_lib_path: str | None = None,
) -> SectionGroupingRuleset:
    """从若干 StructuralDesign 自动建模、分析并发现分组规则。

    这是离线规则发现入口，不建议在优化内层循环中调用。
    """

    design_list = list(designs)
    if not design_list:
        raise ValueError("At least one representative design is required.")

    layout_strategy = design_list[0].layout_strategy
    if any(design.layout_strategy != layout_strategy for design in design_list):
        raise ValueError("Representative designs must use the same layout strategy.")

    section_lib = get_section_lib(layout_strategy, section_lib_path)
    samples = []
    for design in design_list:
        model = design.build_model(section_lib)
        model = MeshGenerator(model, mesh_size).generate()

        solver = OpenSeesSolver(model)
        strength_results = solver.run_load_combination(
            discovery_config.strength_combo,
            pressure_pa=design.pressure,
        )
        service_results = solver.run_load_combination(
            discovery_config.service_combo,
            pressure_pa=design.pressure,
        )
        samples.append(RepresentativeAnalysis(model, strength_results, service_results))

    return discover_section_group_rules(samples, discovery_config, layout_strategy=layout_strategy)


def _collect_orbit_records(
    samples: list[RepresentativeAnalysis],
    config: SectionGroupingDiscoveryConfig,
) -> list[_OrbitRecord]:
    records = []
    for sample_index, sample in enumerate(samples):
        features = extract_beam_features(
            sample.model,
            sample.strength_results,
            sample.service_results,
            mirror_x=config.mirror_x,
            mirror_y=config.mirror_y,
            mirror_z=config.mirror_z,
            symmetry_round_decimals=config.symmetry_round_decimals,
        )
        for members in group_features_by_symmetry(features).values():
            assignment_vectors = np.array([feature_assignment_vector(item) for item in members])
            assignment_vector = np.mean(assignment_vectors, axis=0)
            records.append(
                _OrbitRecord(
                    sample_index=sample_index,
                    beam_type=members[0].beam_type,
                    features=members,
                    assignment_vector=assignment_vector,
                    discovery_vector=_discovery_vector(members, assignment_vector, config),
                )
            )
    return records


def _discovery_vector(
    members: list[BeamFeature],
    assignment_vector: np.ndarray,
    config: SectionGroupingDiscoveryConfig,
) -> np.ndarray:
    stress_ratio = np.mean([item.stress_ratio for item in members])
    stress_mpa = np.mean([item.stress_mpa for item in members])
    disp_max = np.mean([item.displacement_max_mm for item in members])
    disp_mean = np.mean([item.displacement_mean_mm for item in members])
    rel_disp = np.mean([item.relative_displacement_mm for item in members])
    return np.concatenate(
        [
            np.array(
                [
                    stress_ratio * config.stress_weight,
                    stress_mpa * config.stress_weight,
                    disp_max * config.displacement_weight,
                    disp_mean * config.displacement_weight,
                    rel_disp * config.displacement_weight,
                ],
                dtype=float,
            ),
            assignment_vector * config.spatial_weight,
        ]
    )


def _cluster_records(records: list[_OrbitRecord], n_groups: int) -> np.ndarray:
    if n_groups == 1:
        return np.zeros(len(records), dtype=int)
    matrix = np.array([record.discovery_vector for record in records], dtype=float)
    matrix = RobustScaler().fit_transform(matrix)
    return AgglomerativeClustering(n_clusters=n_groups, linkage="ward").fit_predict(matrix)


def _build_type_prototypes(
    beam_type: BeamType,
    records: list[_OrbitRecord],
    labels: np.ndarray,
) -> tuple[list[BeamGroupPrototype], dict[int, str]]:
    raw_prototypes = []
    for label in sorted(set(int(value) for value in labels)):
        members = [record for record, record_label in zip(records, labels) if int(record_label) == label]
        all_features = [feature for record in members for feature in record.features]
        assignment_centroid = np.mean([record.assignment_vector for record in members], axis=0)
        stress_ratios = np.array([feature.stress_ratio for feature in all_features], dtype=float)
        disps = np.array([feature.displacement_max_mm for feature in all_features], dtype=float)
        centers = np.array([feature.midpoint for feature in all_features], dtype=float)
        demand_score = float(np.percentile(stress_ratios, 90) + np.percentile(disps, 90) / 100.0)
        raw_prototypes.append(
            {
                "assignment_centroid": assignment_centroid,
                "label": label,
                "all_features": all_features,
                "stress_ratios": stress_ratios,
                "disps": disps,
                "centers": centers,
                "demand_score": demand_score,
            }
        )

    # 组号按需求从高到低排序，SG1 表示该类型中相对更高需求的组。
    raw_prototypes.sort(key=lambda item: item["demand_score"], reverse=True)

    prototypes = []
    label_to_group_id = {}
    for index, item in enumerate(raw_prototypes, start=1):
        group_id = f"{beam_type.name}-SG{index}"
        label_to_group_id[int(item["label"])] = group_id
        stress_ratios = item["stress_ratios"]
        disps = item["disps"]
        stress_p90 = float(np.percentile(stress_ratios, 90))
        disp_p90 = float(np.percentile(disps, 90))
        stress_score = stress_p90 / (float(np.max(stress_ratios)) + 1e-9)
        disp_score = disp_p90 / (float(np.max(disps)) + 1e-9)
        prototypes.append(
            BeamGroupPrototype(
                group_id=group_id,
                beam_type=beam_type,
                prototype_centroid=tuple(float(value) for value in item["assignment_centroid"]),
                member_count=len(item["all_features"]),
                representative_center=tuple(np.mean(item["centers"], axis=0).tolist()),
                stress_ratio_mean=float(np.mean(stress_ratios)),
                stress_ratio_p90=stress_p90,
                stress_ratio_max=float(np.max(stress_ratios)),
                displacement_mean_mm=float(np.mean(disps)),
                displacement_p90_mm=disp_p90,
                displacement_max_mm=float(np.max(disps)),
                control_factor="stress" if stress_score >= disp_score else "displacement",
            )
        )
    return prototypes, label_to_group_id


if __name__ == "__main__":
    discovery_config = SectionGroupingDiscoveryConfig(
        n_groups_by_type={
            BeamType.FRAME: 2,
            BeamType.MAIN: 2,
            BeamType.SECONDARY: 3,
            BeamType.BRACE: 2,
        },
        description="engineering baseline grouping",
    )

    adapter = get_layout_adapter("engineering")
    design = adapter.build_design_from_sections(
        geometry={"L": 8.9, "W": 3.7, "H": 3.3},
        pressure=30000.0,
        topology={
            "main_beam_spacing": 0.95,
            "support_beam_max_spacing": 0.75,
            "add_support_beam": True,
        },
        sections=[2, 8, 7, 14],
    )

    rules = build_rules_from_designs(
        [design],
        discovery_config,
        mesh_size=0.5,
    )
    rules.save_json("results/section_grouping/engineering_rules.json")
