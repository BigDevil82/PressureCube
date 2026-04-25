from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import RobustScaler

from src.analysis.beam_grouping.features import extract_beam_features, group_features_by_symmetry
from src.analysis.beam_grouping.models import (
    BeamFeature,
    BeamGroupConfig,
    BeamGroupingResult,
    BeamGroupSummary,
)
from src.analysis.results import AnalysisResults
from src.structure.enums import BeamType
from src.structure.structural_model import StructuralModel


class BeamGroupClusterer:
    """基于受力、变形和空间位置的梁分组器。

    算法流程：
    1. 为每根非虚梁提取 BeamFeature。
    2. 按 symmetry_key 把对称梁合并为 symmetry orbit。
    3. 对每一种 BeamType 独立聚类，避免框梁、主梁、次梁、支撑梁混在一起。
    4. 聚类对象是 orbit 的均值特征，而不是单根梁。
    5. 将 orbit 的聚类标签回填到 orbit 内每根梁，得到 beam_id -> group_id。

    这种做法适合“同一类梁内部进一步细分截面”的探索分析。对于拓扑会变化
    的优化问题，不应把当前结果中的 beam_id -> group_id 当成跨模型稳定规则。
    """

    def __init__(self, config: BeamGroupConfig):
        self.config = config

    def fit_predict(
        self,
        model: StructuralModel,
        strength_results: AnalysisResults,
        service_results: AnalysisResults,
    ) -> BeamGroupingResult:
        features = extract_beam_features(
            model,
            strength_results,
            service_results,
            mirror_x=self.config.mirror_x,
            mirror_y=self.config.mirror_y,
            mirror_z=self.config.mirror_z,
            symmetry_round_decimals=self.config.symmetry_round_decimals,
        )
        orbits = group_features_by_symmetry(features)
        orbit_keys_by_type: dict[BeamType, list[tuple]] = defaultdict(list)
        for key, members in orbits.items():
            orbit_keys_by_type[members[0].beam_type].append(key)

        beam_to_group: dict[int, str] = {}
        for beam_type, orbit_keys in orbit_keys_by_type.items():
            requested_groups = self.config.n_groups_by_type.get(beam_type, 1)
            n_groups = max(1, min(int(requested_groups), len(orbit_keys)))
            # 每种构件类型单独聚类，实际组数不能超过该类型的 orbit 数量。
            labels = self._cluster_orbits([orbits[key] for key in orbit_keys], n_groups)
            for orbit_key, label in zip(orbit_keys, labels):
                group_id = f"{beam_type.name}-G{int(label) + 1}"
                for feature in orbits[orbit_key]:
                    beam_to_group[feature.beam_id] = group_id

        summaries = self._build_summaries(features, beam_to_group)
        return BeamGroupingResult(
            beam_to_group=beam_to_group,
            group_summaries=summaries,
            features=features,
            feature_weights={
                "stress": self.config.stress_weight,
                "displacement": self.config.displacement_weight,
                "spatial": self.config.spatial_weight,
            },
            n_groups_by_type=self.config.n_groups_by_type,
        )

    def _cluster_orbits(self, orbit_members: list[list[BeamFeature]], n_groups: int) -> np.ndarray:
        """对 symmetry orbit 聚类。

        RobustScaler 用于降低特征量纲和极端值的影响；Ward 层次聚类适合当前
        这种样本数量不大、希望组内特征尽量接近的场景。
        """

        if n_groups == 1:
            return np.zeros(len(orbit_members), dtype=int)

        matrix = np.array([self._orbit_vector(members) for members in orbit_members], dtype=float)
        matrix = RobustScaler().fit_transform(matrix)
        model = AgglomerativeClustering(n_clusters=n_groups, linkage="ward")
        return model.fit_predict(matrix)

    def _orbit_vector(self, members: list[BeamFeature]) -> np.ndarray:
        """把一个 symmetry orbit 压缩为聚类特征向量。

        orbit 内可能有多根对称梁，因此这里取均值作为该 orbit 的代表特征。
        权重在缩放前乘入，用于表达应力、位移、空间位置三类因素的重要性。
        """

        stress_ratio = np.mean([item.stress_ratio for item in members])
        stress_mpa = np.mean([item.stress_mpa for item in members])
        disp_max = np.mean([item.displacement_max_mm for item in members])
        disp_mean = np.mean([item.displacement_mean_mm for item in members])
        rel_disp = np.mean([item.relative_displacement_mm for item in members])
        midpoint = np.mean([item.normalized_midpoint for item in members], axis=0)
        length_norm = np.mean([item.length_norm for item in members])

        return np.array(
            [
                stress_ratio * self.config.stress_weight,
                stress_mpa * self.config.stress_weight,
                disp_max * self.config.displacement_weight,
                disp_mean * self.config.displacement_weight,
                rel_disp * self.config.displacement_weight,
                midpoint[0] * self.config.spatial_weight,
                midpoint[1] * self.config.spatial_weight,
                midpoint[2] * self.config.spatial_weight,
                length_norm * self.config.spatial_weight,
            ]
        )

    def _build_summaries(
        self, features: dict[int, BeamFeature], beam_to_group: dict[int, str]
    ) -> list[BeamGroupSummary]:
        """生成分组解释信息。

        p90 用于描述组内大部分梁的受力/变形水平，max 用于暴露局部控制梁。
        control_factor 是一个粗略解释字段，只表示该组更偏受力控制还是变形控制。
        """

        grouped: dict[str, list[BeamFeature]] = defaultdict(list)
        for beam_id, group_id in beam_to_group.items():
            grouped[group_id].append(features[beam_id])

        summaries = []
        for group_id, members in sorted(grouped.items()):
            stress_ratios = np.array([item.stress_ratio for item in members], dtype=float)
            disps = np.array([item.displacement_max_mm for item in members], dtype=float)
            centers = np.array([item.midpoint for item in members], dtype=float)
            stress_p90 = float(np.percentile(stress_ratios, 90))
            disp_p90 = float(np.percentile(disps, 90))
            stress_score = stress_p90 / (float(np.max(stress_ratios)) + 1e-9)
            disp_score = disp_p90 / (float(np.max(disps)) + 1e-9)
            control_factor = "stress" if stress_score >= disp_score else "displacement"
            summaries.append(
                BeamGroupSummary(
                    group_id=group_id,
                    beam_type=members[0].beam_type,
                    beam_count=len(members),
                    stress_ratio_mean=float(np.mean(stress_ratios)),
                    stress_ratio_p90=stress_p90,
                    stress_ratio_max=float(np.max(stress_ratios)),
                    displacement_mean_mm=float(np.mean(disps)),
                    displacement_p90_mm=disp_p90,
                    displacement_max_mm=float(np.max(disps)),
                    representative_center=tuple(np.mean(centers, axis=0).tolist()),
                    symmetry_orbit_count=len({item.symmetry_key for item in members}),
                    control_factor=control_factor,
                )
            )
        return summaries
