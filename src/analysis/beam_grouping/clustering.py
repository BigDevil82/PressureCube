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
        if n_groups == 1:
            return np.zeros(len(orbit_members), dtype=int)

        matrix = np.array([self._orbit_vector(members) for members in orbit_members], dtype=float)
        matrix = RobustScaler().fit_transform(matrix)
        model = AgglomerativeClustering(n_clusters=n_groups, linkage="ward")
        return model.fit_predict(matrix)

    def _orbit_vector(self, members: list[BeamFeature]) -> np.ndarray:
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
