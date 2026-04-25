from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.structure.enums import BeamType


@dataclass(frozen=True)
class BeamGroupConfig:
    """梁分组配置。

    分组算法不是直接对单根梁聚类，而是先根据空间对称性把若干根梁合并为
    一个 symmetry orbit，再对 orbit 做聚类。这样可以避免左右、上下近似对
    称的梁被分到不同截面组。

    n_groups_by_type:
        每一种 BeamType 希望划分的组数。实际组数不会超过该类型可用的
        symmetry orbit 数量。
    stress_weight/displacement_weight/spatial_weight:
        聚类特征中受力、变形、空间位置三类信息的相对权重。
    mirror_x/mirror_y/mirror_z:
        是否把对应方向的镜像位置视为同一类对称位置。当前默认认为长度
        方向 x 和高度方向 z 具有镜像对称性，宽度方向 y 不强制镜像。
    """

    n_groups_by_type: Dict[BeamType, int]
    stress_weight: float = 1.0
    displacement_weight: float = 0.7
    spatial_weight: float = 0.35
    symmetry_round_decimals: int = 3
    mirror_x: bool = True
    mirror_y: bool = False
    mirror_z: bool = True


@dataclass(frozen=True)
class BeamFeature:
    """单根梁用于分组的特征及元数据。

    beam_id 仅用于把聚类结果映射回模型和可视化，不进入聚类特征向量。
    真正参与聚类的是应力、位移、归一化空间位置、长度以及 symmetry_key。
    """

    beam_id: int
    beam_type: BeamType
    orientation: str
    face: str
    midpoint: np.ndarray
    normalized_midpoint: np.ndarray
    length_norm: float
    stress_ratio: float
    stress_mpa: float
    displacement_max_mm: float
    displacement_mean_mm: float
    relative_displacement_mm: float
    symmetry_key: tuple


@dataclass(frozen=True)
class BeamGroupSummary:
    """一个梁组的统计摘要，用于解释该组的受力和控制因素。"""

    group_id: str
    beam_type: BeamType
    beam_count: int
    stress_ratio_mean: float
    stress_ratio_p90: float
    stress_ratio_max: float
    displacement_mean_mm: float
    displacement_p90_mm: float
    displacement_max_mm: float
    representative_center: tuple[float, float, float]
    symmetry_orbit_count: int
    control_factor: str


@dataclass(frozen=True)
class BeamGroupingResult:
    """梁分组结果。

    beam_to_group 是最终可用于截面分配或可视化的映射。features 和
    group_summaries 保留了分组依据，便于检查算法是否符合结构直觉。
    """

    beam_to_group: Dict[int, str]
    group_summaries: list[BeamGroupSummary]
    features: Dict[int, BeamFeature]
    feature_weights: Dict[str, float]
    n_groups_by_type: Dict[BeamType, int]
