from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.structure.enums import BeamType


@dataclass(frozen=True)
class BeamGroupPrototype:
    """稳定分组规则中的一个梁组原型。

    discovery 阶段用应力、位移和几何信息确定分组；应用到新拓扑时，只能使用
    几何信息。因此 prototype_centroid 存储的是几何判别向量的中心，用于把
    新模型中的梁分配到最近的稳定组。
    """

    group_id: str
    beam_type: BeamType
    prototype_centroid: tuple[float, ...]
    member_count: int
    representative_center: tuple[float, float, float]
    stress_ratio_mean: float
    stress_ratio_p90: float
    stress_ratio_max: float
    displacement_mean_mm: float
    displacement_p90_mm: float
    displacement_max_mm: float
    control_factor: str


@dataclass(frozen=True)
class SectionGroupingRuleset:
    """可跨拓扑复用的梁分组规则集。

    规则集不保存 beam_id，因为 beam_id 随拓扑和网格变化不稳定。它保存每个
    梁组的几何原型，并在新模型中通过“梁类型 + 几何最近原型”动态分配组别。
    """

    layout_strategy: str
    n_groups_by_type: dict[BeamType, int]
    prototypes: list[BeamGroupPrototype]
    feature_schema: list[str]
    created_from_samples: int
    description: str = ""

    @classmethod
    def by_beam_type(cls, layout_strategy: str) -> "SectionGroupingRuleset":
        """Default ruleset: one section group for each BeamType.

        This makes the historical "four section variables" case the same as the
        grouped case with four groups: FRAME-SG1, MAIN-SG1, SECONDARY-SG1,
        BRACE-SG1.
        """

        return cls(
            layout_strategy=layout_strategy,
            n_groups_by_type={beam_type: 1 for beam_type in BeamType},
            prototypes=[],
            feature_schema=[],
            created_from_samples=0,
            description="Default one group per beam type.",
        )

    def prototypes_by_type(self) -> dict[BeamType, list[BeamGroupPrototype]]:
        grouped: dict[BeamType, list[BeamGroupPrototype]] = {}
        for prototype in self.prototypes:
            grouped.setdefault(prototype.beam_type, []).append(prototype)
        return grouped

    def group_ids(self) -> list[str]:
        if self.prototypes:
            return [prototype.group_id for prototype in self.prototypes]
        group_ids = []
        for beam_type in sorted(self.n_groups_by_type, key=lambda item: int(item.value)):
            for index in range(1, int(self.n_groups_by_type[beam_type]) + 1):
                group_ids.append(f"{beam_type.name}-SG{index}")
        return group_ids

    def group_beam_types(self) -> dict[str, BeamType]:
        if self.prototypes:
            return {prototype.group_id: prototype.beam_type for prototype in self.prototypes}
        return {
            f"{beam_type.name}-SG{index}": beam_type
            for beam_type, count in self.n_groups_by_type.items()
            for index in range(1, int(count) + 1)
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout_strategy": self.layout_strategy,
            "n_groups_by_type": {key.name: value for key, value in self.n_groups_by_type.items()},
            "prototypes": [
                {
                    **asdict(prototype),
                    "beam_type": prototype.beam_type.name,
                }
                for prototype in self.prototypes
            ],
            "feature_schema": self.feature_schema,
            "created_from_samples": self.created_from_samples,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SectionGroupingRuleset":
        return cls(
            layout_strategy=data["layout_strategy"],
            n_groups_by_type={
                BeamType[key]: int(value) for key, value in data.get("n_groups_by_type", {}).items()
            },
            prototypes=[
                BeamGroupPrototype(
                    **{
                        **item,
                        "beam_type": BeamType[item["beam_type"]],
                        "prototype_centroid": tuple(item["prototype_centroid"]),
                        "representative_center": tuple(item["representative_center"]),
                    }
                )
                for item in data.get("prototypes", [])
            ],
            feature_schema=list(data.get("feature_schema", [])),
            created_from_samples=int(data.get("created_from_samples", 0)),
            description=data.get("description", ""),
        )

    def save_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> "SectionGroupingRuleset":
        with open(path, "r", encoding="utf-8") as file:
            return cls.from_dict(json.load(file))
