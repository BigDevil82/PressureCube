# 稳定截面分组规则

本包解决的问题是：梁的 `beam_id -> group_id` 只对固定拓扑稳定，不能直接用于拓扑变量会变化的优化。因此这里采用两阶段策略：

1. 离线发现：对若干代表性模型做 FEA，用应力、位移和空间位置共同聚类，发现各类梁内部的分组规律。
2. 在线应用：把发现结果固化为几何原型。优化新模型时，只根据梁类型、方向、所在面、归一化中点和长度，把每根梁动态分配到最近的稳定组。

## 关键约束

- `beam_id` 不进入规则。
- 应力和位移只用于离线发现规则，不用于优化内层的组别分配。
- `SG1` 表示同一 `BeamType` 中发现阶段需求相对更高的组，后续组号按需求大致递减。
- 如果拓扑变化很大，建议重新用新的代表性模型发现规则。

## 离线发现规则

```python
from src.optimization.section_grouping import (
    SectionGroupingDiscoveryConfig,
    build_rules_from_design_configs,
)
from src.structure.enums import BeamType

discovery_config = SectionGroupingDiscoveryConfig(
    n_groups_by_type={
        BeamType.FRAME: 2,
        BeamType.MAIN: 3,
        BeamType.SECONDARY: 3,
        BeamType.BRACE: 2,
    },
    description="engineering baseline grouping",
)

rules = build_rules_from_design_configs(
    representative_configs,
    discovery_config,
    mesh_size=0.5,
)
rules.save_json("results/section_grouping/engineering_rules.json")
```

## 默认不细分模式

不分组优化现在被视为特殊的分组：每类梁 1 个组。默认 ruleset 为：

- `FRAME-SG1`
- `MAIN-SG1`
- `SECONDARY-SG1`
- `BRACE-SG1`

因此优化器内部不再区分“分组/不分组”两套截面逻辑，只是 group 数量不同。

```python
from src.optimization.section_grouping import SectionGroupingRuleset

rules = SectionGroupingRuleset.by_beam_type("engineering")
```

## 在新模型上应用规则

```python
from src.optimization.section_grouping import SectionGroupingRuleset, apply_group_sections

rules = SectionGroupingRuleset.load_json("results/section_grouping/engineering_rules.json")
group_sections = {
    "MAIN-SG1": 8,
    "MAIN-SG2": 5,
    "MAIN-SG3": 3,
}

model.generate()
apply_group_sections(model, section_lib, rules, group_sections)
```

## 在优化中启用

GA：

```python
from src.heuristic.parallel_ga import ParallelGAOptimizer

optimizer = ParallelGAOptimizer(
    layout_strategy="engineering",
    section_group_rules="results/section_grouping/engineering_rules.json",
)
result = optimizer.optimize({"L": 8.9, "W": 3.6, "H": 3.3, "pressure": 30000})
print(result.optimal_group_sections)
```

贝叶斯优化：

```python
from src.heuristic.extended_opt import JointBayesianOptimizer

optimizer = JointBayesianOptimizer(
    layout_strategy="engineering",
    section_group_rules="results/section_grouping/engineering_rules.json",
)
result = optimizer.optimize({"L": 8.9, "W": 3.6, "H": 3.3, "pressure": 30000})
print(result.optimal_group_sections)
```

未传入 `section_group_rules` 时，优化器自动使用 `SectionGroupingRuleset.by_beam_type(...)`，也就是每类梁 1 组。
