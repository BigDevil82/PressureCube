import json
import re
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app.section_grouping_visualization import build_section_grouping_figure
from src.app.session import AppSession
from src.optimization.grouped_workflow import section_result_rows, validate_and_repair
from src.optimization.parallel_ga import ParallelGAOptimizer
from src.structure.enums import BeamType
from src.structure.section_assignment import BEAM_SECTION_TYPES
from src.structure.section_grouping.discovery import (
    SectionGroupingDiscoveryConfig,
    build_rules_from_random_topology_samples,
)


GROUPED_SURROGATE_PATH = "results/surrogate_data/engineering_200k_group/engineering_model_group_reproduce.pkl"

st.set_page_config(layout="wide", page_title="截面分组设计")
st.title("截面分组设计与优化")

session = AppSession()
session.ensure_defaults()
if not session.has_model():
    st.warning("请先在【参数化模型】页面生成模型。")
    st.stop()

params = session.params
model = session.model
grouping = session.grouping
layout_strategy = params.layout_strategy


class StreamlitLogBuffer(StringIO):
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder

    def write(self, text):
        result = super().write(text)
        self.placeholder.code(_format_log(self.getvalue()[-12000:]))
        return result


def _format_log(text: str) -> str:
    text = text.replace("Initial mass (max sections)", "最大截面基准质量")
    return re.sub(
        r"Gen\s+(\d+)/(\d+):\s+best mass=([0-9.]+)kg,\s+feasible=(\d+)/(\d+)",
        r"优化轮次 \1/\2：当前推荐质量=\3 kg，约束满足候选=\4/\5",
        text,
    )


def _current_sections() -> list[int]:
    assignment = model.section_assignment
    sections = []
    for beam_type in sorted(BEAM_SECTION_TYPES, key=lambda item: int(item.value)):
        group = next(group for group in assignment.groups.values() if group.beam_type == beam_type)
        sections.append(group.section_index)
    return sections


def _prototype_rows(ruleset):
    return [
        {
            "分组": item.group_id,
            "类型": item.beam_type.name,
            "样本梁数": item.member_count,
            "应力比均值": round(item.stress_ratio_mean, 3),
            "应力比P90": round(item.stress_ratio_p90, 3),
            "位移P90(mm)": round(item.displacement_p90_mm, 3),
            "控制因素": "应力" if item.control_factor == "stress" else "位移",
        }
        for item in ruleset.prototypes
    ]


def _topology_rows(topology: dict) -> list[dict]:
    labels = {
        "main_beam_spacing": "主梁间距",
        "support_beam_max_spacing": "支撑梁最大间距",
        "num_secondary_width": "宽向次梁分段数",
        "num_secondary_height": "高向次梁分段数",
        "add_support_beam": "设置支撑梁",
    }
    rows = []
    for key, value in topology.items():
        if isinstance(value, bool):
            value = "是" if value else "否"
        elif isinstance(value, float):
            value = f"{value:.2f} m"
        rows.append({"拓扑参数": labels.get(key, key), "最终取值": value})
    return rows


with st.sidebar:
    st.header("分析设置")
    pressure_kpa = st.number_input(
        "设计内压 (kPa)", min_value=0.0, value=float(session.analysis.pressure_pa) / 1000.0, step=1.0
    )
    mesh_size = st.slider("FEA 网格大小 (m)", 0.1, 1.0, 0.5, 0.05)
    session.analysis.pressure_pa = int(pressure_kpa * 1000)

st.subheader("当前参数化模型")
summary_cols = st.columns(4)
summary_cols[0].metric("舱体尺寸", f"{params.length:.1f} × {params.width:.1f} × {params.height:.1f} m")
summary_cols[1].metric("当前质量", f"{model.calculate_mass_kg():.1f} kg")
summary_cols[2].metric("主梁间距", f"{params.main_beam_spacing:.2f} m")
summary_cols[3].metric("支撑梁", "开启" if params.add_support_beam else "关闭")

st.divider()
st.subheader("1. 离线分组规则发现")
st.caption("随机生成若干代表性拓扑，分别完成强度和位移分析，再根据受力响应、空间位置及对称关系发现稳定分组规则。")

with st.expander("规则发现参数", expanded=grouping.ruleset is None):
    group_cols = st.columns(4)
    frame_groups = group_cols[0].slider("框架梁组数", 1, 3, 2)
    main_groups = group_cols[1].slider("主梁组数", 1, 3, 2)
    secondary_groups = group_cols[2].slider("次梁组数", 1, 3, 3)
    brace_groups = group_cols[3].slider("支撑梁组数", 1, 3, 2)

    sample_cols = st.columns(4)
    n_samples = sample_cols[0].number_input("代表模型数量", 2, 30, 8, 1)
    discovery_jobs = sample_cols[1].number_input("并行进程数", 1, 16, 4, 1, key="group_discovery_jobs")
    discovery_seed = sample_cols[2].number_input("随机种子", 0, 999999, 42, 1, key="group_discovery_seed")
    discovery_mesh = sample_cols[3].slider("发现阶段网格 (m)", 0.2, 1.0, 0.5, 0.05)

    weight_cols = st.columns(3)
    stress_weight = weight_cols[0].slider("应力权重", 0.0, 2.0, 1.0, 0.05)
    displacement_weight = weight_cols[1].slider("位移权重", 0.0, 2.0, 0.7, 0.05)
    spatial_weight = weight_cols[2].slider("空间权重", 0.0, 1.0, 0.35, 0.05)

run_discovery = st.button("运行离线分组", type="primary", use_container_width=True)
if run_discovery:
    config = SectionGroupingDiscoveryConfig(
        n_groups_by_type={
            BeamType.FRAME: frame_groups,
            BeamType.MAIN: main_groups,
            BeamType.SECONDARY: secondary_groups,
            BeamType.BRACE: brace_groups,
        },
        stress_weight=stress_weight,
        displacement_weight=displacement_weight,
        spatial_weight=spatial_weight,
        description="Streamlit offline grouping discovery",
    )
    geometry = {"L": params.length, "W": params.width, "H": params.height}
    with st.spinner("正在分析代表性拓扑并发现分组规则..."):
        grouping.ruleset = build_rules_from_random_topology_samples(
            geometry=geometry,
            pressure=pressure_kpa * 1000.0,
            sections=_current_sections(),
            layout_strategy=layout_strategy,
            discovery_config=config,
            n_samples=int(n_samples),
            seed=int(discovery_seed),
            mesh_size=float(discovery_mesh),
            n_jobs=int(discovery_jobs),
            base_params=params,
        )
    grouping.optimization_result = None
    grouping.discovery_signature = session.model_bundle["signature"]
    grouping.discovery_pressure_pa = pressure_kpa * 1000.0
    st.success("离线分组规则发现完成。")

ruleset = grouping.ruleset
if ruleset is None:
    st.info("请先运行离线分组，生成当前结构的稳定分组规则。")
    st.stop()

pressure_matches_rules = abs(grouping.discovery_pressure_pa - pressure_kpa * 1000.0) < 1e-6
if not pressure_matches_rules:
    st.warning("设计内压已改变。请重新运行离线分组后再进行优化。")

view_col, data_col = st.columns([3, 2])
with view_col:
    st.plotly_chart(build_section_grouping_figure(model, ruleset), use_container_width=True)
with data_col:
    st.dataframe(_prototype_rows(ruleset), use_container_width=True, hide_index=True)
    st.download_button(
        "下载分组规则 JSON",
        data=json.dumps(ruleset.to_dict(), ensure_ascii=False, indent=2),
        file_name=f"{layout_strategy}_section_group_rules.json",
        mime="application/json",
        use_container_width=True,
    )

st.divider()
st.subheader("2. 分组截面与拓扑 GA 优化")
with st.expander("GA 参数", expanded=grouping.optimization_result is None):
    ga_cols = st.columns(5)
    eval_mode = ga_cols[0].selectbox(
        "评估方式", ["surrogate", "fea"], format_func=lambda value: "代理模型" if value == "surrogate" else "有限元"
    )
    pop_size = ga_cols[1].number_input("种群规模", 8, 256, 32, 8)
    generations = ga_cols[2].number_input("迭代代数", 1, 100, 10, 1)
    ga_jobs = ga_cols[3].number_input("并行进程数", 1, 16, 4, 1, key="group_ga_jobs")
    ga_seed = ga_cols[4].number_input("随机种子", 0, 999999, 42, 1, key="group_ga_seed")
    repair_steps = st.number_input("代理结果最大微调次数", 1, 100, 24, 1)

run_optimization = st.button(
    "开始分组 GA 优化",
    type="primary",
    use_container_width=True,
    disabled=not pressure_matches_rules,
)
log_box = st.empty()
if run_optimization:
    surrogate_path = GROUPED_SURROGATE_PATH if eval_mode == "surrogate" else None
    if surrogate_path and not Path(surrogate_path).exists():
        st.error(f"未找到分组代理模型：{surrogate_path}")
        st.stop()

    optimizer = ParallelGAOptimizer(
        pop_size=int(pop_size),
        n_generations=int(generations),
        n_jobs=int(ga_jobs),
        seed=int(ga_seed),
        eval_mode=eval_mode,
        surrogate_path=surrogate_path,
        layout_strategy=layout_strategy,
        section_group_rules=ruleset,
        base_params=params,
        mesh_size=float(mesh_size),
    )
    geometry = {
        "L": params.length,
        "W": params.width,
        "H": params.height,
        "pressure": pressure_kpa * 1000.0,
    }
    log_buffer = StreamlitLogBuffer(log_box)
    with st.spinner("正在执行分组 GA 优化..."):
        with redirect_stdout(log_buffer):
            ga_result = optimizer.optimize(geometry, verbose=True)

    with st.spinner("正在进行真实有限元校核及必要的截面微调..."):
        grouping.optimization_result = validate_and_repair(
            ga_result,
            ruleset,
            params,
            pressure_kpa * 1000.0,
            float(mesh_size),
            max_steps=int(repair_steps),
        )
    st.success("分组优化与有限元校核完成。")

result = grouping.optimization_result
if result is None:
    st.stop()

st.divider()
st.subheader("3. 最终设计与 FEA 校核")
metric_cols = st.columns(5)
metric_cols[0].metric("GA 候选质量", f"{result.ga_result.optimal_mass:.1f} kg")
metric_cols[1].metric("最终质量", f"{result.fea_result.mass:.1f} kg")
metric_cols[2].metric("应力利用率", f"{result.fea_result.u_sigma:.3f}")
metric_cols[3].metric("位移利用率", f"{result.fea_result.u_delta:.3f}")
metric_cols[4].metric("FEA 校核", "通过" if result.fea_result.feasible else "未通过")

detail_cols = st.columns(2)
detail_cols[0].caption(
    f"最大应力：{result.fea_result.max_stress_mpa:.2f} MPa；"
    f"位置：{result.fea_result.max_stress_pos}"
)
detail_cols[1].caption(
    f"最大位移：{result.fea_result.max_deflection_mm:.2f} mm；"
    f"位置：节点 {result.fea_result.max_deflection_pos}"
)

st.dataframe(
    [
        {
            "结果阶段": "GA 评估",
            "质量 (kg)": round(result.ga_result.optimal_mass, 2),
            "应力利用率": round(result.ga_result.u_sigma, 3),
            "位移利用率": round(result.ga_result.u_delta, 3),
            "是否满足": "是" if result.ga_result.is_feasible else "否",
        },
        {
            "结果阶段": "候选方案 FEA",
            "质量 (kg)": round(result.initial_fea_result.mass, 2),
            "应力利用率": round(result.initial_fea_result.u_sigma, 3),
            "位移利用率": round(result.initial_fea_result.u_delta, 3),
            "是否满足": "是" if result.initial_fea_result.feasible else "否",
        },
        {
            "结果阶段": "微调后 FEA",
            "质量 (kg)": round(result.fea_result.mass, 2),
            "应力利用率": round(result.fea_result.u_sigma, 3),
            "位移利用率": round(result.fea_result.u_delta, 3),
            "是否满足": "是" if result.fea_result.feasible else "否",
        },
    ],
    use_container_width=True,
    hide_index=True,
)

if result.was_repaired:
    st.warning(f"代理模型候选未直接通过真实 FEA，已执行 {len(result.repair_steps)} 次截面微调。")
if not result.fea_result.feasible:
    st.error("在设定的最大微调次数内仍未满足约束，请增加微调次数或扩大 GA 搜索规模。")

result_cols = st.columns(2)
with result_cols[0]:
    st.markdown("#### 最终拓扑参数")
    st.dataframe(_topology_rows(result.topology), use_container_width=True, hide_index=True)
with result_cols[1]:
    st.markdown("#### 各分组最终截面")
    st.dataframe(section_result_rows(result), use_container_width=True, hide_index=True)

if result.repair_steps:
    with st.expander("查看截面微调记录"):
        st.dataframe(
            [
                {
                    "调整分组": step.group_id,
                    "新截面索引": step.section_index,
                    "应力利用率": round(step.u_sigma, 3),
                    "位移利用率": round(step.u_delta, 3),
                }
                for step in result.repair_steps
            ],
            use_container_width=True,
            hide_index=True,
        )

st.plotly_chart(
    build_section_grouping_figure(
        result.model,
        ruleset,
        beam_to_group=result.design.section_assignment.beam_groups,
    ),
    use_container_width=True,
)
