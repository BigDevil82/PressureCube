import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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

GROUPED_SURROGATE_PATH = "src/data/engineering_group_model.pkl"
MESH_SIZE = 0.5

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
    type_names = {
        BeamType.FRAME: "框架梁",
        BeamType.MAIN: "主梁",
        BeamType.SECONDARY: "次梁",
        BeamType.BRACE: "支撑梁",
    }
    return [
        {
            "分组": item.group_id,
            "类型": type_names[item.beam_type],
            "样本梁数": item.member_count,
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
    session.analysis.pressure_pa = int(pressure_kpa * 1000)

st.subheader("当前参数化模型")
summary_cols = st.columns(4)
summary_cols[0].metric("舱体尺寸", f"{params.length:.1f} × {params.width:.1f} × {params.height:.1f} m")
summary_cols[1].metric("当前质量", f"{model.calculate_mass_kg():.1f} kg")
summary_cols[2].metric("主梁间距", f"{params.main_beam_spacing:.2f} m")
summary_cols[3].metric("支撑梁", "添加" if params.add_support_beam else "不添加")

st.divider()
st.subheader("1. 离线分组规则发现")
st.caption(
    "随机生成若干代表性拓扑，分别完成强度和位移分析，再根据受力响应、空间位置及对称关系发现稳定分组规则。"
)

with st.expander("规则发现参数", expanded=grouping.ruleset is None):
    group_cols = st.columns(4)
    frame_groups = group_cols[0].slider("框架梁组数", 1, 3, 1)
    main_groups = group_cols[1].slider("主梁组数", 1, 3, 2)
    secondary_groups = group_cols[2].slider("次梁组数", 1, 3, 2)
    brace_groups = group_cols[3].slider("支撑梁组数", 1, 3, 2)

    sample_cols = st.columns(3)
    n_samples = sample_cols[0].number_input("代表模型数量", 2, 50, 24, 1)
    discovery_jobs = sample_cols[1].number_input("并行进程数", 1, 16, 8, 1, key="group_discovery_jobs")
    discovery_seed = sample_cols[2].number_input("随机种子", 0, 999999, 42, 1, key="group_discovery_seed")

run_discovery = st.button("运行离线分组", type="primary", use_container_width=True)
if run_discovery:
    config = SectionGroupingDiscoveryConfig(
        n_groups_by_type={
            BeamType.FRAME: frame_groups,
            BeamType.MAIN: main_groups,
            BeamType.SECONDARY: secondary_groups,
            BeamType.BRACE: brace_groups,
        },
        description="Streamlit offline grouping discovery",
    )
    geometry = {"L": params.length, "W": params.width, "H": params.height}
    progress = st.progress(0, text="正在准备代表模型...")
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            build_rules_from_random_topology_samples,
            geometry=geometry,
            pressure=pressure_kpa * 1000.0,
            sections=_current_sections(),
            layout_strategy=layout_strategy,
            discovery_config=config,
            n_samples=int(n_samples),
            seed=int(discovery_seed),
            mesh_size=MESH_SIZE,
            n_jobs=int(discovery_jobs),
            base_params=params,
        )
        progress_value = 5
        while not future.done():
            progress.progress(progress_value, text="正在分析代表模型并发现分组规则...")
            progress_value = min(progress_value + 3, 90)
            time.sleep(0.4)
        grouping.ruleset = future.result()
    progress.progress(100, text="离线分组完成")
    time.sleep(0.3)
    progress.empty()
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

view_col, data_col = st.columns([1.7, 1.7])
with view_col:
    grouping_figure = build_section_grouping_figure(model, ruleset)
    grouping_figure.update_layout(height=420, margin=dict(l=0, r=0, t=45, b=0))
    st.plotly_chart(grouping_figure, use_container_width=True)
with data_col:
    st.dataframe(_prototype_rows(ruleset), use_container_width=True, hide_index=True)

st.divider()
st.subheader("2. 分组截面与拓扑迭代优化")
with st.expander("优化参数", expanded=grouping.optimization_result is None):
    ga_cols = st.columns(5)
    eval_mode = ga_cols[0].selectbox(
        "评估方式",
        ["surrogate", "fea"],
        format_func=lambda value: "代理模型" if value == "surrogate" else "有限元",
    )
    pop_size = ga_cols[1].number_input("候选方案规模", 8, 256, 32, 8)
    generations = ga_cols[2].number_input("优化轮次", 1, 100, 10, 1)
    ga_jobs = ga_cols[3].number_input("并行进程数", 1, 16, 4, 1, key="group_ga_jobs")
    ga_seed = ga_cols[4].number_input("随机种子", 0, 999999, 42, 1, key="group_ga_seed")

run_optimization = st.button(
    "开始迭代优化",
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
        mesh_size=MESH_SIZE,
    )
    geometry = {
        "L": params.length,
        "W": params.width,
        "H": params.height,
        "pressure": pressure_kpa * 1000.0,
    }
    log_buffer = StreamlitLogBuffer(log_box)
    with st.spinner("正在执行分组迭代优化..."):
        with redirect_stdout(log_buffer):
            ga_result = optimizer.optimize(geometry, verbose=True)

    with st.spinner("正在进行最终方案有限元校核..."):
        grouping.optimization_result = validate_and_repair(
            ga_result,
            ruleset,
            params,
            pressure_kpa * 1000.0,
            MESH_SIZE,
        )
    st.success("分组优化与有限元校核完成。")

result = grouping.optimization_result
if result is None:
    st.stop()

st.divider()
st.subheader("3. 最终设计与 FEA 校核")
metric_cols = st.columns(4)
metric_cols[0].metric("最终质量", f"{result.fea_result.mass:.1f} kg")
metric_cols[1].metric("应力比", f"{result.fea_result.u_sigma:.3f}")
metric_cols[2].metric("位移比", f"{result.fea_result.u_delta:.3f}")
metric_cols[3].metric("FEA 校核", "通过" if result.fea_result.feasible else "未通过")

detail_cols = st.columns(2)
detail_cols[0].metric("最大应力", f"{result.fea_result.max_stress_mpa:.2f} MPa")
detail_cols[0].write(f"位置：{result.fea_result.max_stress_pos}")
detail_cols[1].metric("最大位移", f"{result.fea_result.max_deflection_mm:.2f} mm")
detail_cols[1].write(f"位置：节点 {result.fea_result.max_deflection_pos}")

if not result.fea_result.feasible:
    st.error("最终方案未通过 FEA 校核，请扩大候选方案规模或增加优化轮次后重新优化。")

result_cols = st.columns(2)
with result_cols[0]:
    st.markdown("#### 最终拓扑参数")
    st.dataframe(_topology_rows(result.topology), use_container_width=True, hide_index=True)
with result_cols[1]:
    st.markdown("#### 各分组最终截面")
    st.dataframe(section_result_rows(result), use_container_width=True, hide_index=True)
