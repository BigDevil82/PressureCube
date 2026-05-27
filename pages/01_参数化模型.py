from pathlib import Path
import sys
from contextlib import redirect_stdout
from io import StringIO
import importlib
import re

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src
from src.app.model_visualization import build_structure_figure
from src.app.parametric_helpers import build_engineering_preview, hydrate_defaults
from src.app.session import AppSession
from src.structure.enums import BeamType
from src.structure.layout.params import (
    EngineeringLayoutParameters,
    WindowParameters,
)
from src.structure.mesh_generator import MeshGenerator
from src.structure.section_assignment import SectionAssignment
from src.structure.section_library import SectionLibrary, SectionType
from src.structure.structural_model import StructuralModel

st.set_page_config(layout="wide", page_title="参数化模型")

st.markdown(
    """
    <style>
        .stApp { background-color: #ffffff; }
        .block-container { padding-top: 2rem; padding-bottom: 1rem; }
        h1 { padding-top: 0rem; margin-top: 0rem; }
        [data-testid="stSidebarNav"] li a span{
            font-size: 18px !important;
            font-weight: 600 !important;
            padding: 4px 8px !important;
            margin: 4px 0 !important;
        }
        [data-testid="stSidebarNav"] li a:hover {
            background-color: #f0f2f6 !important;
            border-radius: 8px !important;
        }
        [data-testid="stSidebarNav"] li a[aria-current="page"] {
            background-color: #e8eaf6 !important;
            border-left: 4px solid #1976d2 !important;
            border-radius: 4px !important;
        }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("🏗️ 模块化舱体结构参数化生成")

session = AppSession()

pending_update = st.session_state.pop("pending_optimization_update", None)
if pending_update:
    for key, value in pending_update.items():
        st.session_state[key] = value

defaults = hydrate_defaults()
defaults["layout_mode"] = "engineering"


with st.sidebar:
    layout_mode = "engineering"

    st.header("⚙️ 几何参数控制")
    st.caption("当前使用工程版布局")
    length = st.slider("舱体长度 (Length)", 7.2, 12.0, float(defaults["L"]), 0.1, key="L")
    width = st.number_input("舱体宽度 (Width)", value=3.7, step=0.1, key="W", disabled=True)
    height = st.number_input("舱体高度 (Height)", value=3.3, step=0.1, key="H", disabled=True)
    section_library = SectionLibrary("src/data/steel_sections.json")

    st.divider()
    st.header("📐 拓扑与截面")
    beam_spacing = st.slider(
        "主梁间距",
        0.5,
        1.5 if layout_mode == "engineering" else 3.0,
        min(max(float(defaults["beam_spacing"]), 0.5), 1.5 if layout_mode == "engineering" else 3.0),
        0.05,
        key="beam_spacing",
    )
    add_brace = st.checkbox("添加支撑梁 (Brace)", value=defaults["add_brace"], key="add_brace")

    support_beam_max_spacing = st.slider(
        "支撑梁最大间距 (m)",
        0.3,
        1.5,
        float(defaults["support_beam_max_spacing"]),
        0.05,
        key="support_beam_max_spacing",
    )
    preview_engine = build_engineering_preview(
        length, width, height, beam_spacing, add_brace, support_beam_max_spacing
    )
    main_span_options = list(range(len(preview_engine._main_face_spans())))
    right_span_options = list(range(len(sorted(preview_engine.secondary_x_locs)) + 1))
    st.subheader("🪟 窗户参数")
    front_spans = st.multiselect(
        "前立面窗户 spans",
        options=main_span_options,
        default=[idx for idx in defaults["front_spans"] if idx in main_span_options],
        key="front_spans",
    )
    back_spans = st.multiselect(
        "后立面窗户 spans",
        options=main_span_options,
        default=[idx for idx in defaults["back_spans"] if idx in main_span_options],
        key="back_spans",
    )
    right_spans = st.multiselect(
        "右立面窗户 spans",
        options=right_span_options,
        default=[idx for idx in defaults["right_spans"] if idx in right_span_options],
        key="right_spans",
    )

    st.subheader("🔩 截面选择")
    box_sections = section_library._sections[SectionType.BOX]
    h_main_sections = section_library._sections[SectionType.H_MAIN]
    h_sec_sections = section_library._sections[SectionType.H_SEC]
    channel_sections = section_library._sections[SectionType.CHANNEL]

    frame_idx = st.selectbox(
        "边框梁 (Frame)",
        range(len(box_sections)),
        format_func=lambda i: f"{i}: {box_sections[i].name}",
        index=defaults["frame_idx"],
        key="frame_idx",
    )
    main_idx = st.selectbox(
        "主梁 (Main)",
        range(len(h_main_sections)),
        format_func=lambda i: f"{i}: {h_main_sections[i].name}",
        index=defaults["main_idx"],
        key="main_idx",
    )
    sec_idx = st.selectbox(
        "次梁 (Secondary)",
        range(len(h_sec_sections)),
        format_func=lambda i: f"{i}: {h_sec_sections[i].name}",
        index=defaults["sec_idx"],
        key="sec_idx",
    )
    brace_idx = st.selectbox(
        "支撑梁 (Brace)",
        range(len(channel_sections)),
        format_func=lambda i: f"{i}: {channel_sections[i].name}",
        index=defaults["brace_idx"],
        key="brace_idx",
    )

    st.divider()
    st.header("👁️ 视图设置")
    show_plates = st.checkbox("显示蒙皮 (Plates)", value=True, key="show_plates")
    show_nodes = st.checkbox("显示节点 (Nodes)", value=False, key="show_nodes")
    # st.subheader("🧪 调试")
    # show_meshed_debug = st.checkbox("显示网格划分结果", value=False, key="show_meshed_debug")
    # mesh_size_debug = st.slider("调试网格尺寸 (m)", 0.1, 1.0, 0.5, 0.05, key="mesh_size_debug")

params = EngineeringLayoutParameters(
    length=length,
    width=width,
    height=height,
    main_beam_spacing=beam_spacing,
    add_support_beam=add_brace,
    support_beam_max_spacing=support_beam_max_spacing,
    window=WindowParameters(
        front_spans=sorted(front_spans), back_spans=sorted(back_spans), right_spans=sorted(right_spans)
    ),
)

section_assignment = SectionAssignment.by_beam_type(
    {
        BeamType.FRAME: frame_idx,
        BeamType.MAIN: main_idx,
        BeamType.SECONDARY: sec_idx,
        BeamType.BRACE: brace_idx,
    }
)
model = StructuralModel(params, section_library, section_assignment=section_assignment)
model.generate()

display_model = model
display_label = "原始模型"
meshed_model = None
# if show_meshed_debug:
#     with st.spinner("正在生成调试网格..."):
#         meshed_model = MeshGenerator(model, mesh_size_debug).generate()
#     display_model = meshed_model
#     display_label = "网格模型"

session.set_model_bundle(params, model, section_library)
figure = build_structure_figure(
    display_model,
    show_plates=show_plates,
    show_nodes=show_nodes,
    mesh_debug_view=meshed_model is not None,
    original_model=model if meshed_model is not None else None,
)

col1, col2 = st.columns([3, 1])
with col1:
    st.caption(f"当前显示: {display_label}")
    st.plotly_chart(figure, width="stretch")

with col2:
    st.subheader("📊 结构统计")
    st.metric("布局模式", "工程版" if layout_mode == "engineering" else "研究版")
    st.caption(f"截面库: {section_library.file_path}")
    st.metric("当前视图", display_label)
    st.metric("总质量 (估算)", f"{display_model.calculate_mass_kg():.2f} kg")
    st.metric("节点数量", len(display_model.points))
    st.metric("梁单元数量", len([beam for beam in display_model.beams.values() if not beam.is_virtual]))
    st.metric("钢板数量", len(display_model.plates))
    if meshed_model is not None:
        st.caption(
            f"原始/网格: 节点 {len(model.points)}/{len(meshed_model.points)}，"
            f"梁 {len([b for b in model.beams.values() if not b.is_virtual])}/{len([b for b in meshed_model.beams.values() if not b.is_virtual])}，"
            f"板 {len(model.plates)}/{len(meshed_model.plates)}"
        )


class StreamlitLogBuffer(StringIO):
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder

    def write(self, text):
        result = super().write(text)
        self.placeholder.code(format_optimization_log(self.getvalue()[-12000:]))
        return result


def format_optimization_log(text: str) -> str:
    text = text.replace("Initial mass (max sections)", "基准方案质量")
    text = re.sub(
        r"Gen\s+(\d+)/(\d+):\s+best mass=([0-9.]+)kg,\s+feasible=(\d+)/(\d+),\s+mean_fit=([0-9.]+)",
        r"优化轮次 \1/\2：当前推荐质量=\3 kg，约束满足候选=\4/\5，综合评分=\6",
        text,
    )
    text = text.replace("Model loaded", "智能评估模型已加载")
    text = text.replace("Loaded surrogate model from", "评估模型来源")
    return text


def section_idx(result, beam_type: BeamType) -> int:
    return result.optimal_section_indices_by_group[f"{beam_type.name}-SG1"]


def apply_optimization_result(result):
    st.session_state["pending_optimization_update"] = {
        "beam_spacing": float(result.optimal_topology["main_beam_spacing"]),
        "support_beam_max_spacing": float(result.optimal_topology["support_beam_max_spacing"]),
        "add_brace": bool(result.optimal_topology["add_support_beam"]),
        "frame_idx": section_idx(result, BeamType.FRAME),
        "main_idx": section_idx(result, BeamType.MAIN),
        "sec_idx": section_idx(result, BeamType.SECONDARY),
        "brace_idx": section_idx(result, BeamType.BRACE),
    }


def load_ga_optimizer():
    module_names = [
        "src.optimization.parallel_ga",
        "src.analysis.evaluator",
        "src.analysis.surrogate_model",
        "src.optimization.design_space",
    ]
    for module_name in module_names:
        sys.modules.pop(module_name, None)
    parallel_ga_mod = importlib.import_module("src.optimization.parallel_ga")
    return parallel_ga_mod.ParallelGAOptimizer


st.divider()
st.subheader("结构智能优化")

with st.expander("优化参数", expanded=True):
    opt_col1, opt_col2, opt_col3 = st.columns(3)
    with opt_col1:
        opt_pressure_kpa = st.number_input(
            "内压大小 (kPa)",
            value=float(session.analysis.pressure_pa) / 1000.0,
            step=1.0,
            key="opt_pressure_kpa",
        )
        opt_eval_mode = st.selectbox(
            "评估方式",
            options=["surrogate", "fea"],
            format_func=lambda value: "代理模型" if value == "surrogate" else "有限元",
            key="opt_eval_mode",
        )
        opt_mesh_size = (
            st.slider("有限元网格大小 (m)", 0.1, 1.0, 0.5, 0.05, key="opt_mesh_size")
            if opt_eval_mode == "fea"
            else 0.5
        )
    with opt_col2:
        opt_pop_size = st.number_input(
            "候选方案规模", min_value=8, max_value=256, value=32, step=8, key="opt_pop_size"
        )
        opt_generations = st.number_input(
            "优化轮次", min_value=1, max_value=100, value=10, step=1, key="opt_generations"
        )
    with opt_col3:
        opt_n_jobs = st.number_input(
            "并行进程数", min_value=1, max_value=32, value=4, step=1, key="opt_n_jobs"
        )
        opt_seed = st.number_input(
            "随机种子", min_value=0, max_value=999999, value=42, step=1, key="opt_seed"
        )

run_optimization = st.button("开始智能优化", type="primary", width="stretch")
log_box = st.empty()
summary_box = st.empty()

if run_optimization:
    pressure_pa = float(opt_pressure_kpa) * 1000.0
    geometry = {"L": float(length), "W": float(width), "H": float(height), "pressure": pressure_pa}
    session.analysis.pressure_pa = int(pressure_pa)
    ParallelGAOptimizer = load_ga_optimizer()
    optimizer = ParallelGAOptimizer(
        pop_size=int(opt_pop_size),
        n_generations=int(opt_generations),
        n_jobs=int(opt_n_jobs),
        seed=int(opt_seed),
        eval_mode=opt_eval_mode,
        surrogate_path=("src/data/engineering_model.pkl" if opt_eval_mode == "surrogate" else None),
        layout_strategy="engineering",
        base_params=params,
        mesh_size=float(opt_mesh_size),
    )

    log_buffer = StreamlitLogBuffer(log_box)
    with st.spinner("正在执行智能优化..."):
        with redirect_stdout(log_buffer):
            result = optimizer.optimize(geometry, verbose=True)

    apply_optimization_result(result)
    st.session_state["last_optimization_result"] = {
        "initial_mass": result.initial_mass,
        "optimal_mass": result.optimal_mass,
        "mass_reduction_pct": result.mass_reduction_pct,
        "u_sigma": result.u_sigma,
        "u_delta": result.u_delta,
        "is_feasible": result.is_feasible,
        "topology": result.optimal_topology,
        "sections": result.optimal_section_indices_by_group,
        "n_evaluations": result.n_evaluations,
        "time_seconds": result.time_seconds,
    }
    summary_box.success("优化完成，页面参数已更新为最优方案。")
    st.rerun()

last_result = st.session_state.get("last_optimization_result")
if last_result:
    st.markdown("#### 最近一次优化结果")
    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    r_col1.metric("初始质量", f"{last_result['initial_mass']:.0f} kg")
    r_col2.metric("最优质量", f"{last_result['optimal_mass']:.0f} kg")
    r_col3.metric("材料节省率", f"{last_result['mass_reduction_pct']:.1f}%")
    r_col4.metric("可行性", "满足" if last_result["is_feasible"] else "不满足")

    st.write(
        f"控制指标：u_sigma={last_result['u_sigma']:.3f}，"
        f"u_delta={last_result['u_delta']:.3f}；"
        f"评估次数：{last_result['n_evaluations']}，耗时：{last_result['time_seconds']:.1f}s。"
    )
    st.write(f"拓扑参数：{last_result['topology']}")
    st.write(f"截面分组：{last_result['sections']}")
