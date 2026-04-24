import streamlit as st

from src.app.model_visualization import build_structure_figure
from src.app.parametric_helpers import build_engineering_preview, hydrate_defaults
from src.app.session import AppSession
from src.structure.layout.params import (
    EngineeringLayoutParameters,
    ResearchLayoutParameters,
    WindowParameters,
)
from src.structure.mesh_generator import MeshGenerator
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
defaults = hydrate_defaults()


with st.sidebar:
    st.header("⚙️ 布局策略")
    layout_mode = st.radio(
        "建模方式",
        options=["research", "engineering"],
        index=0 if defaults["layout_mode"] == "research" else 1,
        format_func=lambda value: "研究版布局" if value == "research" else "工程版布局",
        key="layout_mode",
    )

    st.divider()
    st.header("⚙️ 几何参数控制")
    if layout_mode == "engineering":
        length = st.slider("舱体长度 (Length)", 7.2, 12.0, float(defaults["L"]), 0.1, key="L")
        width = st.number_input("舱体宽度 (Width)", value=3.6, step=0.1, key="W", disabled=True)
        height = st.number_input("舱体高度 (Height)", value=3.3, step=0.1, key="H", disabled=True)
        section_library = SectionLibrary("src/data/steel_sections.json")
    else:
        length = st.slider("舱体长度 (Length)", 5.0, 15.0, float(defaults["L"]), 0.2, key="L")
        width = st.slider("舱体宽度 (Width)", 2.0, 5.0, float(defaults["W"]), 0.1, key="W")
        height = st.slider("舱体高度 (Height)", 2.0, 4.0, float(defaults["H"]), 0.1, key="H")
        section_library = SectionLibrary("src/data/steel_sections_large.json")

    st.divider()
    st.header("📐 拓扑与截面")
    beam_spacing = st.slider(
        "主梁间距",
        0.5,
        1.5 if layout_mode == "engineering" else 3.0,
        min(max(float(defaults["beam_spacing"]), 0.5), 1.5 if layout_mode == "engineering" else 3.0),
        0.05,
    )
    add_brace = st.checkbox("添加支撑梁 (Brace)", value=defaults["add_brace"], key="add_brace")

    if layout_mode == "research":
        num_sec_w = st.number_input(
            "宽度方向次梁段数", min_value=1, max_value=8, value=int(defaults["num_sec_w"]), key="num_sec_w"
        )
        num_sec_h = st.number_input(
            "高度方向次梁段数", min_value=1, max_value=8, value=int(defaults["num_sec_h"]), key="num_sec_h"
        )
        support_beam_max_spacing = defaults["support_beam_max_spacing"]
        front_spans = []
        back_spans = []
        right_spans = []
    else:
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
    st.subheader("🧪 调试")
    show_meshed_debug = st.checkbox("显示网格划分结果", value=False, key="show_meshed_debug")
    mesh_size_debug = st.slider("调试网格尺寸 (m)", 0.1, 1.0, 0.5, 0.05, key="mesh_size_debug")

if layout_mode == "research":
    params = ResearchLayoutParameters(
        length=length,
        width=width,
        height=height,
        main_beam_spacing=beam_spacing,
        num_secondary_width=num_sec_w,
        num_secondary_height=num_sec_h,
        add_support_beam=add_brace,
        frame_section_index=frame_idx,
        main_beam_section_index=main_idx,
        secondary_beam_section_index=sec_idx,
        brace_section_index=brace_idx,
    )
else:
    params = EngineeringLayoutParameters(
        length=length,
        width=width,
        height=height,
        main_beam_spacing=beam_spacing,
        add_support_beam=add_brace,
        frame_section_index=frame_idx,
        main_beam_section_index=main_idx,
        secondary_beam_section_index=sec_idx,
        brace_section_index=brace_idx,
        support_beam_max_spacing=support_beam_max_spacing,
        window=WindowParameters(
            front_spans=sorted(front_spans), back_spans=sorted(back_spans), right_spans=sorted(right_spans)
        ),
    )

model = StructuralModel(params, section_library)
model.generate()

display_model = model
display_label = "原始模型"
meshed_model = None
if show_meshed_debug:
    with st.spinner("正在生成调试网格..."):
        meshed_model = MeshGenerator(model, mesh_size_debug).generate()
    display_model = meshed_model
    display_label = "网格模型"

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
    if isinstance(params, EngineeringLayoutParameters):
        st.divider()
        st.caption(
            f"前/后/右窗 spans: {params.window.front_spans} / {params.window.back_spans} / {params.window.right_spans}"
        )
    st.divider()
    st.write("**图例说明:**")
    st.markdown("<span style='color:#FF4B4B'>■</span> Frame (边框)", unsafe_allow_html=True)
    st.markdown("<span style='color:#1E90FF'>■</span> Main (主梁)", unsafe_allow_html=True)
    st.markdown("<span style='color:#FFA500'>■</span> Secondary (次梁)", unsafe_allow_html=True)
    st.markdown("<span style='color:#27C344'>■</span> Brace (支撑)", unsafe_allow_html=True)
    if show_plates:
        st.markdown("<span style='color:#A9A9A9'>■</span> Steel Plate (钢板)", unsafe_allow_html=True)
        st.markdown("<span style='color:#00B7C2'>■</span> Opening Panel (门窗封板)", unsafe_allow_html=True)
