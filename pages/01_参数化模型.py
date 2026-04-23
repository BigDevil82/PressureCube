import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.structure.enums import BeamType
from src.structure.layout.engineering_layout import EngineeringLayoutEngine
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
        .stApp {
            background-color: #ffffff;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 1rem;
        }
        h1 {
            padding-top: 0rem;
            margin-top: 0rem;
        }
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


def default_ui_state():
    return {
        "layout_mode": "research",
        "L": 7.2,
        "W": 3.6,
        "H": 3.3,
        "beam_spacing": 1.05,
        "num_sec_w": 3,
        "num_sec_h": 3,
        "add_brace": True,
        "frame_idx": 2,
        "main_idx": 3,
        "sec_idx": 2,
        "brace_idx": 1,
        "support_beam_max_spacing": 0.75,
        "window_active": True,
        "window_sill_height": 0.9,
        "window_height": 1.5,
        "front_spans": [],
        "back_spans": [],
        "right_spans": [],
    }


def hydrate_defaults():
    defaults = default_ui_state()
    saved_params = st.session_state.get("current_params")
    if saved_params is None:
        return defaults

    defaults["layout_mode"] = getattr(saved_params, "layout_strategy", "research")
    defaults["L"] = saved_params.length
    defaults["W"] = saved_params.width
    defaults["H"] = saved_params.height
    defaults["beam_spacing"] = saved_params.main_beam_spacing
    defaults["add_brace"] = saved_params.add_support_beam
    defaults["frame_idx"] = saved_params.frame_section_index
    defaults["main_idx"] = saved_params.main_beam_section_index
    defaults["sec_idx"] = saved_params.secondary_beam_section_index
    defaults["brace_idx"] = saved_params.brace_section_index

    if isinstance(saved_params, ResearchLayoutParameters):
        defaults["num_sec_w"] = saved_params.num_secondary_width
        defaults["num_sec_h"] = saved_params.num_secondary_height

    if isinstance(saved_params, EngineeringLayoutParameters):
        defaults["support_beam_max_spacing"] = saved_params.support_beam_max_spacing
        defaults["window_active"] = saved_params.window.active
        defaults["window_sill_height"] = saved_params.window.sill_height
        defaults["window_height"] = saved_params.window.height
        defaults["front_spans"] = saved_params.window.front_spans
        defaults["back_spans"] = saved_params.window.back_spans
        defaults["right_spans"] = saved_params.window.right_spans

    return defaults


def build_engineering_preview(length, width, height, beam_spacing, add_brace, support_spacing):
    preview_params = EngineeringLayoutParameters(
        length=length,
        width=width,
        height=height,
        main_beam_spacing=beam_spacing,
        add_support_beam=add_brace,
        support_beam_max_spacing=support_spacing,
        window=WindowParameters(active=False),
    )
    return EngineeringLayoutEngine(preview_params)


def get_beam_color(beam_type):
    if beam_type == BeamType.FRAME:
        return "#FF4B4B", 12
    if beam_type == BeamType.MAIN:
        return "#1E90FF", 8
    if beam_type == BeamType.SECONDARY:
        return "#FFA500", 4
    if beam_type == BeamType.BRACE:
        return "#27C344", 2
    return "#888888", 1


def build_plotly_figure(
    target_model: StructuralModel,
    show_plates: bool,
    show_nodes: bool,
    mesh_debug_view: bool = False,
    original_model: StructuralModel | None = None,
) -> go.Figure:
    traces = []
    beams_by_type = {}
    beam_source_model = original_model if mesh_debug_view and original_model is not None else target_model

    for beam in beam_source_model.beams.values():
        if beam.is_virtual:
            continue

        if beam.beam_type not in beams_by_type:
            beams_by_type[beam.beam_type] = {"x": [], "y": [], "z": []}

        start = beam_source_model.points[beam.start_node]
        end = beam_source_model.points[beam.end_node]
        beams_by_type[beam.beam_type]["x"].extend([start.x, end.x, None])
        beams_by_type[beam.beam_type]["y"].extend([start.y, end.y, None])
        beams_by_type[beam.beam_type]["z"].extend([start.z, end.z, None])

    for beam_type, data in beams_by_type.items():
        color, width = get_beam_color(beam_type)
        traces.append(
            go.Scatter3d(
                x=data["x"],
                y=data["y"],
                z=data["z"],
                mode="lines",
                line=dict(color=color, width=width, dash="solid"),
                name=str(beam_type).split(".")[-1],
            )
        )

    if mesh_debug_view and original_model is not None:
        original_edges = set()
        for beam in original_model.beams.values():
            if beam.is_virtual:
                continue
            p1 = original_model.points[beam.start_node]
            p2 = original_model.points[beam.end_node]
            edge = tuple(
                sorted(
                    (
                        (round(p1.x, 6), round(p1.y, 6), round(p1.z, 6)),
                        (round(p2.x, 6), round(p2.y, 6), round(p2.z, 6)),
                    )
                )
            )
            original_edges.add(edge)

        debug_edge_coords = {"x": [], "y": [], "z": []}
        seen_debug_edges = set()

        def add_debug_edge(point_a, point_b):
            edge = tuple(
                sorted(
                    (
                        (round(point_a.x, 6), round(point_a.y, 6), round(point_a.z, 6)),
                        (round(point_b.x, 6), round(point_b.y, 6), round(point_b.z, 6)),
                    )
                )
            )
            if edge in original_edges or edge in seen_debug_edges:
                return
            seen_debug_edges.add(edge)
            debug_edge_coords["x"].extend([point_a.x, point_b.x, None])
            debug_edge_coords["y"].extend([point_a.y, point_b.y, None])
            debug_edge_coords["z"].extend([point_a.z, point_b.z, None])

        for beam in target_model.beams.values():
            if beam.is_virtual:
                continue
            add_debug_edge(target_model.points[beam.start_node], target_model.points[beam.end_node])

        for plate in target_model.plates.values():
            plate_nodes = plate.nodes
            for i in range(4):
                point_a = target_model.points[plate_nodes[i]]
                point_b = target_model.points[plate_nodes[(i + 1) % 4]]
                add_debug_edge(point_a, point_b)

        if debug_edge_coords["x"]:
            traces.append(
                go.Scatter3d(
                    x=debug_edge_coords["x"],
                    y=debug_edge_coords["y"],
                    z=debug_edge_coords["z"],
                    mode="lines",
                    line=dict(color="#666666", width=2),
                    name="Mesh Debug",
                    opacity=0.75,
                )
            )

    if show_plates and target_model.plates:
        all_node_ids = sorted(target_model.points.keys())
        node_to_idx = {nid: i for i, nid in enumerate(all_node_ids)}
        mesh_x = [target_model.points[nid].x for nid in all_node_ids]
        mesh_y = [target_model.points[nid].y for nid in all_node_ids]
        mesh_z = [target_model.points[nid].z for nid in all_node_ids]

        plate_groups = {
            "steel": {"i": [], "j": [], "k": [], "color": "#A9A9A9", "name": "Steel Plate", "opacity": 0.42},
            "opening": {"i": [], "j": [], "k": [], "color": "#00B7C2", "name": "Opening Panel", "opacity": 0.72},
        }
        for plate in target_model.plates.values():
            plate_idx = [node_to_idx[n] for n in plate.nodes]
            group = plate_groups["opening" if plate.panel_type == "opening" else "steel"]
            group["i"].extend([plate_idx[0], plate_idx[0]])
            group["j"].extend([plate_idx[1], plate_idx[2]])
            group["k"].extend([plate_idx[2], plate_idx[3]])

        for group in plate_groups.values():
            if not group["i"]:
                continue
            traces.append(
                go.Mesh3d(
                    x=mesh_x,
                    y=mesh_y,
                    z=mesh_z,
                    i=group["i"],
                    j=group["j"],
                    k=group["k"],
                    opacity=group["opacity"],
                    color=group["color"],
                    alphahull=-1,
                    lighting=dict(ambient=0.5, diffuse=0.8),
                    name=group["name"],
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

    if show_nodes:
        traces.append(
            go.Scatter3d(
                x=[point.x for point in target_model.points.values()],
                y=[point.y for point in target_model.points.values()],
                z=[point.z for point in target_model.points.values()],
                mode="markers",
                marker=dict(
                    size=1 if mesh_debug_view else 2, color="black", opacity=0.45 if mesh_debug_view else 0.8
                ),
                showlegend=False,
            )
        )

    layout = go.Layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            bgcolor="#ffffff",
        ),
        paper_bgcolor="#ffffff",
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.05, font=dict(color="black")),
        height=850,
    )

    fig = go.Figure(data=traces, layout=layout)
    frames = []
    for k in range(180):
        theta = (2 * np.pi / 180) * k
        frames.append(
            go.Frame(
                layout=dict(
                    scene=dict(camera=dict(eye=dict(x=2 * np.cos(theta), y=2 * np.sin(theta), z=1.25)))
                )
            )
        )
    fig.frames = frames
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=0.1,
                x=0.0,
                xanchor="left",
                yanchor="bottom",
                pad=dict(t=0, r=10),
                buttons=[
                    dict(
                        label="▶ 开始旋转",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=50, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="⏸ 停止",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            )
        ]
    )
    return fig


defaults = hydrate_defaults()
lib = SectionLibrary()

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
        L = st.slider("舱体长度 (Length)", 7.2, 12.0, float(defaults["L"]), 0.1, key="L")
        W = st.number_input("舱体宽度 (Width)", value=3.6, step=0.1, key="W", disabled=True)
        H = st.number_input("舱体高度 (Height)", value=3.3, step=0.1, key="H", disabled=True)
    else:
        L = st.slider("舱体长度 (Length)", 5.0, 15.0, float(defaults["L"]), 0.2, key="L")
        W = st.slider("舱体宽度 (Width)", 2.0, 5.0, float(defaults["W"]), 0.1, key="W")
        H = st.slider("舱体高度 (Height)", 2.0, 4.0, float(defaults["H"]), 0.1, key="H")

    st.divider()
    st.header("📐 拓扑与截面")
    max_spacing = 1.5 if layout_mode == "engineering" else 3.0
    beam_spacing_default = min(max(float(defaults["beam_spacing"]), 0.5), max_spacing)
    beam_spacing = st.slider("主梁间距", 0.5, max_spacing, beam_spacing_default, 0.05)
    add_brace = st.checkbox("添加支撑梁 (Brace)", value=defaults["add_brace"], key="add_brace")

    if layout_mode == "research":
        num_sec_w = st.number_input(
            "宽度方向次梁段数", min_value=1, max_value=8, value=int(defaults["num_sec_w"]), key="num_sec_w"
        )
        num_sec_h = st.number_input(
            "高度方向次梁段数", min_value=1, max_value=8, value=int(defaults["num_sec_h"]), key="num_sec_h"
        )
    else:
        support_beam_max_spacing = st.slider(
            "支撑梁最大间距 (m)",
            0.3,
            1.5,
            float(defaults["support_beam_max_spacing"]),
            0.05,
            key="support_beam_max_spacing",
        )

        preview_engine = build_engineering_preview(L, W, H, beam_spacing, add_brace, support_beam_max_spacing)
        main_span_options = list(range(len(preview_engine._main_face_spans())))
        right_span_options = list(range(len(sorted(preview_engine.secondary_x_locs)) + 1))

        st.subheader("🪟 窗户参数")
        window_active = st.checkbox("启用窗洞", value=defaults["window_active"], key="window_active")
        window_sill_height = st.slider(
            "窗台高度 (m)",
            0.2,
            max(0.2, H - 0.6),
            min(float(defaults["window_sill_height"]), max(0.2, H - 0.6)),
            0.05,
            key="window_sill_height",
        )
        window_height = st.slider(
            "窗高 (m)",
            0.5,
            max(0.5, H - window_sill_height),
            min(float(defaults["window_height"]), max(0.5, H - window_sill_height)),
            0.05,
            key="window_height",
        )
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
    box_sections = lib._sections[SectionType.BOX]
    h_main_sections = lib._sections[SectionType.H_MAIN]
    h_sec_sections = lib._sections[SectionType.H_SEC]
    channel_sections = lib._sections[SectionType.CHANNEL]

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
        length=L,
        width=W,
        height=H,
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
        length=L,
        width=W,
        height=H,
        main_beam_spacing=beam_spacing,
        add_support_beam=add_brace,
        frame_section_index=frame_idx,
        main_beam_section_index=main_idx,
        secondary_beam_section_index=sec_idx,
        brace_section_index=brace_idx,
        support_beam_max_spacing=support_beam_max_spacing,
        window=WindowParameters(
            active=window_active,
            sill_height=window_sill_height,
            height=window_height,
            front_spans=sorted(front_spans),
            back_spans=sorted(back_spans),
            right_spans=sorted(right_spans),
        ),
    )
    params.door.active = True
    params.door.face = "left"
    params.door.width = 0.91
    params.door.height = params.window.sill_height + params.window.height
    params.door.bottom_offset = 0.0
    params.door.center_x = None

model = StructuralModel(params, lib)
model.generate()

display_model = model
display_mode_label = "原始模型"
meshed_model = None

if show_meshed_debug:
    with st.spinner("正在生成调试网格..."):
        meshed_model = MeshGenerator(model, mesh_size_debug).generate()
    display_model = meshed_model
    display_mode_label = "网格模型"

st.session_state["current_model"] = model
st.session_state["current_params"] = params
st.session_state["section_library"] = lib
fig = build_plotly_figure(
    display_model,
    show_plates,
    show_nodes,
    mesh_debug_view=meshed_model is not None,
    original_model=model if meshed_model is not None else None,
)

col1, col2 = st.columns([3, 1])

with col1:
    st.caption(f"当前显示: {display_mode_label}")
    st.plotly_chart(fig, width="stretch")

with col2:
    st.subheader("📊 结构统计")
    st.metric("布局模式", "工程版" if layout_mode == "engineering" else "研究版")
    st.metric("当前视图", display_mode_label)
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
        st.write("---")
        st.caption(
            f"前/后/右窗 spans: {params.window.front_spans} / {params.window.back_spans} / {params.window.right_spans}"
        )

    st.write("---")
    st.write("**图例说明:**")
    st.markdown("<span style='color:#FF4B4B'>■</span> Frame (边框)", unsafe_allow_html=True)
    st.markdown("<span style='color:#1E90FF'>■</span> Main (主梁)", unsafe_allow_html=True)
    st.markdown("<span style='color:#FFA500'>■</span> Secondary (次梁)", unsafe_allow_html=True)
    st.markdown("<span style='color:#27C344'>■</span> Brace (支撑)", unsafe_allow_html=True)
    if show_plates:
        st.markdown("<span style='color:#A9A9A9'>■</span> Steel Plate (钢板)", unsafe_allow_html=True)
        st.markdown("<span style='color:#00B7C2'>■</span> Opening Panel (门窗封板)", unsafe_allow_html=True)
