import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.structure.enums import BeamType
from src.structure.mesh_generator import MeshGenerator
from src.structure.section_library import SectionLibrary, SectionType
from src.structure.structural_model import StructuralModel, StructuralParameters

st.set_page_config(layout="wide", page_title="参数化模型")

# CSS样式
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
        /* 侧边栏导航链接样式优化 */
        [data-testid="stSidebarNav"] li a span{
            font-size: 18px !important;
            font-weight: 600 !important;
            padding: 4px 8px !important;
            margin: 4px 0 !important;
        }
        /* 导航链接悬停效果 */
        [data-testid="stSidebarNav"] li a:hover {
            background-color: #f0f2f6 !important;
            border-radius: 8px !important;
        }
        /* 当前激活页面样式 */
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

# --- 初始化默认参数 ---
# 检查 session_state 中是否有之前保存的参数
default_params = {
    "L": 7.2,
    "W": 3.7,
    "H": 3.3,
    "beam_spacing": 1.05,
    "num_sec_w": 3,
    "num_sec_h": 3,
    "add_brace": True,
    "frame_idx": 2,
    "main_idx": 3,
    "sec_idx": 2,
    "brace_idx": 1,
}

if "current_params" in st.session_state:
    saved_params = st.session_state["current_params"]
    default_params["L"] = saved_params.length
    default_params["W"] = saved_params.width
    default_params["H"] = saved_params.height
    default_params["beam_spacing"] = saved_params.main_beam_spacing
    default_params["num_sec_w"] = saved_params.num_secondary_width
    default_params["num_sec_h"] = saved_params.num_secondary_height
    default_params["add_brace"] = saved_params.add_support_beam
    default_params["frame_idx"] = saved_params.frame_section_index
    default_params["main_idx"] = saved_params.main_beam_section_index
    default_params["sec_idx"] = saved_params.secondary_beam_section_index
    default_params["brace_idx"] = saved_params.brace_section_index

# --- 1. 左侧边栏：控制参数 ---
with st.sidebar:
    st.header("⚙️ 几何参数控制")

    L = st.slider("舱体长度 (Length)", 5.0, 15.0, default_params["L"], 0.2, key="L")
    W = st.slider("舱体宽度 (Width)", 2.0, 5.0, default_params["W"], 0.1, key="W")
    H = st.slider("舱体高度 (Height)", 2.0, 4.0, default_params["H"], 0.1, key="H")

    st.divider()

    st.header("📐 拓扑与截面")
    beam_spacing = st.slider("主梁间距", 0.5, 3.0, default_params["beam_spacing"], 0.05, key="beam_spacing")
    num_sec_w = st.number_input(
        "宽度方向次梁段数", min_value=1, max_value=5, value=default_params["num_sec_w"], key="num_sec_w"
    )
    num_sec_h = st.number_input(
        "高度方向次梁段数", min_value=1, max_value=5, value=default_params["num_sec_h"], key="num_sec_h"
    )
    add_brace = st.checkbox("添加支撑梁 (Brace)", value=default_params["add_brace"], key="add_brace")

    st.subheader("🔩 截面选择")

    # 初始化截面库
    lib = SectionLibrary()

    # 边框梁 (Frame) - 方钢
    box_sections = lib._sections[SectionType.BOX]
    box_options = [f"{i}: {s.name}" for i, s in enumerate(box_sections)]
    frame_idx = st.selectbox(
        "边框梁 (Frame)",
        range(len(box_sections)),
        format_func=lambda i: box_options[i],
        index=default_params["frame_idx"],
        key="frame_idx",
    )

    # 主梁 (Main) - H型钢主梁
    h_main_sections = lib._sections[SectionType.H_MAIN]
    h_main_options = [f"{i}: {s.name}" for i, s in enumerate(h_main_sections)]
    main_idx = st.selectbox(
        "主梁 (Main)",
        range(len(h_main_sections)),
        format_func=lambda i: h_main_options[i],
        index=default_params["main_idx"],
        key="main_idx",
    )

    # 次梁 (Secondary) - H型钢次梁
    h_sec_sections = lib._sections[SectionType.H_SEC]
    h_sec_options = [f"{i}: {s.name}" for i, s in enumerate(h_sec_sections)]
    sec_idx = st.selectbox(
        "次梁 (Secondary)",
        range(len(h_sec_sections)),
        format_func=lambda i: h_sec_options[i],
        index=default_params["sec_idx"],
        key="sec_idx",
    )

    # 支撑梁 (Brace) - C型钢
    channel_sections = lib._sections[SectionType.CHANNEL]
    channel_options = [f"{i}: {s.name}" for i, s in enumerate(channel_sections)]
    brace_idx = st.selectbox(
        "支撑梁 (Brace)",
        range(len(channel_sections)),
        format_func=lambda i: channel_options[i],
        index=default_params["brace_idx"],
        key="brace_idx",
    )

    st.divider()

    st.header("👁️ 视图设置")
    show_plates = st.checkbox("显示蒙皮 (Plates)", value=True, key="show_plates")
    show_nodes = st.checkbox("显示节点 (Nodes)", value=False, key="show_nodes")

    st.info("调整参数后，右侧模型将实时重构。")

# --- 2. 模型生成 ---
# 初始化参数
params = StructuralParameters(
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

# lib 已在侧边栏初始化，直接使用
model = StructuralModel(params, lib)
model.generate()

# 将模型和截面库保存到session state供其他页面使用
st.session_state["current_model"] = model
st.session_state["current_params"] = params
st.session_state["section_library"] = lib
# st.session_state["section_indices"] = {
#     "frame": frame_idx,
#     "main": main_idx,
#     "secondary": sec_idx,
#     "brace": brace_idx,
# }


# --- 3. Plotly 可视化逻辑 ---


def get_beam_color(b_type):
    """根据梁类型返回不同颜色"""
    if b_type == BeamType.FRAME:
        return "#FF4B4B", 12  # 红色, 粗
    elif b_type == BeamType.MAIN:
        return "#1E90FF", 8  # 蓝色
    elif b_type == BeamType.SECONDARY:
        return "#FFA500", 4  # 橙色
    elif b_type == BeamType.BRACE:
        return "#27C344", 2  # 绿色
    return "#888888", 1


# 提取数据用于绘图
traces = []

# 按类型分组梁，以便批量绘制提高性能
beams_by_type = {}

for beam in model.beams.values():
    if beam.is_virtual:
        continue

    if beam.beam_type not in beams_by_type:
        beams_by_type[beam.beam_type] = {"x": [], "y": [], "z": []}

    start = model.points[beam.start_node]
    end = model.points[beam.end_node]

    # 技巧：使用 None 分隔线段，可以在一个 Trace 里画出不连续的线
    d = beams_by_type[beam.beam_type]
    d["x"].extend([start.x, end.x, None])
    d["y"].extend([start.y, end.y, None])
    d["z"].extend([start.z, end.z, None])


# 添加梁
for b_type, data in beams_by_type.items():
    color, width = get_beam_color(b_type)
    traces.append(
        go.Scatter3d(
            x=data["x"],
            y=data["y"],
            z=data["z"],
            mode="lines",
            line=dict(color=color, width=width),
            name=str(b_type).split(".")[-1],
        )
    )

if show_plates and model.plates:
    # 1. 准备 Mesh3d 所需的顶点列表
    all_node_ids = sorted(model.points.keys())
    node_to_idx = {nid: i for i, nid in enumerate(all_node_ids)}

    # 提取所有点的坐标
    mesh_x = [model.points[nid].x for nid in all_node_ids]
    mesh_y = [model.points[nid].y for nid in all_node_ids]
    mesh_z = [model.points[nid].z for nid in all_node_ids]

    # 2. 构建三角形索引 (i, j, k)
    i_indices = []
    j_indices = []
    k_indices = []

    for plate in model.plates.values():
        p_idx = [node_to_idx[n] for n in plate.nodes]

        # 三角形 1: 0 -> 1 -> 2
        i_indices.append(p_idx[0])
        j_indices.append(p_idx[1])
        k_indices.append(p_idx[2])

        # 三角形 2: 0 -> 2 -> 3
        i_indices.append(p_idx[0])
        j_indices.append(p_idx[2])
        k_indices.append(p_idx[3])

    # 3. 创建 Mesh Trace
    traces.append(
        go.Mesh3d(
            x=mesh_x,
            y=mesh_y,
            z=mesh_z,
            i=i_indices,
            j=j_indices,
            k=k_indices,
            opacity=0.5,
            color="#A9A9A9",
            alphahull=-1,
            lighting=dict(ambient=0.5, diffuse=0.8),
            name="Plate",
            showlegend=True,
            hoverinfo="skip",
        )
    )

if show_nodes:
    node_x = [p.x for p in model.points.values()]
    node_y = [p.y for p in model.points.values()]
    node_z = [p.z for p in model.points.values()]
    traces.append(
        go.Scatter3d(
            x=node_x, y=node_y, z=node_z, mode="markers", marker=dict(size=2, color="black"), showlegend=False
        )
    )

# --- 4. 布局设置 ---
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

# --- 5. 自动旋转动画配置 ---
frames = []
n_frames = 180
radius = 2.0

for k in range(n_frames):
    theta = (2 * np.pi / n_frames) * k
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = 1.25

    frames.append(go.Frame(layout=dict(scene=dict(camera=dict(eye=dict(x=x, y=y, z=z))))))

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

# --- 6. 展示与统计 ---
col1, col2 = st.columns([3, 1])

with col1:
    st.plotly_chart(fig, width="stretch")

with col2:
    st.subheader("📊 结构统计")
    st.metric("总质量 (估算)", f"{model.calculate_mass_kg():.2f} kg")
    st.metric("节点数量", len(model.points))
    st.metric("梁单元数量", len([b for b in model.beams.values() if not b.is_virtual]))
    st.metric("钢板数量", len(model.plates))

    st.write("---")
    st.write("**图例说明:**")
    st.markdown(f"<span style='color:#FF4B4B'>■</span> Frame (边框)", unsafe_allow_html=True)
    st.markdown(f"<span style='color:#1E90FF'>■</span> Main (主梁)", unsafe_allow_html=True)
    st.markdown(f"<span style='color:#FFA500'>■</span> Secondary (次梁)", unsafe_allow_html=True)
    st.markdown(f"<span style='color:#27C344'>■</span> Brace (支撑)", unsafe_allow_html=True)
