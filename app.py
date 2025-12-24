import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.structure.enums import BeamType
from src.structure.mesh_generator import MeshGenerator
from src.structure.section_library import SectionLibrary, SectionType
from src.structure.structural_model import StructuralModel, StructuralParameters

st.set_page_config(layout="wide", page_title="结构参数化生成器")

# 30行左右
st.markdown(
    """
    <style>
        /* 1. 设置全局背景为白色 */
        .stApp {
            background-color: #ffffff;
        }

        /* 2. 核心修改：移除顶部大片空白 */
        /* .block-container 是 Streamlit 包裹主要内容的容器 */
        .block-container {
            padding-top: 2rem;  /* 默认是 6rem 左右，改小即可置顶 */
            padding-bottom: 1rem;
        }

        /* (可选) 隐藏右上角的汉堡菜单和顶部彩色条，让画面更干净 */
        /* header {visibility: hidden;} */
        
        /* (可选) 针对标题本身再往上提一点 */
        h1 {
            padding-top: 0rem;
            margin-top: 0rem;
        }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("🏗️ 模块化舱体结构参数化生成")

# --- 1. 左侧边栏：控制参数 ---
with st.sidebar:
    st.header("⚙️ 几何参数控制")

    L = st.slider("舱体长度 (Length)", 5.0, 15.0, 8.0, 0.5)
    W = st.slider("舱体宽度 (Width)", 2.0, 5.0, 3.7, 0.1)
    H = st.slider("舱体高度 (Height)", 2.0, 4.0, 3.3, 0.1)

    st.divider()

    st.header("📐 拓扑与截面")
    beam_spacing = st.slider("主梁间距", 0.5, 3.0, 1.05, 0.05)
    num_sec_w = st.number_input("宽度方向次梁段数", min_value=1, max_value=5, value=3)
    num_sec_h = st.number_input("高度方向次梁段数", min_value=1, max_value=5, value=3)
    add_brace = st.checkbox("添加支撑梁 (Brace)", value=True)

    st.divider()

    st.header("👁️ 视图设置")
    show_plates = st.checkbox("显示蒙皮 (Plates)", value=True)
    show_nodes = st.checkbox("显示节点 (Nodes)", value=False)

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
)

lib = SectionLibrary()

model = StructuralModel(params, lib)
model.generate()
# generator = MeshGenerator(model)
# model = generator.generate()

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


# # 虚拟梁用虚线绘制
# virtual_beams = [b for b in model.beams.values() if b.is_virtual]
# if virtual_beams:
#     v_x = []
#     v_y = []
#     v_z = []
#     for beam in virtual_beams:
#         start = model.points[beam.start_node]
#         end = model.points[beam.end_node]
#         v_x.extend([start.x, end.x, None])
#         v_y.extend([start.y, end.y, None])
#         v_z.extend([start.z, end.z, None])

#     traces.append(
#         go.Scatter3d(
#             x=v_x,
#             y=v_y,
#             z=v_z,
#             mode="lines",
#             line=dict(color="black", width=10, dash="dash"),
#             name="Virtual Beam",
#         )
#     )


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
    # 为了保证索引对应正确，我们先建立一个 node_id 到 列表索引 的映射
    all_node_ids = sorted(model.points.keys())
    node_to_idx = {nid: i for i, nid in enumerate(all_node_ids)}

    # 提取所有点的坐标
    mesh_x = [model.points[nid].x for nid in all_node_ids]
    mesh_y = [model.points[nid].y for nid in all_node_ids]
    mesh_z = [model.points[nid].z for nid in all_node_ids]

    # 2. 构建三角形索引 (i, j, k)
    # Plotly 的 Mesh3d 需要三角形，我们将每个四边形板 (ABCD) 拆分为两个三角形 (ABC 和 ACD)
    i_indices = []
    j_indices = []
    k_indices = []

    for plate in model.plates.values():
        # 获取该板 4 个节点在 mesh_x/y/z 列表中的索引
        # 假设 plate.nodes 是按顺序 [n1, n2, n3, n4] 排列的
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
            opacity=0.5,  # 半透明，以便看到内部梁
            color="#A9A9A9",  # 灰色蒙皮
            alphahull=-1,  # 强制使用提供的索引构建形状
            lighting=dict(ambient=0.5, diffuse=0.8),  # 增加一点光照质感
            name="Plate",
            showlegend=True,
            hoverinfo="skip",  # 鼠标悬停时不显示复杂的坐标信息
        )
    )

if show_nodes:
    # (可选) 绘制节点，如果需要看节点位置
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
        bgcolor="#ffffff",  # Change: 3D场景背景改为白色
    ),
    paper_bgcolor="#ffffff",  # Change: 画布外围背景改为白色
    margin=dict(l=0, r=0, t=0, b=0),
    # Change: 图例字体改为黑色，位置可以微调
    legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.05, font=dict(color="black")),  # Change: 黑色字体
    # Change: 调整这里改变视口高度
    # 之前是 700，如果底部有空余，可以尝试改为 800, 900 甚至 1000
    height=850,
)

fig = go.Figure(data=traces, layout=layout)

# --- 5. (新增) 自动旋转动画配置 ---
# 计算相机旋转的路径 (绕Z轴旋转一圈)
# eye 坐标定义相机的位子，(0,0,0)是模型中心
frames = []
n_frames = 180  # 帧数，越多越平滑，但加载越慢
radius = 2.0  # 相机距离中心的半径，根据模型大小调整，默认约 1.25-2.0

for k in range(n_frames):
    # 计算角度
    theta = (2 * np.pi / n_frames) * k
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = 1.25  # 高度保持不变

    # 创建每一帧的相机位置
    frames.append(go.Frame(layout=dict(scene=dict(camera=dict(eye=dict(x=x, y=y, z=z))))))

# 将帧赋值给图表
fig.frames = frames

# 添加“旋转”和“停止”按钮到布局中
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            y=0.1,  # 按钮位置 (y轴)
            x=0.0,  # 按钮位置 (x轴)
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
                            frame=dict(duration=50, redraw=True),  # duration: 每帧间隔(ms)
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

# --- 5. 展示与统计 ---
col1, col2 = st.columns([3, 1])

with col1:
    st.plotly_chart(fig, use_container_width=True)

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
