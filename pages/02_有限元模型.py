import gc
import os
import shutil
import time

import opstool.vis.plotly as opsvis
import streamlit as st

from src.analysis.opensees_analysis import OpenSeesSolver
from src.analysis.post_processor import DesignCheckResult, ResultProcessor
from src.structure.mesh_generator import MeshGenerator
from src.structure.structural_model import BeamType, StructuralModel, StructuralParameters

st.set_page_config(layout="wide", page_title="有限元模型")

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

st.title("🔬 OpenSees 有限元模型分析")

# 检查是否有模型数据
if "current_model" not in st.session_state or "current_params" not in st.session_state:
    st.warning("⚠️ 请先在【参数化模型】页面生成模型！")
    st.info("👈 使用左侧导航栏切换到参数化模型页面")
    st.stop()

# 初始化 Session State 用于持久化分析结果
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False
    st.session_state["meshed_model"] = None
    st.session_state["analysis_results"] = None
    st.session_state["analysis_stats"] = None
    st.session_state["case_name"] = ""
    st.session_state["solver"] = None
    st.session_state["analysis_type"] = "静力分析"
    st.session_state["modal_results"] = None
    st.session_state["odb_counter"] = 1  # 动态 ODB 计数器

# 从session state获取模型
params: StructuralParameters = st.session_state["current_params"]
model: StructuralModel = st.session_state["current_model"]

# 侧边栏：分析参数设置
with st.sidebar:
    st.header("⚙️ 分析参数设置")

    st.subheader("🌐 网格细化")
    mesh_size = st.slider("网格大小 (m)", 0.1, 1.0, 0.5, 0.05, key="mesh_size")

    st.divider()

    st.subheader("📊 荷载工况")
    pressure_pa = st.slider("内压荷载 (Pa)", 0, 50000, 30000, 1000, key="pressure_pa")

    st.divider()

    st.subheader("🔍 分析选项")
    analysis_type = st.selectbox(
        "分析类型",
        ["静力分析", "模态分析"],
        index=0 if st.session_state["analysis_type"] == "静力分析" else 1,
        key="analysis_type_select",
    )

    # 检测分析类型是否改变，如果改变则重置分析状态
    if st.session_state["analysis_type"] != analysis_type:
        st.session_state["analysis_done"] = False
        st.session_state["analysis_type"] = analysis_type

    if analysis_type == "静力分析":
        load_case = st.selectbox(
            "荷载工况", ["恒载 (DL)", "活载 (LL)", "内压 (PL)", "组合工况"], index=2, key="load_case"
        )
    else:
        num_modes = st.slider("模态数量", 1, 10, 6, 1, key="num_modes")

    st.divider()

    if st.button("🚀 运行分析", type="primary", use_container_width=True):
        # 标记需要重新运行分析
        st.session_state["analysis_done"] = False
        run_analysis_trigger = True
    else:
        run_analysis_trigger = False

# 主内容区域
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📐 模型信息")

    # 显示模型基本信息
    info_cols = st.columns(4)
    with info_cols[0]:
        st.metric("长度", f"{params.length:.1f} m")
    with info_cols[1]:
        st.metric("宽度", f"{params.width:.1f} m")
    with info_cols[2]:
        st.metric("高度", f"{params.height:.1f} m")
    with info_cols[3]:
        st.metric("节点数", len(model.points))

    st.divider()

    # 截面信息展示
    st.subheader("📏 梁截面信息")

    section_info = {}
    for beam in model.beams.values():
        if not beam.is_virtual and beam.beam_type not in section_info:
            section_info[beam.beam_type] = beam.section

    section_cols = st.columns(4)
    beam_type_names = {
        BeamType.FRAME: "边框梁",
        BeamType.MAIN: "主梁",
        BeamType.SECONDARY: "次梁",
        BeamType.BRACE: "支撑梁",
    }
    disp_order = [BeamType.FRAME, BeamType.MAIN, BeamType.SECONDARY, BeamType.BRACE]

    display_idx = 0
    for beam_type in disp_order:
        section = section_info.get(beam_type)
        if section is None:
            continue
        col_idx = display_idx % 4
        with section_cols[col_idx]:
            st.markdown(
                f"""
                <div style="font-size:18px; font-weight:600; margin-bottom:4px;">
                    {beam_type_names.get(beam_type, str(beam_type))}
                </div>
                <div style="font-size:16px; color:#333;">
                    类型: {section.name}
                </div>
                """,
                unsafe_allow_html=True,
            )
        display_idx += 1

    st.divider()

with col2:
    st.subheader("📊 模型统计")
    st.metric("梁单元", len([b for b in model.beams.values() if not b.is_virtual]))
    st.metric("板单元", len(model.plates))
    st.metric("总质量", f"{model.calculate_mass_kg():.1f} kg")


# 运行分析逻辑
if run_analysis_trigger:
    with st.spinner("🔄 正在生成网格..."):
        # 网格细化
        mesh_generator = MeshGenerator(model, mesh_size)
        meshed_model = mesh_generator.generate()
        st.session_state["meshed_model"] = meshed_model

    st.success(f"✅ 网格生成完成！节点数: {len(meshed_model.points)}")

    # 清理旧的求解器对象，释放文件句柄
    if "solver" in st.session_state and st.session_state["solver"] is not None:
        try:
            del st.session_state["solver"]
            gc.collect()  # 强制垃圾回收
            time.sleep(0.1)  # 给系统一点时间释放文件句柄
        except:
            pass

    # 清理旧的 .opstool.output 文件夹（带重试机制）
    if os.path.exists(".opstool.output"):
        for attempt in range(3):
            try:
                shutil.rmtree(".opstool.output")
                print("✅ 清理旧的 .opstool.output 文件夹")
                break
            except PermissionError:
                if attempt < 2:
                    time.sleep(0.5)  # 等待后重试
                    gc.collect()
                else:
                    st.warning("⚠️ 无法完全清理旧文件，将使用新的 ODB 标签")
            except Exception as e:
                print(f"清理文件夹时出错: {e}")
                break

    # 创建 OpenSees 求解器
    solver = OpenSeesSolver(meshed_model, enable_odb=True)
    st.session_state["solver"] = solver

    # 生成唯一的 ODB 基础标签
    base_odb_tag = st.session_state["odb_counter"] * 10
    st.session_state["odb_counter"] += 1

    if analysis_type == "静力分析":
        with st.spinner("🔄 正在运行静力分析..."):
            if load_case == "恒载 (DL)":
                odb_tag = base_odb_tag + 1
                combo_dl = {"DL": 1, "LL": 0, "PL": 0}
                results = solver.run_load_combination(combo_dl, odb_tag=odb_tag)
                case_name = "恒载"
                st.session_state["active_odb_tag"] = odb_tag
            elif load_case == "活载 (LL)":
                odb_tag = base_odb_tag + 2
                combo_ll = {"DL": 0, "LL": 1, "PL": 0}
                results = solver.run_load_combination(combo_ll, odb_tag=odb_tag)
                case_name = "活载"
                st.session_state["active_odb_tag"] = odb_tag
            elif load_case == "内压 (PL)":
                odb_tag = base_odb_tag + 3
                combo_pl = {"DL": 0, "LL": 0, "PL": 1}
                results = solver.run_load_combination(combo_pl, pressure_pa=pressure_pa, odb_tag=odb_tag)
                case_name = f"内压 ({pressure_pa/1000:.0f} kPa)"
                st.session_state["active_odb_tag"] = odb_tag
            elif load_case == "应力校核(1.3DL + 1.5LL + 1.25PL)":  # 组合工况
                odb_tag = base_odb_tag + 4
                combo_strength = {"DL": 1.3, "LL": 1.5, "PL": 1.25}  # 承载能力极限状态
                results = solver.run_load_combination(
                    combo_strength, pressure_pa=pressure_pa, odb_tag=odb_tag
                )
                case_name = "组合工况 (1.3DL + 1.5LL + 1.25PL)"
                st.session_state["active_odb_tag"] = base_odb_tag + 4
            elif load_case == "位移校核(1.0DL + 1.0LL + 1.0PL)":  # 服务极限状态
                odb_tag = base_odb_tag + 5
                combo_service = {"DL": 1.0, "LL": 1.0, "PL": 1.0}  # 服务极限状态
                results = solver.run_load_combination(combo_service, pressure_pa=pressure_pa, odb_tag=odb_tag)
                case_name = "组合工况 (1.0DL + 1.0LL + 1.0PL)"
                st.session_state["active_odb_tag"] = base_odb_tag + 5

        if results is not None:
            st.session_state["analysis_results"] = results
            st.session_state["case_name"] = case_name

            # 后处理
            with st.spinner("🔄 正在处理结果..."):
                processor = ResultProcessor(meshed_model)
                stats = processor.run_gb50017_check(results)
                st.session_state["analysis_stats"] = stats

            st.session_state["analysis_done"] = True
            st.success(f"✅ 静力分析完成！工况: {case_name}")
        else:
            st.error("❌ 分析失败！请检查模型和参数设置。")

    else:  # 模态分析
        with st.spinner("🔄 正在运行模态分析..."):
            odb_tag = base_odb_tag + 6
            modal_results = solver.run_modal_case(num_modes, odb_tag=odb_tag)
            if modal_results:
                st.session_state["modal_results"] = modal_results
                st.session_state["active_odb_tag"] = odb_tag
                st.session_state["analysis_done"] = True
                st.success(f"✅ 模态分析完成！共计算 {len(modal_results[0])} 阶模态")
            else:
                st.error("❌ 模态分析失败！")

# 显示分析结果 (如果已完成)
if st.session_state["analysis_done"]:
    meshed_model: StructuralModel = st.session_state["meshed_model"]
    solver: OpenSeesSolver = st.session_state["solver"]

    if st.session_state["analysis_type"] == "静力分析":
        results = st.session_state["analysis_results"]
        stats: DesignCheckResult = st.session_state["analysis_stats"]
        case_name = st.session_state["case_name"]

        with st.expander("🎨 OpenSees 模型几何可视化", expanded=False):
            try:
                # 重新构建模型几何用于显示 (OpenSees 状态可能已丢失)
                solver._setup_model_geometry()
                opsvis.set_plot_props(point_size=0, window_size=(800, 800))
                fig_model = opsvis.plot_model()
                st.plotly_chart(fig_model)
            except Exception as e:
                st.error(f"模型可视化失败: {str(e)}")

        # 显示结果指标
        st.divider()
        st.subheader(f"📈 分析结果: {case_name}")

        result_cols = st.columns(3)
        with result_cols[0]:
            st.metric(
                "最大应力", f"{stats.max_stress:.2f} MPa", delta=f"应力比: {stats.max_stress_ratio:.3f}"
            )
            if stats.max_stress_ratio > 1.0:
                st.error("❌ 应力超限！")
            else:
                st.success("✅ 应力满足要求")

        with result_cols[1]:
            st.metric(
                "最大位移",
                f"{stats.max_deflection:.2f} mm",
                delta=f"挠度比: {stats.max_deflection_ratio:.3f}",
            )
            if stats.max_deflection_ratio > 1.0:
                st.error("❌ 位移超限！")
            else:
                st.success("✅ 位移满足要求")

        with result_cols[2]:
            max_disp_node = meshed_model.points[stats.max_deflection_pos]
            st.metric("最大位移位置", f"节点 {stats.max_deflection_pos}")
            st.caption(f"坐标: ({max_disp_node.x:.2f}, {max_disp_node.y:.2f}, {max_disp_node.z:.2f})")

        # 结果可视化选项
        st.divider()
        st.subheader("🎨 结果可视化 (opstool)")

        # 如果是组合工况，提示可视化限制
        if st.session_state.get("load_case") == "组合工况":
            st.warning(
                "⚠️ 组合工况的可视化目前显示的是内压 (PL) 的云图。数值指标 (如最大应力) 是准确的组合值。"
            )

        active_tag = st.session_state.get("active_odb_tag", 1)

        v_col1, v_col2 = st.columns([2, 1])
        with v_col1:
            vis_option = st.radio(
                "选择可视化内容",
                ["变形图", "壳应力云图", "梁内力云图"],
                horizontal=True,
                key="vis_option_radio",
            )

        stress_comp = "σ11"
        beam_force = "Mz"
        if vis_option == "壳应力云图":
            with v_col2:
                stress_comp = st.selectbox(
                    "选择应力分量",
                    ["σ11", "σ22", "σ12", "σ23", "σ13"],
                    index=0,
                    key="stress_comp_select",
                )
        elif vis_option == "梁内力云图":
            with v_col2:
                beam_force = st.selectbox(
                    "选择内力分量",
                    ["N", "Mz", "Vy", "My", "Vz", "T"],
                    index=1,
                    key="beam_force_select",
                )

        try:
            if vis_option == "变形图":
                st.info("💡 显示结构变形...")
                fig = opsvis.plot_nodal_responses(
                    odb_tag=active_tag,
                    slides=False,
                    defo_scale="auto",
                    resp_type="disp",
                    resp_dof=["UX", "UY", "UZ"],
                    unit_symbol="mm",
                )
                st.plotly_chart(fig)

            elif vis_option == "壳应力云图":
                st.info(f"💡 显示 {stress_comp} 应力云图...")
                opsvis.set_plot_props(cmap="plasma", point_size=0, window_size=(800, 800))
                stress_comp_map = {
                    "σ11": "sigma11",
                    "σ22": "sigma22",
                    "σ12": "sigma12",
                    "σ23": "sigma23",
                    "σ13": "sigma13",
                }
                fig = opsvis.plot_unstruct_responses(
                    odb_tag=active_tag,
                    slides=False,
                    step="absMax",
                    ele_type="Shell",
                    resp_type="stressesAtNodes",
                    resp_dof=stress_comp_map[stress_comp],
                    unit_symbol="MPa",
                )
                st.plotly_chart(fig)

            elif vis_option == "梁内力云图":
                st.info(f"💡 显示梁 {beam_force} 内力云图...")
                opsvis.set_plot_props(cmap="jet", point_size=0, window_size=(800, 800))

                fig = opsvis.plot_frame_responses(
                    odb_tag=active_tag,
                    slides=False,
                    step="absMax",
                    resp_type="sectionForces",
                    resp_dof=beam_force,
                    unit_symbol="N(/N·mm)",
                    scale=3.0,
                    show_values=True,  # hover to show values
                    line_width=5,
                    show_bc=True,
                    bc_scale=2,
                    style="surface",
                    opacity=1.0,
                )
                st.plotly_chart(fig)

        except Exception as e:
            st.error(f"❌ 可视化失败: {str(e)}")
            st.info("💡 提示: 如果切换页面后可视化失效，请尝试重新运行分析。")

    else:  # 模态分析
        frequencies, periods = st.session_state["modal_results"]
        active_tag = st.session_state.get("active_odb_tag", 5)

        st.divider()
        st.subheader("📈 模态分析结果")

        modal_data = []
        for i, (freq, period) in enumerate(zip(frequencies, periods), start=1):
            modal_data.append({"模态阶数": i, "频率 (Hz)": f"{freq:.3f}", "周期 (s)": f"{period:.4f}"})

        st.dataframe(modal_data, width="stretch")

        # 模态形状可视化
        st.subheader("🎨 振型可视化")
        opsvis.set_plot_props(window_size=(600, 800))
        try:

            fig_eigen = opsvis.plot_eigen(odb_tag=active_tag, mode_tags=st.session_state["num_modes"])
            st.plotly_chart(fig_eigen)
        except Exception as e:
            st.error(f"振型可视化失败: {str(e)}")


elif not run_analysis_trigger:
    # 未运行分析时显示提示
    st.info("👈 请在左侧设置分析参数，然后点击【运行分析】按钮开始计算")
    st.divider()
    st.subheader("💡 使用说明")
    st.markdown(
        """
    ### 分析流程
    1. **网格细化**: 调整网格大小以控制计算精度和速度
    2. **荷载设置**: 选择分析类型和荷载工况
    3. **运行分析**: 点击运行按钮执行有限元分析
    4. **查看结果**: 分析完成后查看应力、位移等结果
    """
    )
