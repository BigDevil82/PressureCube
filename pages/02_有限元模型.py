import json
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src
from src.analysis.post_processor import DesignCheckResult
from src.analysis.workflows import run_modal_workflow, run_static_workflow
from src.app.model_visualization import build_structure_figure
from src.app.result_visualization import build_modal_result_figure, build_static_result_figure
from src.app.section_grouping_visualization import build_section_grouping_figure
from src.app.session import AppSession
from src.structure.section_grouping.discovery import (
    RepresentativeAnalysis,
    SectionGroupingDiscoveryConfig,
    discover_section_group_rules_with_assignments,
)
from src.structure.enums import BeamType
from src.structure.layout.params import EngineeringLayoutParameters, ResearchLayoutParameters
from src.structure.section_assignment import BEAM_SECTION_TYPES
from src.structure.structural_model import StructuralModel

st.set_page_config(layout="wide", page_title="有限元模型")

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

st.title("🔬 OpenSees 有限元模型分析")

session = AppSession()
session.ensure_defaults()

if not session.has_model():
    st.warning("⚠️ 请先在【参数化模型】页面生成模型！")
    st.info("👈 使用左侧导航栏切换到参数化模型页面")
    st.stop()

model = session.model
analysis = session.analysis

with st.sidebar:
    st.header("⚙️ 分析参数设置")
    mesh_size = st.slider("网格大小 (m)", 0.1, 1.0, float(analysis.mesh_size), 0.05, key="mesh_size")
    pressure_pa = st.slider("内压荷载 (Pa)", 0, 50000, int(analysis.pressure_pa), 1000, key="pressure_pa")
    analysis_type = st.selectbox(
        "分析类型",
        ["静力分析", "模态分析"],
        index=0 if analysis.analysis_type == "静力分析" else 1,
        key="analysis_type_select",
    )
    if analysis.analysis_type != analysis_type:
        analysis.analysis_type = analysis_type
        analysis.analysis_done = False

    if analysis_type == "静力分析":
        load_case = st.selectbox(
            "荷载工况",
            [
                "恒载 (DL)",
                "活载 (LL)",
                "内压 (PL)",
                "应力校核(1.3DL + 1.5LL + 1.25PL)",
                "位移校核(1.0DL + 1.0LL + 1.0PL)",
            ],
            index=[
                "恒载 (DL)",
                "活载 (LL)",
                "内压 (PL)",
                "应力校核(1.3DL + 1.5LL + 1.25PL)",
                "位移校核(1.0DL + 1.0LL + 1.0PL)",
            ].index(
                analysis.load_case
                if analysis.load_case
                in [
                    "恒载 (DL)",
                    "活载 (LL)",
                    "内压 (PL)",
                    "应力校核(1.3DL + 1.5LL + 1.25PL)",
                    "位移校核(1.0DL + 1.0LL + 1.0PL)",
                ]
                else "内压 (PL)"
            ),
            key="load_case",
        )
        num_modes = analysis.num_modes
    else:
        num_modes = st.slider("模态数量", 1, 10, int(analysis.num_modes), 1, key="num_modes")
        load_case = analysis.load_case

    run_analysis = st.button("🚀 运行分析", type="primary", use_container_width=True)

analysis.mesh_size = mesh_size
analysis.pressure_pa = pressure_pa
analysis.load_case = load_case
analysis.num_modes = num_modes

col1, col2 = st.columns([3, 1])
with col1:
    params = session.params
    with st.expander("🧾 当前参数摘要（按列查看）", expanded=True):
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        with summary_col1:
            st.write(f"布局模式: {'工程版' if isinstance(params, EngineeringLayoutParameters) else '研究版'}")
            st.write(f"尺寸: {params.length:.2f} × {params.width:.2f} × {params.height:.2f} m")
            st.write(f"主梁间距: {params.main_beam_spacing:.2f} m")
            st.write(f"支撑梁: {'开启' if params.add_support_beam else '关闭'}")
            if isinstance(params, EngineeringLayoutParameters):
                st.write(f"支撑梁最大间距: {params.support_beam_max_spacing:.2f} m")
        with summary_col2:
            lib = session.section_library
            assignment = session.model.section_assignment
            for beam_type in sorted(BEAM_SECTION_TYPES, key=lambda item: int(item.value)):
                group_ids = [
                    group_id for group_id, group in assignment.groups.items() if group.beam_type == beam_type
                ]
                names = [
                    lib.get_section(
                        BEAM_SECTION_TYPES[beam_type],
                        assignment.groups[group_id].section_index,
                    ).name
                    for group_id in group_ids
                ]
                st.write(f"{beam_type.name}: {', '.join(names)}")
        with summary_col3:
            st.write(f"截面库: {'研究版' if 'large' in getattr(lib, 'file_path', '') else '工程版'}")
            if isinstance(params, ResearchLayoutParameters):
                st.write(f"宽向次梁分段: {params.num_secondary_width}")
                st.write(f"高向次梁分段: {params.num_secondary_height}")
            elif isinstance(params, EngineeringLayoutParameters):
                st.write(f"前窗 spans: {params.window.front_spans}")
                st.write(f"后窗 spans: {params.window.back_spans}")
                st.write(f"右窗 spans: {params.window.right_spans}")
                st.write(f"门洞: {params.door.width:.2f}m × {params.door.height:.2f}m")
        with summary_col4:
            st.write(f"长：{params.length:.2f} m")
            st.write(f"宽：{params.width:.2f} m")
            st.write(f"高：{params.height:.2f} m")

    # preview_model = analysis.meshed_model if analysis.meshed_model is not None else model
    # preview_label = "网格模型" if analysis.meshed_model is not None else "原始模型"
    # with st.expander(f"🧩 {preview_label}预览（高亮门窗封板）", expanded=False):
    #     st.plotly_chart(build_structure_figure(preview_model), width="stretch")

with col2:
    st.subheader("📊 模型统计")
    stat_col1, stat_col2 = st.columns(2)

    with stat_col1:
        st.metric("梁单元", len([beam for beam in model.beams.values() if not beam.is_virtual]))
        st.metric("板单元", len(model.plates))

    with stat_col2:
        st.metric(
            "封板数量", len([plate for plate in model.plates.values() if plate.panel_type == "opening"])
        )
        st.metric("总质量", f"{model.calculate_mass_kg():.1f} kg")

if run_analysis:
    try:
        if analysis.analysis_type == "静力分析":
            with st.spinner("🔄 正在生成网格并运行静力分析..."):
                meshed_model, results, stats, case_name = run_static_workflow(
                    model=model,
                    mesh_size=analysis.mesh_size,
                    load_case=analysis.load_case,
                    pressure_pa=analysis.pressure_pa,
                )
            analysis.meshed_model = meshed_model
            analysis.analysis_results = results
            analysis.analysis_stats = stats
            analysis.modal_results = None
            analysis.case_name = case_name
            analysis.analysis_done = True
            st.success(f"✅ 静力分析完成！工况: {case_name}")
        else:
            with st.spinner("🔄 正在生成网格并运行模态分析..."):
                meshed_model, modal_results = run_modal_workflow(
                    model=model,
                    mesh_size=analysis.mesh_size,
                    num_modes=analysis.num_modes,
                )
            analysis.meshed_model = meshed_model
            analysis.modal_results = modal_results
            analysis.analysis_results = None
            analysis.analysis_stats = None
            analysis.case_name = ""
            analysis.analysis_done = True
            st.success(f"✅ 模态分析完成！共计算 {len(modal_results.frequencies_hz)} 阶模态")
    except Exception as exc:
        analysis.analysis_done = False
        st.error(f"❌ 分析失败: {exc}")

if analysis.analysis_done and analysis.analysis_type == "静力分析" and analysis.analysis_results is not None:
    meshed_model: StructuralModel = analysis.meshed_model
    stats: DesignCheckResult = analysis.analysis_stats
    st.divider()
    st.subheader(f"📈 分析结果: {analysis.case_name}")
    if stats.max_stress_pos[0] == "Beam":
        beam = meshed_model.beams[stats.max_stress_pos[1]]
        stress_pos = (
            meshed_model.points[beam.start_node].coordinates()
            + meshed_model.points[beam.end_node].coordinates()
        ) / 2.0
    else:
        plate = meshed_model.plates[stats.max_stress_pos[1]]
        stress_pos = sum((meshed_model.points[node_id].coordinates() for node_id in plate.nodes)) / 4.0

    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    with metrics_col1:
        st.metric("最大应力", f"{stats.max_stress:.2f} MPa", delta=f"应力比: {stats.max_stress_ratio:.3f}")
        st.caption(f"位置: {stats.max_stress_pos[0]} {stats.max_stress_pos[1]}")
    with metrics_col2:
        st.metric(
            "最大位移", f"{stats.max_deflection:.2f} mm", delta=f"挠度比: {stats.max_deflection_ratio:.3f}"
        )
        st.caption(f"位置: 节点 {stats.max_deflection_pos}")
    with metrics_col3:
        max_disp_node = meshed_model.points[stats.max_deflection_pos]
        st.metric("最大位移位置", f"节点 {stats.max_deflection_pos}")
        st.caption(f"坐标: ({max_disp_node.x:.2f}, {max_disp_node.y:.2f}, {max_disp_node.z:.2f})")
        st.caption(f"最大应力坐标: ({stress_pos[0]:.2f}, {stress_pos[1]:.2f}, {stress_pos[2]:.2f})")

    st.divider()
    st.subheader("🎨 结果可视化")
    option_col1, option_col2, option_col3 = st.columns([2, 1.2, 1])
    with option_col1:
        vis_option = st.radio("选择可视化内容", ["变形图", "板应力云图", "梁内力云图"], horizontal=True)
    with option_col2:
        if vis_option == "变形图":
            component = st.selectbox("位移分量", ["总位移", "Ux", "Uy", "Uz"], index=0)
        elif vis_option == "板应力云图":
            component = st.selectbox("应力分量", ["Von Mises", "σx", "σy", "τxy"], index=0)
        else:
            component = st.selectbox("梁内力分量", ["N", "Vy", "Vz", "My", "Mz", "T"], index=4)
    with option_col3:
        deformation_scale = st.slider("变形放大系数", 0.1, 50.0, 10.0, 0.1)

    component_map = {
        "总位移": "magnitude",
        "Ux": "ux",
        "Uy": "uy",
        "Uz": "uz",
        "Von Mises": "von_mises",
        "σx": "sigma_x",
        "σy": "sigma_y",
        "τxy": "tau_xy",
    }
    view_map = {"变形图": "deformation", "板应力云图": "plate_stress", "梁内力云图": "beam_force"}
    plot_component = component_map.get(component, component)
    figure = build_static_result_figure(
        model=meshed_model,
        results=analysis.analysis_results,
        view=view_map[vis_option],
        component=plot_component,
        deformation_scale=deformation_scale,
    )
    st.plotly_chart(figure, width="stretch")

    # st.divider()
    # with st.expander("截面分组", expanded=True):
    #     st.caption(
    #         "这里使用当前静力分析结果作为一个代表样本发现稳定分组规则。"
    #         "更严谨的优化规则建议离线选取多个代表模型，并分别使用强度组合和位移组合结果。"
    #     )
    #     group_cols = st.columns(4)
    #     with group_cols[0]:
    #         frame_groups = st.slider("框架梁组数", 1, 6, 2)
    #     with group_cols[1]:
    #         main_groups = st.slider("主梁组数", 1, 8, 3)
    #     with group_cols[2]:
    #         secondary_groups = st.slider("次梁组数", 1, 8, 3)
    #     with group_cols[3]:
    #         brace_groups = st.slider("支撑梁组数", 1, 8, 2)

    #     option_cols = st.columns(3)
    #     with option_cols[0]:
    #         stress_weight = st.slider("应力权重", 0.0, 2.0, 1.0, 0.05)
    #     with option_cols[1]:
    #         displacement_weight = st.slider("位移权重", 0.0, 2.0, 0.7, 0.05)
    #     with option_cols[2]:
    #         spatial_weight = st.slider("空间权重", 0.0, 1.0, 0.35, 0.05)

    #     discovery_config = SectionGroupingDiscoveryConfig(
    #         n_groups_by_type={
    #             BeamType.FRAME: frame_groups,
    #             BeamType.MAIN: main_groups,
    #             BeamType.SECONDARY: secondary_groups,
    #             BeamType.BRACE: brace_groups,
    #         },
    #         stress_weight=stress_weight,
    #         displacement_weight=displacement_weight,
    #         spatial_weight=spatial_weight,
    #         description=f"Streamlit discovery from current case: {analysis.case_name}",
    #     )

    #     discovery = discover_section_group_rules_with_assignments(
    #         [
    #             RepresentativeAnalysis(
    #                 model=meshed_model,
    #                 strength_results=analysis.analysis_results,
    #                 service_results=analysis.analysis_results,
    #             )
    #         ],
    #         discovery_config,
    #         layout_strategy=getattr(session.params, "layout_strategy", "research"),
    #     )
    #     ruleset = discovery.ruleset
    #     st.plotly_chart(
    #         build_section_grouping_figure(
    #             meshed_model,
    #             ruleset,
    #             beam_to_group=discovery.sample_beam_to_group[0],
    #         ),
    #         width="stretch",
    #     )
    #     st.dataframe(
    #         [
    #             {
    #                 "组": prototype.group_id,
    #                 "类型": prototype.beam_type.name,
    #                 "发现样本梁数": prototype.member_count,
    #                 "应力比均值": f"{prototype.stress_ratio_mean:.3f}",
    #                 "应力比P90": f"{prototype.stress_ratio_p90:.3f}",
    #                 "应力比最大": f"{prototype.stress_ratio_max:.3f}",
    #                 "位移均值(mm)": f"{prototype.displacement_mean_mm:.3f}",
    #                 "位移P90(mm)": f"{prototype.displacement_p90_mm:.3f}",
    #                 "位移最大(mm)": f"{prototype.displacement_max_mm:.3f}",
    #                 "代表中心": "({:.2f}, {:.2f}, {:.2f})".format(*prototype.representative_center),
    #                 "控制因素": prototype.control_factor,
    #             }
    #             for prototype in ruleset.prototypes
    #         ],
    #         width="stretch",
    #     )
    #     st.download_button(
    #         "下载当前分组规则 JSON",
    #         data=json.dumps(ruleset.to_dict(), ensure_ascii=False, indent=2),
    #         file_name=f"{ruleset.layout_strategy}_section_group_rules.json",
    #         mime="application/json",
    #         use_container_width=True,
    #     )

elif analysis.analysis_done and analysis.analysis_type == "模态分析" and analysis.modal_results is not None:
    modal_results = analysis.modal_results
    st.divider()
    st.subheader("📈 模态分析结果")
    st.dataframe(
        [
            {"模态阶数": index + 1, "频率 (Hz)": f"{freq:.3f}", "周期 (s)": f"{period:.4f}"}
            for index, (freq, period) in enumerate(zip(modal_results.frequencies_hz, modal_results.periods_s))
        ],
        width="stretch",
    )

    control_col1, control_col2 = st.columns([1, 1])
    with control_col1:
        mode_index = st.selectbox("振型阶次", list(range(1, len(modal_results.frequencies_hz) + 1)), index=0)
    with control_col2:
        deformation_scale = st.slider("振型放大系数", 0.1, 100.0, 20.0, 0.1)

    figure = build_modal_result_figure(
        model=analysis.meshed_model,
        modal_results=modal_results,
        mode=mode_index,
        deformation_scale=deformation_scale,
    )
    st.plotly_chart(figure, width="stretch")

elif not run_analysis:
    st.info("👈 请在左侧设置分析参数，然后点击【运行分析】按钮开始计算")
    st.divider()
    st.markdown("""
        1. 先在参数化模型页确认几何与构件布局。
        2. 在本页设置网格大小和分析工况。
        3. 运行后可直接查看位移、板应力和梁内力的 Plotly 结果图。
        """)
