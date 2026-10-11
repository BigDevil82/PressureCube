"""Export configured RFEM cases and report their OpenSees comparison metrics."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_exported_model_opensees import make_report, print_summary, write_report
from scripts.export_geometry import main as export_geometry
from src.analysis.opensees_analysis import OpenSeesSolver
from src.analysis.unit import E_STEEL_MPA, G_M_S2, G_STEEL_MPA, NU_STEEL, RHO_STEEL_KG_M3
from src.structure.enums import BeamType
from src.structure.layout.params import (
    DoorParameters, EngineeringLayoutParameters, ExtensionParameters,
    OpeningPanelParameters, WindowParameters,
)
from src.structure.section_assignment import SectionAssignment
from src.structure.section_library import SectionLibrary
from src.structure.structural_model import StructuralModel
from src.utils.constants import INTERNAL_PRESSURE, LOAD_DEAD_BOTTOM_KPA, LOAD_LIVE_BOTTOM_KPA


GEOMETRY_FILES = (
    "1_nodes.txt", "2_lines.txt", "3_beams.txt", "4_plates.txt", "5_bottom_plates_id.txt",
    "6_dead_loads.txt", "7_live_loads.txt", "8_pressure_loads.txt",
)


def build_case(config: dict) -> StructuralModel:
    values = dict(config["params"])
    for key, cls in (
        ("extension", ExtensionParameters), ("door", DoorParameters),
        ("window", WindowParameters), ("opening_panel", OpeningPanelParameters),
    ):
        if key in values:
            values[key] = cls(**values[key])
    params = EngineeringLayoutParameters(**values)
    assignment = SectionAssignment.by_beam_type(
        {BeamType[key]: index for key, index in config["sections"].items()}
    )
    model = StructuralModel(params, SectionLibrary("src/data/steel_sections.json"), section_assignment=assignment)
    model.generate()
    return model


def export_or_verify(model: StructuralModel, case_dir: Path, pressure_pa: float) -> None:
    if pressure_pa != INTERNAL_PRESSURE:
        raise ValueError("The geometry exporter uses INTERNAL_PRESSURE; configure the same pressure for analysis.")
    if not any((case_dir / name).exists() for name in GEOMETRY_FILES):
        export_geometry(model, str(case_dir))
        return
    with TemporaryDirectory() as tmp:
        with redirect_stdout(StringIO()):
            export_geometry(model, tmp)
        for name in GEOMETRY_FILES:
            actual = case_dir / name
            if not actual.exists() or actual.read_text(encoding="utf-8") != (Path(tmp) / name).read_text(encoding="utf-8"):
                raise ValueError(f"Existing export does not match the configured model: {actual}")
    print(f"Verified all eight geometry/load files: {case_dir}", flush=True)


def coordinates(values: list[float]) -> str:
    return "(" + ", ".join(f"{value:.3f}" for value in values) + ") m"


def section_rows(model: StructuralModel) -> list[str]:
    rows = []
    for kind, label in ((BeamType.FRAME, "框架梁"), (BeamType.MAIN, "主梁"),
                        (BeamType.SECONDARY, "次梁"), (BeamType.BRACE, "支撑梁")):
        section = next(beam.section for beam in model.beams.values() if beam.beam_type == kind)
        rows.append(f"| {label} | {section.name} |")
    return rows


def write_comparison(report: dict, model: StructuralModel, case_dir: Path) -> None:
    p = model.params
    summary = report["model"]
    mass = summary["mass_breakdown_kg"]
    mesh = report["input"]["mesh_size"]
    supports = OpenSeesSolver(model).support_node_ids
    lines = [
        "# RFEM 与 OpenSees 模型分析对比", "", "## 1. 对比模型", "",
        "本算例采用同一参数化模型导出至 RFEM，并使用现有 OpenSees 求解流程进行静力与模态分析。RFEM 结果及差值留空，待建模计算后补充。", "",
        "| 项目 | 数值 |", "|---|---|",
        f"| 长度 | {p.length:g} m |", f"| 宽度 | {p.width:g} m |", f"| 高度 | {p.height:g} m |",
        f"| 主梁间距 | {p.main_beam_spacing:g} m |", f"| 支撑梁最大间距 | {p.support_beam_max_spacing:g} m |",
        "| 支撑梁 | 设置 |", f"| 右侧窗洞 | 第 {', '.join(map(str, p.window.right_spans))} 跨 |",
        f"| 门洞 | {p.door.face} 面，宽 {p.door.width:g} m，高 {p.door.height:g} m |",
        f"| 侧向扩展 | 长 {p.extension.length:g} m，宽 {p.extension.width:g} m |",
        f"| 钢板厚度 | {p.plate_thickness * 1000:g} mm |",
        f"| 门窗封板厚度 | {p.opening_panel.thickness * 1000:g} mm |",
        "| 支座设置 | 底部四角点 X/Y/Z 平动约束，转动自由 |",
        f"| 分析网格尺寸 | {mesh:g} m |", f"| 内压荷载 | {report['input']['pressure_pa'] / 1000:g} kPa |",
        f"| 重力加速度 | {G_M_S2:g} m/s² |", "", "杆件截面：", "",
        "| 构件类型 | 截面 |", "|---|---|", *section_rows(model), "",
        f"钢材采用 E = {E_STEEL_MPA:g} MPa、G = {G_STEEL_MPA:g} MPa、ν = {NU_STEEL:g}、密度 {RHO_STEEL_KG_M3:g} kg/m³。",
        f"门窗封板采用 E = {p.opening_panel.elastic_modulus_mpa:g} MPa、ν = {p.opening_panel.poisson_ratio:g}、密度 {p.opening_panel.density_kg_m3:g} kg/m³。", "",
        "RFEM 支座按以下导出节点设置：", "",
        "| 导出节点编号（从 1 起） | 坐标 |", "|---:|---|",
        *[f"| {nid + 1} | {coordinates(model.points[nid].coordinates())} |" for nid in sorted(supports)], "",
        "| 项目 | 原始模型 | OpenSees 网格模型 |", "|---|---:|---:|",
        f"| 节点数 | {len(model.points)} | {summary['nodes']} |",
        f"| 几何线数（含辅助边界线） | {len((case_dir / '2_lines.txt').read_text().splitlines())} | — |",
        f"| 梁单元数 | {len(model.beams)} | {summary['beams_total']} |",
        f"| 板单元数 | {len(model.plates)} | {summary['plates_total']} |",
        f"| 门窗封板单元数（包含于板单元数） | {sum(p.panel_type == 'opening' for p in model.plates.values())} | {summary['plates_opening']} |", "",
        "模型质量：", "", "| 项目 | RFEM | OpenSees |", "|---|---:|---:|",
        f"| 总质量 | | {summary['structural_mass_kg'] / 1000:.3f} t |",
        f"| 钢杆件质量 | | {mass['beams'] / 1000:.3f} t |",
        f"| 钢板质量 | | {mass['steel_plates'] / 1000:.3f} t |",
        f"| 门窗封板质量 | | {mass['opening_panels'] / 1000:.3f} t |", "",
        "## 2. 荷载组合", "", "| 组合 | 组合系数 |", "|---|---|",
        "| 强度组合 | 1.3DL + 1.5LL + 1.25PL |", "| 使用组合 | DL + LL + PL |", "",
        f"DL 包含结构自重与楼板恒载 {LOAD_DEAD_BOTTOM_KPA:g} kPa，LL 为楼板活载 {LOAD_LIVE_BOTTOM_KPA:g} kPa，PL 为各围护面向外的内压。",
        "RFEM 恒载工况开启自重，Z 向自重系数设为 -1；楼板附加恒载另行施加。", "",
        "下表填写实际支座 Z 向合反力（向上为正），用于核对竖向荷载平衡：", "",
        "| 组合 | RFEM Z 向合反力 | OpenSees Z 向合反力 | 差值 |", "|---|---:|---:|---:|",
    ]
    names = {"strength": "强度组合", "service": "使用组合"}
    for combo in report["static_combos"]:
        reaction = combo["reactions"]["support_reaction_sum_n"][2] / 1000
        lines.append(f"| {names[combo['name']]} | | {reaction:.3f} kN | |")
    lines += ["", "## 3. 静力分析结果", "", "最大矢量位移：", "",
              "| 组合 | RFEM 最大位移 | RFEM 位置 | OpenSees 最大位移 | OpenSees 位置 | 差值 |",
              "|---|---:|---|---:|---|---:|"]
    for combo in report["static_combos"]:
        disp = combo["displacements"]
        pos = f"节点 {disp['max_translation_node_0_based'] + 1}，{coordinates(disp['max_translation_coord_m'])}"
        lines.append(f"| {names[combo['name']]} | | | {disp['max_translation_mm']:.3f} mm | {pos} | |")
    lines += ["", "最大应力：", "", "| 组合 | RFEM 最大应力 | RFEM 控制对象 | OpenSees 最大应力 | OpenSees 控制对象 | 差值 |",
              "|---|---:|---|---:|---|---:|"]
    for combo in report["static_combos"]:
        check = combo["design_check"]
        kind, index = check["max_stress_pos"]
        label = "梁单元" if kind == "Beam" else "板单元"
        lines.append(f"| {names[combo['name']]} | | | {check['max_stress_mpa']:.3f} MPa | {label} {index + 1} | |")
    lines += ["", "OpenSees 网格节点及单元编号按内部编号加 1 显示，不等于 RFEM 网格编号；跨软件定位应优先核对物理坐标。", "",
              "| 组合 | OpenSees 最大应力利用率 | OpenSees 最大位移利用率 |", "|---|---:|---:|"]
    for combo in report["static_combos"]:
        check = combo["design_check"]
        lines.append(f"| {names[combo['name']]} | {check['max_stress_ratio']:.4f} | {check['max_deflection_ratio']:.4f} |")
    lines += ["", f"位移利用率沿用现有后处理，参考限值为宽度/250，即 {p.width * 1000 / 250:.3f} mm。设计验算采用强度组合的应力利用率和使用组合的位移利用率。", "",
              "## 4. 模态分析结果", "",
              "质量源仅包含钢杆件、钢板及门窗封板的结构质量，X/Y/Z 平动采用相同节点质量，无转动质量；楼板附加恒载、活载和压力不转换为模态质量。RFEM 应采用相同质量源。", "",
              "| 阶次 | RFEM 周期 | OpenSees 周期 | 差值 |", "|---:|---:|---:|---:|"]
    for mode in report["modal"]["modes"]:
        lines.append(f"| {mode['mode']} | | {mode['period_s']:.6f} s | |")
    lines += ["", "## 5. 对比结论", "",
              "OpenSees 静力与模态计算已完成。待补充 RFEM 的质量、合反力、最大位移及位置、最大应力及控制对象、前六阶周期后，再评价两软件的一致性。", ""]
    combos = {combo["name"]: combo["design_check"] for combo in report["static_combos"]}
    failures = []
    if combos["strength"]["max_stress_ratio"] > 1:
        failures.append("强度组合应力")
    if combos["service"]["max_deflection_ratio"] > 1:
        failures.append("使用组合位移")
    if failures:
        lines += [f"当前截面配置下，{'、'.join(failures)}利用率超过 1.0。本算例用于求解结果对比，尚不属于通过上述设计验算的方案。", ""]
    lines += ["复现命令：", "", "```powershell",
              f".\\.venv\\Scripts\\python.exe scripts\\compare_geometry_cases.py --case-dir {case_dir.as_posix()}",
              "```", ""]
    (case_dir / "rfem_opensees_comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_case(case_dir: Path, export_only: bool) -> None:
    config = json.loads((case_dir / "case_config.json").read_text(encoding="utf-8"))
    model = build_case(config)
    export_or_verify(model, case_dir, config["pressure_pa"])
    if export_only:
        return
    args = argparse.Namespace(
        pressure_pa=config["pressure_pa"], modes=config.get("modes", 6),
        strength_combo="DL=1.3,LL=1.5,PL=1.25", service_combo="DL=1,LL=1,PL=1",
        probe_member_rfem=None, probe_x_m=None,
    )
    print(f"Analyzing {case_dir} with mesh={config['mesh_size']} m", flush=True)
    report = make_report(args, config["mesh_size"], model)
    if not all(combo["success"] for combo in report["static_combos"]) or not report["modal"]["success"]:
        raise RuntimeError(f"Analysis failed for {case_dir}")
    report["input"]["case_config"] = config
    report["input"]["model_params"] = asdict(model.params)
    report["input"]["gravity_m_s2"] = G_M_S2
    json_path, txt_path = write_report(report, case_dir)
    write_comparison(report, model, case_dir)
    print_summary(report, json_path, txt_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    for case_dir in args.case_dir:
        run_case(case_dir, args.export_only)


if __name__ == "__main__":
    main()
