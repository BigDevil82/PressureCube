from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.export_geometry import build_default_engineering_model
from src.analysis.opensees_analysis import OpenSeesSolver
from src.analysis.post_processor import DesignCheckResult, ResultProcessor
from src.analysis.results import AnalysisResults, ModalAnalysisResults
from src.analysis.unit import G_M_S2
from src.structure.enums import BeamType
from src.structure.mesh_generator import MeshGenerator
from src.structure.structural_model import Beam
from src.structure.structural_model import StructuralModel
from src.utils.constants import LOAD_DEAD_BOTTOM_KPA, LOAD_LIVE_BOTTOM_KPA


DEFAULT_STRENGTH_COMBO = {"DL": 1.3, "LL": 1.5, "PL": 1.25}
DEFAULT_SERVICE_COMBO = {"DL": 1.0, "LL": 1.0, "PL": 1.0}


def build_analysis_model(
    mesh_size: float | None, base_model: StructuralModel | None = None
) -> tuple[StructuralModel, StructuralModel]:
    if base_model is None:
        base_model = build_default_engineering_model()
    if mesh_size is None:
        return base_model, base_model
    return base_model, MeshGenerator(base_model, mesh_size).generate()


def parse_combo(value: str) -> dict[str, float]:
    factors = {"DL": 0.0, "LL": 0.0, "PL": 0.0}
    for item in value.split(","):
        if not item.strip():
            continue
        key, raw_factor = item.split("=", 1)
        key = key.strip().upper()
        if key not in factors:
            raise ValueError(f"Unsupported load case '{key}'. Use DL, LL, PL.")
        factors[key] = float(raw_factor)
    return factors


def parse_mesh_sweep(value: str) -> list[float | None]:
    mesh_sizes = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item in {"none", "no-mesh", "coarse"}:
            mesh_sizes.append(None)
        else:
            mesh_sizes.append(float(item))
    return mesh_sizes


def mesh_label(mesh_size: float | None) -> str:
    if mesh_size is None:
        return "no_mesh"
    return f"mesh_{str(mesh_size).replace('.', '')}"


def vector_to_list(value: np.ndarray | tuple[float, ...] | list[float] | None) -> list[float] | None:
    if value is None:
        return None
    return [float(v) for v in value]


def model_summary(model: StructuralModel, solver: OpenSeesSolver) -> dict[str, Any]:
    beam_type_summary = {}
    for beam_type in BeamType:
        beams = [beam for beam in model.beams.values() if beam.beam_type == beam_type and not beam.is_virtual]
        total_length = sum(model.points[b.start_node].distance_to(model.points[b.end_node]) for b in beams)
        beam_type_summary[beam_type.name] = {
            "count": len(beams),
            "total_length_m": total_length,
        }

    section_summary: dict[str, dict[str, Any]] = {}
    for beam in model.beams.values():
        if beam.is_virtual:
            continue
        section_summary.setdefault(
            beam.section.name,
            {"count": 0, "beam_types": sorted({beam.beam_type.name})},
        )
        section_summary[beam.section.name]["count"] += 1
        section_summary[beam.section.name]["beam_types"] = sorted(
            set(section_summary[beam.section.name]["beam_types"]) | {beam.beam_type.name}
        )

    node_masses_tonne = solver._calculate_node_mass()
    modal_mass_kg = sum(node_masses_tonne.values()) * 1000.0

    mass_breakdown = mass_breakdown_kg(model)

    return {
        "nodes": len(model.points),
        "beams_total": len(model.beams),
        "beams_physical": sum(1 for beam in model.beams.values() if not beam.is_virtual),
        "beams_virtual": sum(1 for beam in model.beams.values() if beam.is_virtual),
        "plates_total": len(model.plates),
        "plates_opening": sum(1 for plate in model.plates.values() if plate.panel_type == "opening"),
        "plates_bottom": len(solver.bottom_plate_ids),
        "support_nodes_0_based": sorted(solver.support_node_ids),
        "support_nodes_rfem_1_based": sorted(nid + 1 for nid in solver.support_node_ids),
        "structural_mass_kg": model.calculate_mass_kg(),
        "modal_nodal_mass_sum_kg": modal_mass_kg,
        "mass_breakdown_kg": mass_breakdown,
        "self_weight_kn_g_9_80665": model.calculate_mass_kg() * G_M_S2 / 1000.0,
        "self_weight_kn_g_10_0": model.calculate_mass_kg() * 10.0 / 1000.0,
        "beam_types": beam_type_summary,
        "sections": section_summary,
    }


def mass_breakdown_kg(model: StructuralModel) -> dict[str, float]:
    beam_mass = 0.0
    steel_plate_mass = 0.0
    opening_plate_mass = 0.0

    for beam in model.beams.values():
        if beam.is_virtual:
            continue
        length = model.points[beam.start_node].distance_to(model.points[beam.end_node])
        beam_mass += beam.section.density * beam.section.area * length

    for plate in model.plates.values():
        if plate.panel_type == "opening":
            opening_plate_mass += (
                model.params.opening_panel.density_kg_m3
                * model.params.opening_panel.thickness
                * plate.area
            )
        else:
            steel_plate_mass += 7850.0 * model.params.plate_thickness * plate.area

    return {
        "beams": beam_mass,
        "steel_plates": steel_plate_mass,
        "opening_panels": opening_plate_mass,
        "total": beam_mass + steel_plate_mass + opening_plate_mass,
    }


def load_reconciliation(model: StructuralModel, combos: dict[str, dict[str, float]]) -> dict[str, Any]:
    bottom_area_m2 = sum(plate.area for plate in model.plates.values() if plate.on_face == "bottom")
    dead_bottom_kn = LOAD_DEAD_BOTTOM_KPA * bottom_area_m2
    live_bottom_kn = LOAD_LIVE_BOTTOM_KPA * bottom_area_m2
    self_weight_kn_g_code = model.calculate_mass_kg() * G_M_S2 / 1000.0
    self_weight_kn_g_10 = model.calculate_mass_kg() * 10.0 / 1000.0

    combo_rows = {}
    for name, factors in combos.items():
        combo_rows[name] = {
            "vertical_load_kn_g_9_80665": (
                factors.get("DL", 0.0) * (self_weight_kn_g_code + dead_bottom_kn)
                + factors.get("LL", 0.0) * live_bottom_kn
            ),
            "vertical_load_kn_g_10_0": (
                factors.get("DL", 0.0) * (self_weight_kn_g_10 + dead_bottom_kn)
                + factors.get("LL", 0.0) * live_bottom_kn
            ),
        }

    return {
        "bottom_area_m2": bottom_area_m2,
        "dead_bottom_kn": dead_bottom_kn,
        "live_bottom_kn": live_bottom_kn,
        "self_weight_kn_g_9_80665": self_weight_kn_g_code,
        "self_weight_kn_g_10_0": self_weight_kn_g_10,
        "combos": combo_rows,
    }


def max_displacement_metrics(results: AnalysisResults, model: StructuralModel) -> dict[str, Any]:
    max_node = -1
    max_vec = np.zeros(3)
    max_norm = 0.0
    max_components = {"ux": (-1, 0.0), "uy": (-1, 0.0), "uz": (-1, 0.0)}

    for node_id, disp in results.displacements.items():
        vec = np.array(disp[:3], dtype=float)
        norm = float(np.linalg.norm(vec))
        if norm > max_norm:
            max_norm = norm
            max_node = node_id
            max_vec = vec
        for idx, comp in enumerate(("ux", "uy", "uz")):
            if abs(vec[idx]) > abs(max_components[comp][1]):
                max_components[comp] = (node_id, float(vec[idx]))

    return {
        "max_translation_node_0_based": max_node,
        "max_translation_node_rfem_1_based": max_node + 1 if max_node >= 0 else None,
        "max_translation_mm": max_norm,
        "max_translation_vector_mm": vector_to_list(max_vec),
        "max_translation_coord_m": vector_to_list(model.points[max_node].coordinates()) if max_node in model.points else None,
        "max_ux": {
            "node_0_based": max_components["ux"][0],
            "node_rfem_1_based": max_components["ux"][0] + 1 if max_components["ux"][0] >= 0 else None,
            "value_mm": max_components["ux"][1],
        },
        "max_uy": {
            "node_0_based": max_components["uy"][0],
            "node_rfem_1_based": max_components["uy"][0] + 1 if max_components["uy"][0] >= 0 else None,
            "value_mm": max_components["uy"][1],
        },
        "max_uz": {
            "node_0_based": max_components["uz"][0],
            "node_rfem_1_based": max_components["uz"][0] + 1 if max_components["uz"][0] >= 0 else None,
            "value_mm": max_components["uz"][1],
        },
    }


def reaction_metrics(results: AnalysisResults) -> dict[str, Any]:
    if not results.reactions:
        return {"support_reaction_sum_n": [0.0, 0.0, 0.0], "support_reaction_sum_abs_n": [0.0, 0.0, 0.0]}

    vectors = np.array([reaction[:3] for reaction in results.reactions.values()], dtype=float)
    return {
        "support_reaction_sum_n": vector_to_list(np.sum(vectors, axis=0)),
        "support_reaction_sum_abs_n": vector_to_list(np.sum(np.abs(vectors), axis=0)),
    }


def check_to_dict(check: DesignCheckResult) -> dict[str, Any]:
    return {
        "max_stress_ratio": check.max_stress_ratio,
        "max_stress_mpa": check.max_stress,
        "max_stress_pos": list(check.max_stress_pos),
        "critical_beam_id": check.critical_beam_id,
        "critical_beam_rfem_1_based": check.critical_beam_id + 1 if check.critical_beam_id >= 0 else None,
        "critical_plate_id": check.critical_plate_id,
        "failure_mode": check.failure_mode,
        "max_deflection_ratio": check.max_deflection_ratio,
        "max_deflection_mm": check.max_deflection,
        "max_deflection_node_0_based": check.max_deflection_pos,
        "max_deflection_node_rfem_1_based": check.max_deflection_pos + 1 if check.max_deflection_pos >= 0 else None,
        "max_deflection_vector_mm": vector_to_list(check.max_deflection_vector),
        "max_deflection_coord_m": vector_to_list(check.max_deflection_coord),
    }


def member_probe_metrics(
    results: AnalysisResults,
    model: StructuralModel,
    member_rfem_1_based: int | None,
    x_m: float | None,
) -> dict[str, Any] | None:
    if member_rfem_1_based is None or x_m is None:
        return None

    beam_id = member_rfem_1_based - 1
    member_segments = member_segments_for_probe(model, beam_id)
    if not member_segments:
        return {
            "member_rfem_1_based": member_rfem_1_based,
            "x_m": x_m,
            "found": False,
        }

    lengths = [
        model.points[segment.start_node].distance_to(model.points[segment.end_node])
        for segment in member_segments
    ]
    member_length = sum(lengths)
    target_x = min(max(x_m, 0.0), member_length)

    accumulated = 0.0
    segment = member_segments[-1]
    local_ratio = 1.0
    segment_start_x = max(member_length - lengths[-1], 0.0)
    for candidate, length in zip(member_segments, lengths):
        if accumulated + length >= target_x or length == 0:
            segment = candidate
            segment_start_x = accumulated
            local_ratio = 0.0 if length == 0 else (target_x - accumulated) / length
            break
        accumulated += length

    d_start = results.displacements.get(segment.start_node, np.zeros(6))[:3]
    d_end = results.displacements.get(segment.end_node, np.zeros(6))[:3]
    disp = (1.0 - local_ratio) * d_start + local_ratio * d_end

    return {
        "member_rfem_1_based": member_rfem_1_based,
        "member_0_based": beam_id,
        "x_m": x_m,
        "member_length_m": member_length,
        "normalized_x": target_x / member_length if member_length > 0 else 0.0,
        "found": True,
        "segment_member_id": segment.id,
        "segment_parent_id": segment.parent_id,
        "segment_start_x_m": segment_start_x,
        "segment_local_ratio": local_ratio,
        "start_node_rfem_1_based": segment.start_node + 1,
        "end_node_rfem_1_based": segment.end_node + 1,
        "interpolated_displacement_vector_mm": vector_to_list(disp),
        "interpolated_displacement_mm": float(np.linalg.norm(disp)),
    }


def member_segments_for_probe(model: StructuralModel, source_beam_id: int) -> list[Beam]:
    direct = model.beams.get(source_beam_id)
    if direct is not None and direct.parent_id is None:
        return [direct]

    segments = [beam for beam in model.beams.values() if beam.parent_id == source_beam_id]
    if not segments:
        return [direct] if direct is not None else []

    return sorted(segments, key=lambda beam: beam.id)


def run_static_combo(
    model: StructuralModel,
    combo_name: str,
    factors: dict[str, float],
    pressure_pa: float,
    probe_member_rfem: int | None = None,
    probe_x_m: float | None = None,
) -> dict[str, Any]:
    solver = OpenSeesSolver(model)
    results = solver.run_load_combination(factors, pressure_pa=pressure_pa)
    if results is None:
        return {"name": combo_name, "factors": factors, "success": False}

    check = ResultProcessor(model).run_gb50017_check(results)
    return {
        "name": combo_name,
        "factors": factors,
        "success": True,
        "design_check": check_to_dict(check),
        "displacements": max_displacement_metrics(results, model),
        "member_probe": member_probe_metrics(results, model, probe_member_rfem, probe_x_m),
        "reactions": reaction_metrics(results),
        "result_counts": {
            "displacements": len(results.displacements),
            "beam_forces": len(results.beam_forces),
            "plate_stresses": len(results.plate_stresses),
            "reactions": len(results.reactions),
        },
    }


def modal_to_dict(modal: ModalAnalysisResults | None, model: StructuralModel | None = None) -> dict[str, Any]:
    if modal is None:
        return {"success": False}
    rows = []
    for index, frequency in enumerate(modal.frequencies_hz, start=1):
        period = modal.periods_s[index - 1] if index - 1 < len(modal.periods_s) else None
        row = {
            "mode": index,
            "frequency_hz": float(frequency),
            "period_s": float(period) if period is not None else None,
        }
        if model is not None:
            row.update(mode_shape_summary(modal, model, index))
        rows.append(row)
    return {"success": True, "modes": rows}


def mode_shape_summary(modal: ModalAnalysisResults, model: StructuralModel, mode: int) -> dict[str, Any]:
    max_node = -1
    max_norm = -1.0
    max_vec = np.zeros(3)
    component_sums = np.zeros(3)

    for node_id, vector in modal.mode_shapes.get(mode, {}).items():
        translation = np.array(vector[:3], dtype=float)
        component_sums += np.abs(translation)
        norm = float(np.linalg.norm(translation))
        if norm > max_norm:
            max_norm = norm
            max_node = node_id
            max_vec = translation

    dominant_direction = ("X", "Y", "Z")[int(np.argmax(component_sums))]
    return {
        "dominant_translation_direction": dominant_direction,
        "max_shape_node_0_based": max_node,
        "max_shape_node_rfem_1_based": max_node + 1 if max_node >= 0 else None,
        "max_shape_node_coord_m": vector_to_list(model.points[max_node].coordinates()) if max_node in model.points else None,
        "max_shape_vector": vector_to_list(max_vec),
    }


def write_text_report(report: dict[str, Any], path: Path) -> None:
    lines = []
    model = report["model"]
    lines.append("OpenSees RFEM Comparison Report")
    lines.append("")
    lines.append("[Input]")
    for key in ("mesh_size", "base_nodes", "base_beams", "base_plates", "pressure_pa"):
        lines.append(f"{key}: {report['input'][key]}")
    lines.append("")
    lines.append("[Model]")
    for key in (
        "nodes",
        "beams_total",
        "beams_physical",
        "beams_virtual",
        "plates_total",
        "plates_opening",
        "plates_bottom",
        "structural_mass_kg",
        "modal_nodal_mass_sum_kg",
        "mass_breakdown_kg",
        "self_weight_kn_g_9_80665",
        "self_weight_kn_g_10_0",
        "support_nodes_rfem_1_based",
    ):
        lines.append(f"{key}: {model[key]}")

    lines.append("")
    lines.append("[Load Reconciliation]")
    loads = report["load_reconciliation"]
    for key in (
        "bottom_area_m2",
        "dead_bottom_kn",
        "live_bottom_kn",
        "self_weight_kn_g_9_80665",
        "self_weight_kn_g_10_0",
    ):
        lines.append(f"{key}: {loads[key]}")
    for combo_name, values in loads["combos"].items():
        lines.append(f"{combo_name}: {values}")

    lines.append("")
    lines.append("[Static Combos]")
    for combo in report["static_combos"]:
        lines.append(f"{combo['name']}: factors={combo['factors']} success={combo['success']}")
        if not combo["success"]:
            continue
        check = combo["design_check"]
        disp = combo["displacements"]
        reaction = combo["reactions"]
        lines.append(f"  max_stress_ratio: {check['max_stress_ratio']:.6g}")
        lines.append(f"  max_stress_mpa: {check['max_stress_mpa']:.6g}")
        lines.append(f"  max_stress_pos: {check['max_stress_pos']}")
        lines.append(f"  max_deflection_ratio: {check['max_deflection_ratio']:.6g}")
        lines.append(f"  max_deflection_mm: {check['max_deflection_mm']:.6g}")
        lines.append(f"  max_deflection_node_rfem_1_based: {check['max_deflection_node_rfem_1_based']}")
        lines.append(f"  max_translation_mm: {disp['max_translation_mm']:.6g}")
        lines.append(f"  max_translation_node_rfem_1_based: {disp['max_translation_node_rfem_1_based']}")
        if combo.get("member_probe") is not None:
            probe = combo["member_probe"]
            lines.append(f"  member_probe: {probe}")
        lines.append(f"  support_reaction_sum_n: {reaction['support_reaction_sum_n']}")

    lines.append("")
    lines.append("[Modal]")
    if report["modal"]["success"]:
        for row in report["modal"]["modes"]:
            lines.append(
                f"  mode {row['mode']}: f={row['frequency_hz']:.6g} Hz, "
                f"T={row['period_s']:.6g} s, "
                f"dominant={row.get('dominant_translation_direction')}, "
                f"max_node_rfem={row.get('max_shape_node_rfem_1_based')}"
            )
    else:
        lines.append("  modal analysis failed")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_report(
    args: argparse.Namespace, mesh_size: float | None, base_model: StructuralModel | None = None
) -> dict[str, Any]:
    base_model, model = build_analysis_model(mesh_size, base_model)
    model_solver = OpenSeesSolver(model)

    report = {
        "input": {
            "pressure_pa": args.pressure_pa,
            "mesh_size": mesh_size,
            "base_nodes": len(base_model.points),
            "base_beams": len(base_model.beams),
            "base_plates": len(base_model.plates),
            "strength_combo": parse_combo(args.strength_combo),
            "service_combo": parse_combo(args.service_combo),
            "modes": args.modes,
        },
        "model": model_summary(model, model_solver),
        "load_reconciliation": {},
        "static_combos": [],
        "modal": {},
    }

    report["load_reconciliation"] = load_reconciliation(
        model,
        {
            "strength": report["input"]["strength_combo"],
            "service": report["input"]["service_combo"],
        },
    )

    report["static_combos"].append(
        run_static_combo(
            model,
            "strength",
            report["input"]["strength_combo"],
            args.pressure_pa,
            args.probe_member_rfem,
            args.probe_x_m,
        )
    )
    report["static_combos"].append(
        run_static_combo(
            model,
            "service",
            report["input"]["service_combo"],
            args.pressure_pa,
            args.probe_member_rfem,
            args.probe_x_m,
        )
    )

    modal_solver = OpenSeesSolver(model)
    modal_results = modal_solver.run_modal_case(args.modes)
    report["modal"] = modal_to_dict(modal_results, model)
    return report


def write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "opensees_rfem_compare_metrics.json"
    txt_path = output_dir / "opensees_rfem_compare_report.txt"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_text_report(report, txt_path)
    return json_path, txt_path


def print_summary(report: dict[str, Any], json_path: Path, txt_path: Path) -> None:
    print(f"OpenSees comparison metrics exported: {json_path}")
    print(f"OpenSees comparison report exported: {txt_path}")
    input_data = report["input"]
    model = report["model"]
    print(
        f"Model: base nodes/beams/plates={input_data['base_nodes']}/{input_data['base_beams']}/{input_data['base_plates']}, "
        f"analysis nodes/beams/plates={model['nodes']}/{model['beams_total']}/{model['plates_total']}, "
        f"mesh_size={input_data['mesh_size']}"
    )
    print(f"Mass: {model['structural_mass_kg']:.3f} kg")
    for combo in report["static_combos"]:
        if combo["success"]:
            check = combo["design_check"]
            print(
                f"{combo['name']}: stress_ratio={check['max_stress_ratio']:.4f}, "
                f"stress={check['max_stress_mpa']:.3f} MPa, "
                f"deflection={check['max_deflection_mm']:.3f} mm, "
                f"deflection_ratio={check['max_deflection_ratio']:.4f}"
            )
        else:
            print(f"{combo['name']}: analysis failed")
    if report["modal"]["success"]:
        first_mode = report["modal"]["modes"][0]
        print(
            f"Mode 1: frequency={first_mode['frequency_hz']:.4f} Hz, "
            f"period={first_mode['period_s']:.4f} s"
        )


def sweep_row(report: dict[str, Any]) -> dict[str, Any]:
    row = {
        "mesh_size": report["input"]["mesh_size"],
        "nodes": report["model"]["nodes"],
        "beams": report["model"]["beams_total"],
        "plates": report["model"]["plates_total"],
    }
    for combo in report["static_combos"]:
        name = combo["name"]
        if not combo["success"]:
            row[f"{name}_success"] = False
            continue
        check = combo["design_check"]
        disp = combo["displacements"]
        row[f"{name}_success"] = True
        row[f"{name}_max_deflection_mm"] = check["max_deflection_mm"]
        row[f"{name}_max_translation_mm"] = disp["max_translation_mm"]
        row[f"{name}_max_translation_node"] = disp["max_translation_node_rfem_1_based"]
        row[f"{name}_max_stress_mpa"] = check["max_stress_mpa"]
        probe = combo.get("member_probe")
        if probe and probe["found"]:
            row[f"{name}_probe_displacement_mm"] = probe["interpolated_displacement_mm"]
    return row


def write_sweep_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenSees static and modal checks for the RFEM-exported model.")
    parser.add_argument("--pressure-pa", type=float, default=30_000.0)
    parser.add_argument("--mesh-size", type=float, default=0.5, help="OpenSees preprocessing mesh size in meters.")
    parser.add_argument("--no-mesh", action="store_true", help="Analyze the exported coarse model without preprocessing mesh.")
    parser.add_argument(
        "--mesh-sweep",
        help="Comma-separated mesh sizes, e.g. no-mesh,0.5,0.4,0.25. Overrides --mesh-size/--no-mesh.",
    )
    parser.add_argument("--modes", type=int, default=6)
    parser.add_argument("--strength-combo", default="DL=1.3,LL=1.5,PL=1.25")
    parser.add_argument("--service-combo", default="DL=1.0,LL=1.0,PL=1.0")
    parser.add_argument("--probe-member-rfem", type=int)
    parser.add_argument("--probe-x-m", type=float)
    parser.add_argument("--output-dir", default="results/opensees_rfem_compare")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.mesh_sweep:
        rows = []
        for mesh_size in parse_mesh_sweep(args.mesh_sweep):
            report = make_report(args, mesh_size)
            run_dir = output_dir / mesh_label(mesh_size)
            json_path, txt_path = write_report(report, run_dir)
            print_summary(report, json_path, txt_path)
            rows.append(sweep_row(report))

        csv_path = output_dir / "mesh_sweep_summary.csv"
        output_dir.mkdir(parents=True, exist_ok=True)
        write_sweep_csv(rows, csv_path)
        print(f"Mesh sweep summary exported: {csv_path}")
        return

    mesh_size = None if args.no_mesh else args.mesh_size
    report = make_report(args, mesh_size)
    json_path, txt_path = write_report(report, output_dir)
    print_summary(report, json_path, txt_path)


if __name__ == "__main__":
    main()
