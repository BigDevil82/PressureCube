from __future__ import annotations

import argparse
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
from src.structure.enums import BeamType
from src.structure.structural_model import StructuralModel


DEFAULT_STRENGTH_COMBO = {"DL": 1.3, "LL": 1.5, "PL": 1.25}
DEFAULT_SERVICE_COMBO = {"DL": 1.0, "LL": 1.0, "PL": 1.0}


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
        "beam_types": beam_type_summary,
        "sections": section_summary,
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


def run_static_combo(
    model: StructuralModel,
    combo_name: str,
    factors: dict[str, float],
    pressure_pa: float,
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
        "reactions": reaction_metrics(results),
        "result_counts": {
            "displacements": len(results.displacements),
            "beam_forces": len(results.beam_forces),
            "plate_stresses": len(results.plate_stresses),
            "reactions": len(results.reactions),
        },
    }


def modal_to_dict(modal: ModalAnalysisResults | None) -> dict[str, Any]:
    if modal is None:
        return {"success": False}
    rows = []
    for index, frequency in enumerate(modal.frequencies_hz, start=1):
        period = modal.periods_s[index - 1] if index - 1 < len(modal.periods_s) else None
        rows.append(
            {
                "mode": index,
                "frequency_hz": float(frequency),
                "period_s": float(period) if period is not None else None,
            }
        )
    return {"success": True, "modes": rows}


def write_text_report(report: dict[str, Any], path: Path) -> None:
    lines = []
    model = report["model"]
    lines.append("OpenSees RFEM Comparison Report")
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
        "support_nodes_rfem_1_based",
    ):
        lines.append(f"{key}: {model[key]}")

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
        lines.append(f"  support_reaction_sum_n: {reaction['support_reaction_sum_n']}")

    lines.append("")
    lines.append("[Modal]")
    if report["modal"]["success"]:
        for row in report["modal"]["modes"]:
            lines.append(
                f"  mode {row['mode']}: f={row['frequency_hz']:.6g} Hz, T={row['period_s']:.6g} s"
            )
    else:
        lines.append("  modal analysis failed")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenSees static and modal checks for the RFEM-exported model.")
    parser.add_argument("--pressure-pa", type=float, default=30_000.0)
    parser.add_argument("--modes", type=int, default=6)
    parser.add_argument("--strength-combo", default="DL=1.3,LL=1.5,PL=1.25")
    parser.add_argument("--service-combo", default="DL=1.0,LL=1.0,PL=1.0")
    parser.add_argument("--output-dir", default="results/opensees_rfem_compare")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_default_engineering_model()
    model_solver = OpenSeesSolver(model)

    report = {
        "input": {
            "pressure_pa": args.pressure_pa,
            "strength_combo": parse_combo(args.strength_combo),
            "service_combo": parse_combo(args.service_combo),
            "modes": args.modes,
        },
        "model": model_summary(model, model_solver),
        "static_combos": [],
        "modal": {},
    }

    report["static_combos"].append(
        run_static_combo(model, "strength", report["input"]["strength_combo"], args.pressure_pa)
    )
    report["static_combos"].append(
        run_static_combo(model, "service", report["input"]["service_combo"], args.pressure_pa)
    )

    modal_solver = OpenSeesSolver(model)
    modal_results = modal_solver.run_modal_case(args.modes)
    report["modal"] = modal_to_dict(modal_results)

    json_path = output_dir / "opensees_rfem_compare_metrics.json"
    txt_path = output_dir / "opensees_rfem_compare_report.txt"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_text_report(report, txt_path)

    print(f"OpenSees comparison metrics exported: {json_path}")
    print(f"OpenSees comparison report exported: {txt_path}")
    print(f"Mass: {report['model']['structural_mass_kg']:.3f} kg")
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


if __name__ == "__main__":
    main()
