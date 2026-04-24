from __future__ import annotations

from src.analysis.opensees_analysis import OpenSeesSolver
from src.analysis.post_processor import DesignCheckResult, ResultProcessor
from src.analysis.results import AnalysisResults, ModalAnalysisResults
from src.structure.mesh_generator import MeshGenerator
from src.structure.structural_model import StructuralModel


STATIC_LOAD_CASES = {
    "恒载 (DL)": ("恒载", {"DL": 1.0, "LL": 0.0, "PL": 0.0}),
    "活载 (LL)": ("活载", {"DL": 0.0, "LL": 1.0, "PL": 0.0}),
    "内压 (PL)": ("内压", {"DL": 0.0, "LL": 0.0, "PL": 1.0}),
    "应力校核(1.3DL + 1.5LL + 1.25PL)": ("组合工况 (1.3DL + 1.5LL + 1.25PL)", {"DL": 1.3, "LL": 1.5, "PL": 1.25}),
    "位移校核(1.0DL + 1.0LL + 1.0PL)": ("组合工况 (1.0DL + 1.0LL + 1.0PL)", {"DL": 1.0, "LL": 1.0, "PL": 1.0}),
}


def mesh_model(model: StructuralModel, mesh_size: float) -> StructuralModel:
    return MeshGenerator(model, mesh_size).generate()


def run_static_workflow(
    model: StructuralModel, mesh_size: float, load_case: str, pressure_pa: float
) -> tuple[StructuralModel, AnalysisResults, DesignCheckResult, str]:
    meshed_model = mesh_model(model, mesh_size)
    solver = OpenSeesSolver(meshed_model)
    case_name, combo = STATIC_LOAD_CASES[load_case]
    results = solver.run_load_combination(combo, pressure_pa=pressure_pa)
    stats = ResultProcessor(meshed_model).run_gb50017_check(results)
    if load_case == "内压 (PL)":
        case_name = f"内压 ({pressure_pa / 1000:.0f} kPa)"
    return meshed_model, results, stats, case_name


def run_modal_workflow(
    model: StructuralModel, mesh_size: float, num_modes: int
) -> tuple[StructuralModel, ModalAnalysisResults]:
    meshed_model = mesh_model(model, mesh_size)
    solver = OpenSeesSolver(meshed_model)
    modal_results = solver.run_modal_case(num_modes)
    return meshed_model, modal_results
