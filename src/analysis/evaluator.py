"""
Unified structural evaluator.

The evaluator consumes a complete StructuralDesign. It no longer reconstructs
section assignment from optimization-side rule fragments.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.analysis.opensees_analysis import OpenSeesSolver
from src.analysis.post_processor import ResultProcessor
from src.analysis.surrogate_model import AdvancedSurrogateModel
from src.optimization.design_space import get_section_lib
from src.structure.mesh_generator import MeshGenerator
from src.structure.structural_design import StructuralDesign


@dataclass
class EvalResult:
    """Result of structural evaluation."""

    mass: float
    u_sigma: float
    u_delta: float
    feasible: bool
    max_stress_mpa: float | None = None
    max_stress_pos: tuple[str, int] | None = None
    max_deflection_mm: float | None = None
    max_deflection_pos: int | None = None
    max_deflection_vector: tuple[float, float, float] | None = None
    max_deflection_coord: tuple[float, float, float] | None = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "mass": self.mass,
            "u_sigma": self.u_sigma,
            "u_delta": self.u_delta,
            "feasible": self.feasible,
            "max_stress_mpa": self.max_stress_mpa,
            "max_stress_pos": self.max_stress_pos,
            "max_deflection_mm": self.max_deflection_mm,
            "max_deflection_pos": self.max_deflection_pos,
            "max_deflection_vector": self.max_deflection_vector,
            "max_deflection_coord": self.max_deflection_coord,
        }


class StructuralEvaluator:
    """Evaluate StructuralDesign objects with FEA or a trained surrogate."""

    COMBO_STRENGTH = {"DL": 1.3, "LL": 1.5, "PL": 1.25}
    COMBO_SERVICE = {"DL": 1.0, "LL": 1.0, "PL": 1.0}

    def __init__(
        self,
        mode: str = "fea",
        surrogate_path: Optional[str] = None,
        section_lib_path: Optional[str] = None,
        mesh_size: float = 0.5,
    ):
        self.mode = mode
        self.mesh_size = mesh_size
        self.section_lib_path = section_lib_path

        self.surrogate = None
        if mode == "surrogate":
            self.surrogate = AdvancedSurrogateModel.load(surrogate_path)
            print(f"Loaded surrogate model from: {surrogate_path}")

    def evaluate(self, design: StructuralDesign) -> EvalResult:
        if self.mode == "surrogate":
            return self._evaluate_surrogate(design)
        return self._evaluate_fea(design)

    def _evaluate_fea(self, design: StructuralDesign) -> EvalResult:
        section_lib = get_section_lib(design.layout_strategy, self.section_lib_path)
        model = design.build_model(section_lib)
        mass = model.calculate_mass_kg()

        model = MeshGenerator(model, self.mesh_size).generate()

        solver = OpenSeesSolver(model)
        res_strength = solver.run_load_combination(self.COMBO_STRENGTH, pressure_pa=design.pressure)
        res_service = solver.run_load_combination(self.COMBO_SERVICE, pressure_pa=design.pressure)

        processor = ResultProcessor(model)
        strength_check = processor.run_gb50017_check(res_strength)
        disp_check = processor.run_gb50017_check(res_service)

        u_sigma = strength_check.max_stress_ratio
        u_delta = disp_check.max_deflection_ratio

        return EvalResult(
            mass=mass,
            u_sigma=u_sigma,
            u_delta=u_delta,
            feasible=u_sigma <= 1.0 and u_delta <= 1.0,
            max_stress_mpa=strength_check.max_stress,
            max_stress_pos=strength_check.max_stress_pos,
            max_deflection_mm=disp_check.max_deflection,
            max_deflection_pos=disp_check.max_deflection_pos,
            max_deflection_vector=disp_check.max_deflection_vector,
            max_deflection_coord=disp_check.max_deflection_coord,
        )

    def _evaluate_surrogate(self, design: StructuralDesign) -> EvalResult:
        mass = self.get_mass_only(design)
        u_sigma, u_delta = self.surrogate.predict(design)

        return EvalResult(
            mass=mass,
            u_sigma=u_sigma,
            u_delta=u_delta,
            feasible=u_sigma <= 1.0 and u_delta <= 1.0,
        )

    def evaluate_batch(self, designs: List[StructuralDesign]) -> List[EvalResult]:
        if self.mode != "surrogate":
            return [self._evaluate_fea(design) for design in designs]

        masses = [self.get_mass_only(design) for design in designs]
        sigmas, deltas = self.surrogate.predict_batch(designs)
        return [
            EvalResult(
                mass=mass,
                u_sigma=float(u_sigma),
                u_delta=float(u_delta),
                feasible=bool(u_sigma <= 1.0 and u_delta <= 1.0),
            )
            for mass, u_sigma, u_delta in zip(masses, sigmas, deltas)
        ]

    def get_mass_only(self, design: StructuralDesign) -> float:
        section_lib = get_section_lib(design.layout_strategy, self.section_lib_path)
        return design.build_model(section_lib).calculate_mass_kg()


def evaluate_design(
    design: StructuralDesign,
    mode: str = "fea",
    surrogate_path: Optional[str] = None,
    mesh_size: float = 0.5,
) -> Tuple[float, float, float, bool]:
    evaluator = StructuralEvaluator(mode=mode, surrogate_path=surrogate_path, mesh_size=mesh_size)
    result = evaluator.evaluate(design)
    return result.mass, result.u_sigma, result.u_delta, result.feasible
