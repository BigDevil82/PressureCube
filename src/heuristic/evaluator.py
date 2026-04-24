"""
Unified Structural Evaluator.

Provides a single interface for evaluating structural designs, supporting:
1. FEA mode: Real finite element analysis using OpenSeesSolver
2. Surrogate mode: Fast ML-based prediction using trained surrogate model

Usage:
    # FEA mode (default)
    evaluator = StructuralEvaluator(mode="fea")
    result = evaluator.evaluate(L=8.0, W=3.6, H=3.6, pressure=30000.0,
                                 main_beam_spacing=1.0, num_secondary_width=4,
                                 num_secondary_height=4, add_support_beam=True,
                                 sections=[10, 15, 5, 12])

    # Surrogate mode
    evaluator = StructuralEvaluator(mode="surrogate",
                                     surrogate_path="results/surrogate_data/model_advanced.pkl")
    result = evaluator.evaluate(...)

Returns:
    {"mass": float, "u_sigma": float, "u_delta": float, "feasible": bool}
"""

import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.analysis.opensees_analysis import OpenSeesSolver
from src.analysis.post_processor import ResultProcessor
from src.analysis.surrogate_data import DesignConfig, compute_mass_fast
from src.analysis.surrogate_model import AdvancedSurrogateModel
from src.optimization.design_space import (
    EngineeringDesignConfig,
    ResearchDesignConfig,
    build_params_from_config,
    design_config_from_params,
    get_section_lib,
)
from src.structure.layout.params import BaseLayoutParameters, StructuralParameters
from src.structure.mesh_generator import MeshGenerator
from src.structure.structural_model import StructuralModel


@dataclass
class EvalResult:
    """Result of structural evaluation."""

    mass: float
    u_sigma: float
    u_delta: float
    feasible: bool

    def to_dict(self) -> Dict[str, float]:
        return {
            "mass": self.mass,
            "u_sigma": self.u_sigma,
            "u_delta": self.u_delta,
            "feasible": self.feasible,
        }


class StructuralEvaluator:
    """
    Unified structural evaluation interface.

    Supports two modes:
    - "fea": Real finite element analysis (accurate but slow)
    - "surrogate": ML-based prediction (fast but approximate)
    """

    # Standard load combinations
    COMBO_STRENGTH = {"DL": 1.3, "LL": 1.5, "PL": 1.25}
    COMBO_SERVICE = {"DL": 1.0, "LL": 1.0, "PL": 1.0}

    def __init__(
        self,
        mode: str = "fea",
        surrogate_path: Optional[str] = None,
        section_lib_path: Optional[str] = None,
        mesh_size: float = 0.5,
    ):
        """
        Initialize evaluator.

        Args:
            mode: "fea" for finite element, "surrogate" for ML prediction
            surrogate_path: Path to trained surrogate model (required for surrogate mode)
            section_lib_path: Path to section library JSON
            mesh_size: Mesh size for FEA (only used in FEA mode)
        """
        self.mode = mode
        self.mesh_size = mesh_size
        self.section_lib_path = section_lib_path

        # Load surrogate model if needed
        self.surrogate = None
        if mode == "surrogate":
            if surrogate_path is None:
                raise ValueError("surrogate_path required for surrogate mode")
            self._load_surrogate(surrogate_path)

    def _load_surrogate(self, path: str):
        """Load surrogate model from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        # Detect model type
        if "strategy" in data:
            self.surrogate = AdvancedSurrogateModel.load(path)
        else:
            raise ValueError(f"Unknown surrogate model format in {path}")

        print(f"Loaded surrogate model from: {path}")

    def evaluate(self, *args, **kwargs) -> EvalResult:
        """
        Evaluate a structural design.
        """
        if args and isinstance(args[0], DesignConfig):
            config = args[0]
        elif "config" in kwargs and isinstance(kwargs["config"], DesignConfig):
            config = kwargs["config"]
        else:
            config = ResearchDesignConfig(
                L=kwargs["L"],
                W=kwargs["W"],
                H=kwargs["H"],
                pressure=kwargs["pressure"],
                main_beam_spacing=kwargs["main_beam_spacing"],
                num_secondary_width=kwargs["num_secondary_width"],
                num_secondary_height=kwargs["num_secondary_height"],
                add_support_beam=kwargs["add_support_beam"],
                frame_section_index=kwargs["sections"][0],
                main_beam_section_index=kwargs["sections"][1],
                secondary_beam_section_index=kwargs["sections"][2],
                brace_section_index=kwargs["sections"][3],
            )
        if self.mode == "surrogate":
            return self._evaluate_surrogate(config)
        return self._evaluate_fea(config)

    def evaluate_legacy(
        self,
        L: float,
        W: float,
        H: float,
        pressure: float,
        main_beam_spacing: float,
        num_secondary_width: int,
        num_secondary_height: int,
        add_support_beam: bool,
        sections: List[int],
    ) -> EvalResult:
        config = ResearchDesignConfig(
            L=L,
            W=W,
            H=H,
            pressure=pressure,
            main_beam_spacing=main_beam_spacing,
            num_secondary_width=num_secondary_width,
            num_secondary_height=num_secondary_height,
            add_support_beam=add_support_beam,
            frame_section_index=sections[0],
            main_beam_section_index=sections[1],
            secondary_beam_section_index=sections[2],
            brace_section_index=sections[3],
        )
        return self.evaluate(config)

    def _evaluate_fea(self, config: DesignConfig) -> EvalResult:
        """Real FEA evaluation using OpenSeesSolver."""
        params = build_params_from_config(config)
        section_lib = get_section_lib(config.layout_strategy, self.section_lib_path)

        # Build model
        model = StructuralModel(params, section_lib)
        model.generate()
        mass = model.calculate_mass_kg()

        # Mesh
        mesh_gen = MeshGenerator(model, self.mesh_size)
        model = mesh_gen.generate()

        # Run FEA
        solver = OpenSeesSolver(model)
        res_strength = solver.run_load_combination(self.COMBO_STRENGTH, pressure_pa=config.pressure)
        res_service = solver.run_load_combination(self.COMBO_SERVICE, pressure_pa=config.pressure)

        # Check results
        processor = ResultProcessor(model)
        strength_check = processor.run_gb50017_check(res_strength)
        disp_check = processor.run_gb50017_check(res_service)

        u_sigma = strength_check.max_stress_ratio
        u_delta = disp_check.max_deflection_ratio

        feasible = u_sigma <= 1.0 and u_delta <= 1.0

        return EvalResult(mass=mass, u_sigma=u_sigma, u_delta=u_delta, feasible=feasible)

    def _evaluate_surrogate(self, config: DesignConfig) -> EvalResult:
        """Fast surrogate model evaluation."""
        # Compute mass (fast, no FEA)
        mass = compute_mass_fast(config, get_section_lib(config.layout_strategy, self.section_lib_path))

        # Predict u_sigma and u_delta
        u_sigma, u_delta = self.surrogate.predict(config)

        feasible = u_sigma <= 1.0 and u_delta <= 1.0

        return EvalResult(mass=mass, u_sigma=u_sigma, u_delta=u_delta, feasible=feasible)

    def evaluate_params(self, params: BaseLayoutParameters, pressure: float) -> EvalResult:
        """
        Evaluate using layout parameters directly.

        Convenience method for integration with existing code.
        """
        return self.evaluate(design_config_from_params(params, pressure))

    def get_mass_only(self, config: DesignConfig) -> float:
        """
        Fast mass calculation without any analysis.

        Useful for quick filtering before full evaluation.
        """
        params = build_params_from_config(config)
        model = StructuralModel(params, get_section_lib(config.layout_strategy, self.section_lib_path))
        model.generate()
        return model.calculate_mass_kg()


# Module-level cache for parallel processing
# Each worker process will have its own cache
_EVALUATOR_CACHE: Dict[str, StructuralEvaluator] = {}


def get_evaluator(
    mode: str = "fea",
    surrogate_path: Optional[str] = None,
    force_new: bool = False,
) -> StructuralEvaluator:
    """
    Get or create cached evaluator.

    Uses a cache keyed by (mode, surrogate_path) to avoid reloading models.
    Each worker process maintains its own cache.

    Args:
        mode: "fea" or "surrogate"
        surrogate_path: Path to surrogate model (required for surrogate mode)
        force_new: If True, always create new evaluator

    Returns:
        StructuralEvaluator instance
    """
    cache_key = f"{mode}:{surrogate_path or 'none'}"

    if force_new or cache_key not in _EVALUATOR_CACHE:
        _EVALUATOR_CACHE[cache_key] = StructuralEvaluator(mode=mode, surrogate_path=surrogate_path)

    return _EVALUATOR_CACHE[cache_key]


# Standalone function for parallel processing
def evaluate_design(
    config: Optional[DesignConfig] = None,
    mode: str = "fea",
    surrogate_path: Optional[str] = None,
    **legacy_kwargs,
) -> Tuple[float, float, float, bool]:
    """
    Standalone evaluation function for parallel processing.

    Uses cached evaluator to avoid reloading model on every call.
    Each worker process maintains its own cache.

    Returns:
        (mass, u_sigma, u_delta, feasible)
    """
    evaluator = get_evaluator(mode=mode, surrogate_path=surrogate_path)
    try:
        if config is None:
            config = ResearchDesignConfig(
                L=legacy_kwargs["L"],
                W=legacy_kwargs["W"],
                H=legacy_kwargs["H"],
                pressure=legacy_kwargs["pressure"],
                main_beam_spacing=legacy_kwargs["main_beam_spacing"],
                num_secondary_width=legacy_kwargs["num_secondary_width"],
                num_secondary_height=legacy_kwargs["num_secondary_height"],
                add_support_beam=legacy_kwargs["add_support_beam"],
                frame_section_index=legacy_kwargs["sections"][0],
                main_beam_section_index=legacy_kwargs["sections"][1],
                secondary_beam_section_index=legacy_kwargs["sections"][2],
                brace_section_index=legacy_kwargs["sections"][3],
            )
        result = evaluator.evaluate(config)
        return result.mass, result.u_sigma, result.u_delta, result.feasible
    except Exception as e:
        print(f"Error occurred while evaluating design: {e}")
        return 1e9, 10.0, 10.0, False
