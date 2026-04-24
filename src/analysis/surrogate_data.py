"""
Surrogate Model for FEA Acceleration.

Trains a machine learning model to predict structural responses (stress ratio, displacement ratio)
from design parameters, replacing expensive FEA simulations during optimization.

Key features:
1. Data collection via Latin Hypercube Sampling (LHS) with parallel FEA
2. XGBoost/LightGBM for fast, accurate predictions
3. Fast mass calculation without FEA
4. Drop-in replacement interface for optimization loops
"""

import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from joblib import Parallel, delayed
from scipy.stats import qmc
from torch.utils.data import DataLoader, TensorDataset

from src.analysis.opensees_analysis import OpenSeesSolver
from src.analysis.post_processor import ResultProcessor
from src.structure.layout.params import StructuralParameters
from src.structure.mesh_generator import MeshGenerator
from src.structure.section_library import SectionLibrary, SectionType
from src.structure.structural_model import StructuralModel

# ============================================================================
# Feature Engineering
# ============================================================================

# Global section library for feature extraction
_SECTION_LIB: Optional[SectionLibrary] = None


def get_section_lib() -> SectionLibrary:
    """Get or create singleton section library."""
    global _SECTION_LIB
    if _SECTION_LIB is None:
        _SECTION_LIB = SectionLibrary("src/data/steel_sections_large.json")
    return _SECTION_LIB


@dataclass
class DesignConfig:
    """A complete design configuration for surrogate model."""

    # Geometry
    L: float
    W: float
    H: float
    pressure: float  # in Pa

    # Topology
    main_beam_spacing: float
    num_secondary_width: int
    num_secondary_height: int
    add_support_beam: bool

    # Sections (indices)
    frame_section_index: int
    main_beam_section_index: int
    secondary_beam_section_index: int
    brace_section_index: int

    def to_feature_vector(self, use_physics: bool = True) -> np.ndarray:
        """
        Convert to feature vector for ML model.

        Args:
            use_physics: If True, use physical section properties instead of indices
        """
        if use_physics:
            return self._to_physics_features()
        else:
            raise NotImplementedError(
                "Index-based features are deprecated. Use physics-based features instead."
            )

    def _to_physics_features(self) -> np.ndarray:
        """
        Physics-based features using section properties.

        Features:
        - Geometry: L, W, H, volume (L*W*H)
        - Load: pressure, total_load (pressure * L * W)
        - Topology: spacing, n_sec_w, n_sec_h, add_support, n_main_beams, n_sec_beams
        - Section properties (for each of 4 types): area, Ix, Iy
        - Derived: total_area, avg_Ix, stiffness_ratio
        """
        lib = get_section_lib()
        sec_types = [SectionType.BOX, SectionType.H_MAIN, SectionType.H_SEC, SectionType.CHANNEL]
        indices = [
            self.frame_section_index,
            self.main_beam_section_index,
            self.secondary_beam_section_index,
            self.brace_section_index,
        ]

        # Get section properties
        sections = [lib.get_section(st, idx) for st, idx in zip(sec_types, indices)]

        # Geometry features
        volume = self.L * self.W * self.H
        floor_area = self.L * self.W
        aspect_ratio = self.L / self.W if self.W > 0 else 1.0

        # Load features
        pressure_kPa = self.pressure / 1000.0  # Pa -> kPa
        total_load = self.pressure * floor_area  # Total pressure force (N)

        # Topology features
        n_main_beams = int(self.L / self.main_beam_spacing) + 1

        # Section physical properties (scaled for numerical stability)
        # area in cm², Ix in cm⁴ for better numerical range
        areas = [s.area * 1e4 for s in sections]  # m² -> cm²
        Ixs = [s.Ix * 1e8 for s in sections]  # m⁴ -> cm⁴
        Iys = [s.Iy * 1e8 for s in sections]

        # Derived features
        total_area = sum(areas)
        avg_Ix = np.mean(Ixs)
        min_Ix = min(Ixs)
        max_Ix = max(Ixs)
        Ix_ratio = min_Ix / (max_Ix + 1e-10)  # Stiffness balance

        # Stiffness-to-load ratio (proxy for displacement)
        stiffness_load_ratio = avg_Ix / (total_load / 1e6 + 1e-10)

        # Strength proxy: area / load
        strength_load_ratio = total_area / (total_load / 1e6 + 1e-10)

        features = [
            # Geometry (6)
            self.L,
            self.W,
            self.H,
            volume,
            floor_area,
            aspect_ratio,
            # Load (2)
            pressure_kPa,
            total_load / 1e6,  # Scale to MN
            # Topology (5)
            self.main_beam_spacing,
            float(self.num_secondary_width),
            float(self.num_secondary_height),
            float(self.add_support_beam),
            float(n_main_beams),
            # Section areas (4)
            areas[0],
            areas[1],
            areas[2],
            areas[3],
            # Section Ix (4)
            Ixs[0],
            Ixs[1],
            Ixs[2],
            Ixs[3],
            # Section Iy (4)
            Iys[0],
            Iys[1],
            Iys[2],
            Iys[3],
            # Derived (6)
            total_area,
            avg_Ix,
            min_Ix,
            Ix_ratio,
            stiffness_load_ratio,
            strength_load_ratio,
        ]

        return np.array(features, dtype=np.float32)

    @staticmethod
    def feature_names(use_physics: bool = True) -> List[str]:
        if use_physics:
            return ["L", "W", "H", "volume", "floor_area", "aspect_ratio", "pressure_kPa", "total_load_MN", "main_beam_spacing", "num_secondary_width", "num_secondary_height", "add_support_beam", "n_main_beams",  "frame_area", "main_area", "sec_area", "brace_area", "frame_Ix", "main_Ix", "sec_Ix", "brace_Ix","frame_Iy", "main_Iy", "sec_Iy", "brace_Iy", "total_area", "avg_Ix", "min_Ix", "Ix_ratio", "stiffness_load_ratio", "strength_load_ratio",
            ] # fmt: skip
        else:
            raise NotImplementedError(
                "Index-based features are deprecated. Use physics-based features instead."
            )


def compute_mass_fast(config: DesignConfig, section_lib: SectionLibrary) -> float:
    """
    Fast mass calculation without FEA.
    Only builds the structural model and computes mass.
    """
    params = StructuralParameters(
        length=config.L,
        width=config.W,
        height=config.H,
        main_beam_spacing=config.main_beam_spacing,
        num_secondary_width=config.num_secondary_width,
        num_secondary_height=config.num_secondary_height,
        add_support_beam=config.add_support_beam,
        frame_section_index=config.frame_section_index,
        main_beam_section_index=config.main_beam_section_index,
        secondary_beam_section_index=config.secondary_beam_section_index,
        brace_section_index=config.brace_section_index,
    )
    model = StructuralModel(params, section_lib)
    model.generate()
    return model.calculate_mass_kg()


def detect_feature_format(X: np.ndarray) -> str:
    """
    Detect whether features are in old index format (12-dim) or new physics format (31-dim).

    Returns: "index" or "physics"
    """
    n_features = X.shape[1]
    if n_features == 12:
        print(
            "Detected index-based features (12 dimensions). This format is deprecated. Please use physics-based features instead."
        )
        return "index"
    elif n_features == 31:
        return "physics"
    else:
        raise ValueError(f"Unknown feature format with {n_features} dimensions. Expected 12 or 31.")


# ============================================================================
# Data Collection
# ============================================================================


def _run_single_fea(model: StructuralModel, pressure: float) -> Tuple[float, float, float, bool]:
    """
    Run single FEA in isolated process.
    Returns: (u_sigma, u_delta, mass, success)
    """
    try:
        mass = model.calculate_mass_kg()
        mesh_generator = MeshGenerator(model, 0.5)
        model = mesh_generator.generate()

        solver = OpenSeesSolver(model)
        combo_strength = {"DL": 1.3, "LL": 1.5, "PL": 1.25}
        combo_service = {"DL": 1.0, "LL": 1.0, "PL": 1.0}

        res_strength = solver.run_load_combination(combo_strength, pressure_pa=pressure)
        strength_check = ResultProcessor(model).run_gb50017_check(res_strength)
        u_sigma = strength_check.max_stress_ratio

        res_service = solver.run_load_combination(combo_service, pressure_pa=pressure)
        disp_check = ResultProcessor(model).run_gb50017_check(res_service)
        u_delta = disp_check.max_deflection_ratio

        return (u_sigma, u_delta, mass, True)

    except Exception as e:
        print(f"FEA failed: {e}")
        return (0.0, 0.0, 0.0, False)


class DataCollector:
    """
    Collect training data for surrogate model using Latin Hypercube Sampling.
    """

    # Default parameter ranges
    GEOMETRY_RANGES = {
        "L": (6.0, 12.0),
        "W": (3.0, 4.5),
        "H": (3.0, 4.5),
    }

    TOPOLOGY_OPTIONS = {
        "main_beam_spacing": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        "num_secondary_width": [2, 3, 4, 5, 6],
        "num_secondary_height": [2, 3, 4, 5, 6],
        "add_support_beam": [True, False],
    }

    LOAD_OPTIONS = {
        "pressure_range": (20.0, 40.0),  # in kPa
    }

    def __init__(
        self,
        n_samples: int = 2000,
        n_jobs: int = 4,
        seed: int = 42,
    ):
        self.n_samples = n_samples
        self.n_jobs = n_jobs
        self.seed = seed

        # Get section counts
        self.section_lib = SectionLibrary("src/data/steel_sections_large.json")
        self.sec_types = [SectionType.BOX, SectionType.H_MAIN, SectionType.H_SEC, SectionType.CHANNEL]
        self.sec_counts = [self.section_lib.section_counts[t] for t in self.sec_types]

    def _generate_lhs_samples(self) -> List[DesignConfig]:
        """Generate samples using Latin Hypercube Sampling."""

        # Dimensions: L, W, H, spacing_idx, sec_w_idx, sec_h_idx, support_idx, 4 section indices, pressure
        n_dims = 12

        sampler = qmc.LatinHypercube(d=n_dims, seed=self.seed)
        samples = sampler.random(n=self.n_samples)  # [0, 1] range

        configs = []
        for i in range(self.n_samples):
            s = samples[i]

            # Map to actual values
            L = self.GEOMETRY_RANGES["L"][0] + s[0] * (
                self.GEOMETRY_RANGES["L"][1] - self.GEOMETRY_RANGES["L"][0]
            )
            W = self.GEOMETRY_RANGES["W"][0] + s[1] * (
                self.GEOMETRY_RANGES["W"][1] - self.GEOMETRY_RANGES["W"][0]
            )
            H = self.GEOMETRY_RANGES["H"][0] + s[2] * (
                self.GEOMETRY_RANGES["H"][1] - self.GEOMETRY_RANGES["H"][0]
            )

            # Topology (discrete)
            spacing_idx = int(s[3] * len(self.TOPOLOGY_OPTIONS["main_beam_spacing"]))
            spacing_idx = min(spacing_idx, len(self.TOPOLOGY_OPTIONS["main_beam_spacing"]) - 1)
            spacing = self.TOPOLOGY_OPTIONS["main_beam_spacing"][spacing_idx]

            sec_w_idx = int(s[4] * len(self.TOPOLOGY_OPTIONS["num_secondary_width"]))
            sec_w_idx = min(sec_w_idx, len(self.TOPOLOGY_OPTIONS["num_secondary_width"]) - 1)
            sec_w = self.TOPOLOGY_OPTIONS["num_secondary_width"][sec_w_idx]

            sec_h_idx = int(s[5] * len(self.TOPOLOGY_OPTIONS["num_secondary_height"]))
            sec_h_idx = min(sec_h_idx, len(self.TOPOLOGY_OPTIONS["num_secondary_height"]) - 1)
            sec_h = self.TOPOLOGY_OPTIONS["num_secondary_height"][sec_h_idx]

            add_support = s[6] > 0.5

            # Section indices
            sections = [
                int(s[7] * self.sec_counts[0]),
                int(s[8] * self.sec_counts[1]),
                int(s[9] * self.sec_counts[2]),
                int(s[10] * self.sec_counts[3]),
            ]
            # Clamp to valid range
            sections = [min(sec, cnt - 1) for sec, cnt in zip(sections, self.sec_counts)]

            pressure = self.LOAD_OPTIONS["pressure_range"][0] + s[11] * (
                self.LOAD_OPTIONS["pressure_range"][1] - self.LOAD_OPTIONS["pressure_range"][0]
            )

            configs.append(
                DesignConfig(
                    L=round(L, 2),
                    W=round(W, 2),
                    H=round(H, 2),
                    pressure=float(round(pressure, 2)) * 1000.0,  # convert to Pa
                    main_beam_spacing=spacing,
                    num_secondary_width=sec_w,
                    num_secondary_height=sec_h,
                    add_support_beam=add_support,
                    frame_section_index=sections[0],
                    main_beam_section_index=sections[1],
                    secondary_beam_section_index=sections[2],
                    brace_section_index=sections[3],
                )
            )

        return configs

    def collect(self, output_dir: str = "results/surrogate_data") -> str:
        """
        Collect training data via parallel FEA.
        Processes in batches to avoid memory issues.

        Returns: Path to saved dataset
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"Generating {self.n_samples} LHS samples...")
        configs = self._generate_lhs_samples()

        print(f"Running FEA in parallel (n_jobs={self.n_jobs})...")
        start_time = time.time()

        # Process in batches to avoid memory issues
        batch_size = 5000
        all_results = []

        for batch_start in range(0, self.n_samples, batch_size):
            batch_end = min(batch_start + batch_size, self.n_samples)
            batch_configs = configs[batch_start:batch_end]

            print(
                f"Processing batch {batch_start//batch_size + 1}/{(self.n_samples + batch_size - 1)//batch_size} "
                f"(samples {batch_start}-{batch_end-1})..."
            )

            models = []
            for cfg in batch_configs:
                model = StructuralModel(
                    StructuralParameters(
                        length=cfg.L,
                        width=cfg.W,
                        height=cfg.H,
                        main_beam_spacing=cfg.main_beam_spacing,
                        num_secondary_width=cfg.num_secondary_width,
                        num_secondary_height=cfg.num_secondary_height,
                        add_support_beam=cfg.add_support_beam,
                        frame_section_index=cfg.frame_section_index,
                        main_beam_section_index=cfg.main_beam_section_index,
                        secondary_beam_section_index=cfg.secondary_beam_section_index,
                        brace_section_index=cfg.brace_section_index,
                    ),
                    self.section_lib,
                )
                model.generate()
                models.append(model)

            batch_results = Parallel(n_jobs=self.n_jobs, backend="loky", verbose=10)(
                delayed(_run_single_fea)(model, cfg.pressure) for model, cfg in zip(models, batch_configs)
            )

            all_results.extend(batch_results)

            # Clear memory
            del models
            del batch_results

        results = all_results

        elapsed = time.time() - start_time
        print(
            f"FEA completed in {elapsed:.1f}s ({elapsed/self.n_samples:.2f}s per sample with {self.n_jobs} workers)"
        )

        # Build dataset
        X = []
        y_sigma = []
        y_delta = []
        masses = []
        valid_count = 0

        for cfg, res in zip(configs, results):
            # print(f"Processing sample: {cfg}")
            # print(f"  Result: {res}")
            u_sigma, u_delta, mass, success = res
            if success:
                X.append(cfg.to_feature_vector())
                y_sigma.append(u_sigma)
                y_delta.append(u_delta)
                masses.append(mass)
                valid_count += 1

        X = np.array(X)
        y_sigma = np.array(y_sigma)
        y_delta = np.array(y_delta)
        masses = np.array(masses)

        print(f"Valid samples: {valid_count}/{self.n_samples} ({100*valid_count/self.n_samples:.1f}%)")

        # Save dataset
        dataset = {
            "X": X,
            "y_sigma": y_sigma,
            "y_delta": y_delta,
            "masses": masses,
            "feature_names": DesignConfig.feature_names(),
            "n_samples": valid_count,
            "collection_time": elapsed,
        }

        save_path = os.path.join(output_dir, "dataset.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(dataset, f)

        # Also save as JSON for inspection
        json_data = {
            "n_samples": valid_count,
            "collection_time": elapsed,
            "feature_names": DesignConfig.feature_names(),
            "y_sigma_stats": {
                "min": float(y_sigma.min()),
                "max": float(y_sigma.max()),
                "mean": float(y_sigma.mean()),
            },
            "y_delta_stats": {
                "min": float(y_delta.min()),
                "max": float(y_delta.max()),
                "mean": float(y_delta.mean()),
            },
        }
        with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
            json.dump(json_data, f, indent=2)

        print(f"Dataset saved to: {save_path}")
        return save_path


# ============================================================================
# CLI
# ============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Surrogate Model for FEA Acceleration")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Data collection
    collect_parser = subparsers.add_parser("collect", help="Collect training data")
    collect_parser.add_argument("--n_samples", type=int, default=2000, help="Number of samples")
    collect_parser.add_argument("--n_jobs", type=int, default=4, help="Parallel workers")
    collect_parser.add_argument("--output_dir", type=str, default="results/surrogate_data")
    collect_parser.add_argument("--seed", type=int, default=42)

    # Train model
    train_parser = subparsers.add_parser("train", help="Train surrogate model")
    train_parser.add_argument("--dataset", type=str, default="results/surrogate_data/dataset.pkl")
    train_parser.add_argument("--model_type", type=str, default="xgb", choices=["xgb", "lgb", "mlp"])
    train_parser.add_argument("--output", type=str, default="results/surrogate_data/model.pkl")

    args = parser.parse_args()

    if args.command == "collect":
        collector = DataCollector(
            n_samples=args.n_samples,
            n_jobs=args.n_jobs,
            seed=args.seed,
        )
        collector.collect(output_dir=args.output_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
