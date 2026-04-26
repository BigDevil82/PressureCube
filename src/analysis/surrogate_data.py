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
from src.optimization.design_space import (
    DesignConfig,
    build_params_from_config,
    compute_mass_fast,
    get_layout_adapter,
    get_section_lib,
    sample_spec_from_unit,
)
from src.structure.mesh_generator import MeshGenerator
from src.structure.section_library import SectionLibrary, SectionType
from src.structure.structural_model import StructuralModel

# ============================================================================
# Feature Engineering
# ============================================================================


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
    elif n_features >= 30:
        return "physics"
    else:
        raise ValueError(f"Unknown feature format with {n_features} dimensions. Expected 12 or >=30.")


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
        mesh_generator = MeshGenerator(model, 0.3)
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
        "W": (3.4, 4.0),
        "H": (3.0, 3.6),
    }

    LOAD_OPTIONS = {
        "pressure_range": (20.0, 40.0),  # in kPa
    }

    def __init__(
        self,
        n_samples: int = 2000,
        n_jobs: int = 4,
        seed: int = 42,
        layout_strategy: str = "research",
    ):
        self.n_samples = n_samples
        self.n_jobs = n_jobs
        self.seed = seed
        self.layout_strategy = layout_strategy
        self.adapter = get_layout_adapter(layout_strategy)

        # Get section counts
        self.section_lib = get_section_lib(layout_strategy)
        self.sec_types = [SectionType.BOX, SectionType.H_MAIN, SectionType.H_SEC, SectionType.CHANNEL]
        self.sec_counts = [self.section_lib.section_counts[t] for t in self.sec_types]

    def _generate_lhs_samples(self) -> List[DesignConfig]:
        """Generate samples using Latin Hypercube Sampling."""

        topology_specs = self.adapter.topology_specs()
        # Dimensions: geometry(3) + topology + sections(4) + pressure
        n_dims = 3 + len(topology_specs) + 4 + 1

        sampler = qmc.LatinHypercube(d=n_dims, seed=self.seed)

        configs = []
        seen_configs = set()
        oversample_factor = 3
        max_attempts = 5
        attempts = 0

        while len(configs) < self.n_samples and attempts < max_attempts:
            remaining = self.n_samples - len(configs)
            batch_size = max(remaining * oversample_factor, remaining)
            samples = sampler.random(n=batch_size)  # [0, 1] range
            attempts += 1

            for i in range(batch_size):
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

                offset = 3
                topology = {
                    spec.name: sample_spec_from_unit(spec, s[offset + idx])
                    for idx, spec in enumerate(topology_specs)
                }
                offset += len(topology_specs)

                sections = [
                    int(s[offset + 0] * self.sec_counts[0]),
                    int(s[offset + 1] * self.sec_counts[1]),
                    int(s[offset + 2] * self.sec_counts[2]),
                    int(s[offset + 3] * self.sec_counts[3]),
                ]
                sections = [min(sec, cnt - 1) for sec, cnt in zip(sections, self.sec_counts)]
                offset += 4

                pressure = self.LOAD_OPTIONS["pressure_range"][0] + s[offset] * (
                    self.LOAD_OPTIONS["pressure_range"][1] - self.LOAD_OPTIONS["pressure_range"][0]
                )

                geometry = {"L": round(L, 2), "W": round(W, 2), "H": round(H, 2)}
                pressure_pa = float(round(pressure, 2)) * 1000.0
                config_key = (
                    geometry["L"],
                    geometry["W"],
                    geometry["H"],
                    pressure_pa,
                    tuple(sorted(topology.items())),
                    tuple(sections),
                )
                if config_key in seen_configs:
                    continue

                seen_configs.add(config_key)
                configs.append(
                    self.adapter.build_config(
                        geometry=geometry,
                        pressure=pressure_pa,
                        topology=topology,
                        sections=sections,
                    )
                )

                if len(configs) >= self.n_samples:
                    break

        if len(configs) < self.n_samples:
            print(
                f"Warning: only generated {len(configs)} unique configs out of requested {self.n_samples}. "
                "Parameter space may be too small or too discretized."
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
                model = StructuralModel(build_params_from_config(cfg), self.section_lib)
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
            "feature_names": DesignConfig.feature_names(self.layout_strategy),
            "layout_strategy": self.layout_strategy,
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
            "config": {
                "geometry_ranges": self.GEOMETRY_RANGES,
                "load_options": self.LOAD_OPTIONS,
            },
            "feature_names": DesignConfig.feature_names(self.layout_strategy),
            "layout_strategy": self.layout_strategy,
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
    # python -m src.analysis.surrogate_data collect --n_samples 200000 --n_jobs 12 --output_dir results\surrogate_data\engineering_200k
    collect_parser = subparsers.add_parser("collect", help="Collect training data")
    collect_parser.add_argument("--n_samples", type=int, default=2000, help="Number of samples")
    collect_parser.add_argument("--n_jobs", type=int, default=4, help="Parallel workers")
    collect_parser.add_argument("--output_dir", type=str, default="results/surrogate_data")
    collect_parser.add_argument(
        "--layout_strategy",
        type=str,
        default="research",
        choices=["research", "engineering"],
        help="Layout strategy to use for data collection",
    )
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
            layout_strategy=args.layout_strategy,
        )
        collector.collect(output_dir=args.output_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
