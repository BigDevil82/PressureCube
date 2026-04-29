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
from joblib import Parallel, delayed
from scipy.stats import qmc

from src.analysis.opensees_analysis import OpenSeesSolver
from src.analysis.post_processor import ResultProcessor
from src.optimization.design_space import (
    design_to_feature_vector,
    feature_names,
    get_layout_adapter,
    get_section_lib,
    sample_spec_from_unit,
)
from src.structure.enums import BeamType
from src.structure.mesh_generator import MeshGenerator
from src.structure.section_assignment import BEAM_SECTION_TYPES, SectionAssignment, zero_assignment
from src.structure.section_grouping.models import SectionGroupingRuleset
from src.structure.structural_design import StructuralDesign
from src.structure.structural_model import StructuralModel

# ============================================================================
# Data Collection
# ============================================================================
MESH_SIZE = 0.5  # Global mesh size for FEA


def _run_single_fea(model: StructuralModel, pressure: float) -> Tuple[float, float, float, bool]:
    """
    Run single FEA in isolated process.
    Returns: (u_sigma, u_delta, mass, success)
    """
    try:
        mass = model.calculate_mass_kg()
        mesh_generator = MeshGenerator(model, MESH_SIZE)
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
    GROUP_COUNT_RANGE = (1, 3)

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

        self.section_lib = get_section_lib(layout_strategy)
        self.beam_types = sorted(BEAM_SECTION_TYPES, key=lambda item: int(item.value))

    def _generate_lhs_samples(self) -> List[StructuralDesign]:
        """Generate samples using Latin Hypercube Sampling."""

        topology_specs = self.adapter.topology_specs()
        max_groups_per_type = self.GROUP_COUNT_RANGE[1]
        n_group_count_dims = len(self.beam_types)
        n_section_dims = len(self.beam_types) * max_groups_per_type
        # Dimensions: geometry(3) + topology + group counts(4) + section groups(up to 12) + pressure
        n_dims = 3 + len(topology_specs) + n_group_count_dims + n_section_dims + 1

        sampler = qmc.LatinHypercube(d=n_dims, seed=self.seed)
        rng = np.random.default_rng(self.seed)

        designs = []
        seen_designs = set()
        oversample_factor = 3
        max_attempts = 5
        attempts = 0

        while len(designs) < self.n_samples and attempts < max_attempts:
            remaining = self.n_samples - len(designs)
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

                group_counts = self._sample_group_counts(s[offset : offset + n_group_count_dims])
                offset += n_group_count_dims

                section_indices_by_group = self._sample_group_section_indices(
                    group_counts,
                    s[offset : offset + n_section_dims],
                )
                offset += n_section_dims

                pressure = self.LOAD_OPTIONS["pressure_range"][0] + s[offset] * (
                    self.LOAD_OPTIONS["pressure_range"][1] - self.LOAD_OPTIONS["pressure_range"][0]
                )

                geometry = {"L": round(L, 2), "W": round(W, 2), "H": round(H, 2)}
                pressure_pa = float(round(pressure, 2)) * 1000.0

                design = self._build_random_grouped_design(
                    geometry=geometry,
                    pressure=pressure_pa,
                    topology=topology,
                    group_counts=group_counts,
                    section_indices_by_group=section_indices_by_group,
                    rng=rng,
                )
                design_key = (
                    geometry["L"],
                    geometry["W"],
                    geometry["H"],
                    pressure_pa,
                    tuple(sorted(topology.items())),
                    tuple((beam_type.name, group_counts[beam_type]) for beam_type in self.beam_types),
                    tuple(sorted(section_indices_by_group.items())),
                    tuple(sorted(design.section_assignment.beam_groups.items())),
                )
                if design_key in seen_designs:
                    continue

                seen_designs.add(design_key)
                designs.append(design)

                if len(designs) >= self.n_samples:
                    break

        if len(designs) < self.n_samples:
            print(
                f"Warning: only generated {len(designs)} unique designs out of requested {self.n_samples}. "
                "Parameter space may be too small or too discretized."
            )

        return designs

    def _sample_group_counts(self, values: np.ndarray) -> dict[BeamType, int]:
        low, high = self.GROUP_COUNT_RANGE
        span = high - low + 1
        return {
            beam_type: min(low + int(value * span), high) for beam_type, value in zip(self.beam_types, values)
        }

    def _sample_group_section_indices(
        self,
        group_counts: dict[BeamType, int],
        values: np.ndarray,
    ) -> dict[str, int]:
        section_indices = {}
        value_offset = 0
        max_groups_per_type = self.GROUP_COUNT_RANGE[1]
        for beam_type in self.beam_types:
            section_count = self.section_lib.section_counts[BEAM_SECTION_TYPES[beam_type]]
            for group_index in range(1, group_counts[beam_type] + 1):
                value = values[value_offset + group_index - 1]
                section_indices[f"{beam_type.name}-SG{group_index}"] = min(
                    int(value * section_count),
                    section_count - 1,
                )
            value_offset += max_groups_per_type
        return section_indices

    def _build_random_grouped_design(
        self,
        *,
        geometry: dict[str, float],
        pressure: float,
        topology: dict,
        group_counts: dict[BeamType, int],
        section_indices_by_group: dict[str, int],
        rng: np.random.Generator,
    ) -> StructuralDesign:
        ruleset = SectionGroupingRuleset(
            layout_strategy=self.layout_strategy,
            n_groups_by_type=group_counts,
            prototypes=[],
            feature_schema=[],
            created_from_samples=0,
            description="Random sampled grouping for surrogate data.",
        )
        params = self.adapter.build_params(geometry=geometry, topology=topology)
        model = StructuralModel(params, self.section_lib, section_assignment=zero_assignment())
        model.generate()
        beam_groups = self._sample_beam_groups(model, ruleset, rng)
        assignment = SectionAssignment.from_group_indices(
            ruleset.group_beam_types(),
            section_indices_by_group,
            beam_groups=beam_groups,
        )
        return StructuralDesign(params=params, pressure=pressure, section_assignment=assignment)

    def _sample_beam_groups(
        self,
        model: StructuralModel,
        ruleset: SectionGroupingRuleset,
        rng: np.random.Generator,
    ) -> dict[int, str]:
        beam_groups = {}
        group_types = ruleset.group_beam_types()
        for beam_type in self.beam_types:
            group_ids = [group_id for group_id in ruleset.group_ids() if group_types[group_id] == beam_type]
            beam_ids = [
                beam_id
                for beam_id, beam in model.beams.items()
                if not beam.is_virtual and beam.beam_type == beam_type
            ]
            rng.shuffle(beam_ids)
            for offset, beam_id in enumerate(beam_ids):
                if offset < len(group_ids):
                    group_id = group_ids[offset]
                else:
                    group_id = group_ids[int(rng.integers(0, len(group_ids)))]
                beam_groups[beam_id] = group_id
        return beam_groups

    def collect(self, output_dir: str = "results/surrogate_data") -> str:
        """
        Collect training data via parallel FEA.
        Processes in batches to avoid memory issues.

        Returns: Path to saved dataset
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"Generating {self.n_samples} LHS samples...")
        designs = self._generate_lhs_samples()

        print(f"Running FEA in parallel (n_jobs={self.n_jobs})...")
        start_time = time.time()

        # Process in batches to avoid memory issues
        batch_size = 5000
        all_results = []

        for batch_start in range(0, len(designs), batch_size):
            batch_end = min(batch_start + batch_size, len(designs))
            batch_designs = designs[batch_start:batch_end]

            print(
                f"Processing batch {batch_start//batch_size + 1}/{(len(designs) + batch_size - 1)//batch_size} "
                f"(samples {batch_start}-{batch_end-1})..."
            )

            models = []
            for design in batch_designs:
                models.append(design.build_model(self.section_lib))

            batch_results = Parallel(n_jobs=self.n_jobs, backend="loky", verbose=10)(
                delayed(_run_single_fea)(model, design.pressure)
                for model, design in zip(models, batch_designs)
            )

            all_results.extend(batch_results)

            # Clear memory
            del models
            del batch_results

        results = all_results

        elapsed = time.time() - start_time
        print(
            f"FEA completed in {elapsed:.1f}s ({elapsed/len(designs):.2f}s per sample with {self.n_jobs} workers)"
        )

        # Build dataset
        X = []
        y_sigma = []
        y_delta = []
        masses = []
        valid_count = 0

        for design, res in zip(designs, results):
            # print(f"Processing sample: {design}")
            # print(f"  Result: {res}")
            u_sigma, u_delta, mass, success = res
            if success:
                X.append(design_to_feature_vector(design))
                y_sigma.append(u_sigma)
                y_delta.append(u_delta)
                masses.append(mass)
                valid_count += 1

        X = np.array(X)
        y_sigma = np.array(y_sigma)
        y_delta = np.array(y_delta)
        masses = np.array(masses)

        print(f"Valid samples: {valid_count}/{len(designs)} ({100*valid_count/len(designs):.1f}%)")

        # Save dataset
        dataset = {
            "X": X,
            "y_sigma": y_sigma,
            "y_delta": y_delta,
            "masses": masses,
            "feature_names": feature_names(self.layout_strategy),
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
                "group_count_range": self.GROUP_COUNT_RANGE,
            },
            "feature_names": feature_names(self.layout_strategy),
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
