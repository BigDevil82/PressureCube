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
from joblib import Parallel, delayed
from scipy.stats import qmc
from tqdm import tqdm

from src.analysis.opensees_analysis import OpenSeesSolver
from src.analysis.post_processor import ResultProcessor
from src.optimization.design_space import (
    LayoutDesignAdapter,
    design_to_group_feature_vector,
    group_feature_names,
    get_layout_adapter,
    get_section_lib,
    sample_spec_from_unit,
)
from src.structure.enums import BeamType
from src.structure.mesh_generator import MeshGenerator
from src.structure.section_assignment import BEAM_SECTION_TYPES, SectionAssignment
from src.structure.section_grouping.models import SectionGroupingRuleset
from src.structure.section_library import SectionLibrary
from src.structure.structural_design import StructuralDesign
from src.structure.structural_model import StructuralModel

# ============================================================================
# Data Collection
# ============================================================================
MESH_SIZE = 0.5  # Global mesh size for FEA


@dataclass(frozen=True)
class SampleSpec:
    geometry: dict[str, float]
    pressure: float
    topology: dict
    group_counts: dict[BeamType, int]
    section_indices_by_group: dict[str, int]
    layout_strategy: str
    group_assignment_method: str = "geometric_kmeans_v1"

    def build_design(self) -> StructuralDesign:
        adapter = get_layout_adapter(self.layout_strategy)
        return _build_random_grouped_design(
            geometry=self.geometry,
            pressure=self.pressure,
            topology=self.topology,
            group_counts=self.group_counts,
            section_indices_by_group=self.section_indices_by_group,
            layout_strategy=self.layout_strategy,
            adapter=adapter,
        )

    def to_dict(self) -> dict:
        return {
            "geometry": {key: float(value) for key, value in self.geometry.items()},
            "pressure": float(self.pressure),
            "topology": {key: _plain_scalar(value) for key, value in self.topology.items()},
            "group_counts": {beam_type.name: int(count) for beam_type, count in self.group_counts.items()},
            "section_indices_by_group": {
                group_id: int(index) for group_id, index in self.section_indices_by_group.items()
            },
            "layout_strategy": self.layout_strategy,
            "group_assignment_method": self.group_assignment_method,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SampleSpec":
        return cls(
            geometry=dict(data["geometry"]),
            pressure=float(data["pressure"]),
            topology=dict(data["topology"]),
            group_counts={BeamType[name]: int(count) for name, count in data["group_counts"].items()},
            section_indices_by_group={
                group_id: int(index) for group_id, index in data["section_indices_by_group"].items()
            },
            layout_strategy=data["layout_strategy"],
            group_assignment_method=data.get("group_assignment_method", "geometric_kmeans_v1"),
        )


def _plain_scalar(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return value


def _generate_design_chunk(
    samples: np.ndarray,
    start_index: int,
    layout_strategy: str,
    seed: int,
    geometry_ranges: dict,
    load_options: dict,
    group_count_range: tuple[int, int],
) -> list[SampleSpec]:
    adapter = get_layout_adapter(layout_strategy)
    section_lib = get_section_lib(layout_strategy)
    beam_types = sorted(BEAM_SECTION_TYPES, key=lambda item: int(item.value))
    topology_specs = adapter.topology_specs()
    max_groups_per_type = group_count_range[1]
    n_group_count_dims = len(beam_types)
    n_section_dims = len(beam_types) * max_groups_per_type

    specs = []
    for local_index, sample in enumerate(samples):
        sample_index = start_index + local_index

        L = geometry_ranges["L"][0] + sample[0] * (geometry_ranges["L"][1] - geometry_ranges["L"][0])
        W = geometry_ranges["W"][0] + sample[1] * (geometry_ranges["W"][1] - geometry_ranges["W"][0])
        H = geometry_ranges["H"][0] + sample[2] * (geometry_ranges["H"][1] - geometry_ranges["H"][0])

        offset = 3
        topology = {
            spec.name: sample_spec_from_unit(spec, sample[offset + idx])
            for idx, spec in enumerate(topology_specs)
        }
        offset += len(topology_specs)

        group_counts = _sample_group_counts(
            sample[offset : offset + n_group_count_dims],
            beam_types,
            group_count_range,
        )
        offset += n_group_count_dims

        section_indices_by_group = _sample_group_section_indices(
            group_counts,
            sample[offset : offset + n_section_dims],
            beam_types,
            group_count_range,
            section_lib,
        )
        offset += n_section_dims

        pressure = load_options["pressure_range"][0] + sample[offset] * (
            load_options["pressure_range"][1] - load_options["pressure_range"][0]
        )

        specs.append(
            SampleSpec(
                geometry={"L": round(L, 2), "W": round(W, 2), "H": round(H, 2)},
                pressure=float(round(pressure, 2)) * 1000.0,
                topology=topology,
                group_counts=group_counts,
                section_indices_by_group=section_indices_by_group,
                layout_strategy=layout_strategy,
            )
        )
    return specs


def _sample_group_counts(
    values: np.ndarray,
    beam_types: list[BeamType],
    group_count_range: tuple[int, int],
) -> dict[BeamType, int]:
    low, high = group_count_range
    span = high - low + 1
    return {beam_type: min(low + int(value * span), high) for beam_type, value in zip(beam_types, values)}


def _sample_group_section_indices(
    group_counts: dict[BeamType, int],
    values: np.ndarray,
    beam_types: list[BeamType],
    group_count_range: tuple[int, int],
    section_lib: SectionLibrary,
) -> dict[str, int]:
    section_indices = {}
    value_offset = 0
    max_groups_per_type = group_count_range[1]
    for beam_type in beam_types:
        section_count = section_lib.section_counts[BEAM_SECTION_TYPES[beam_type]]
        for group_index in range(1, group_counts[beam_type] + 1):
            value = values[value_offset + group_index - 1]
            section_indices[f"{beam_type.name}-SG{group_index}"] = min(
                int(value * section_count),
                section_count - 1,
            )
        value_offset += max_groups_per_type
    return section_indices


def _build_random_grouped_design(
    *,
    geometry: dict[str, float],
    pressure: float,
    topology: dict,
    group_counts: dict[BeamType, int],
    section_indices_by_group: dict[str, int],
    layout_strategy: str,
    adapter: LayoutDesignAdapter,
) -> StructuralDesign:
    ruleset = SectionGroupingRuleset(
        layout_strategy=layout_strategy,
        n_groups_by_type=group_counts,
        prototypes=[],
        feature_schema=[],
        created_from_samples=0,
        description="Geometry-clustered grouping for surrogate data.",
    )
    params = adapter.build_params(geometry=geometry, topology=topology)
    assignment = SectionAssignment.from_group_indices(
        ruleset.group_beam_types(),
        section_indices_by_group,
        assign_missing_by_geometry=True,
    )
    return StructuralDesign(params=params, pressure=pressure, section_assignment=assignment)


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


def _run_single_spec_fea(
    spec: SampleSpec,
    layout_strategy: str,
) -> Tuple[float, float, float, bool]:
    section_lib = get_section_lib(layout_strategy)
    design = spec.build_design()
    model = design.build_model(section_lib)
    return _run_single_fea(model, design.pressure)


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

    def _generate_lhs_samples(self) -> List[SampleSpec]:
        """Generate samples using Latin Hypercube Sampling."""

        topology_specs = self.adapter.topology_specs()
        max_groups_per_type = self.GROUP_COUNT_RANGE[1]
        n_group_count_dims = len(self.beam_types)
        n_section_dims = len(self.beam_types) * max_groups_per_type
        # Dimensions: geometry(3) + topology + group counts(4) + section groups(up to 12) + pressure
        n_dims = 3 + len(topology_specs) + n_group_count_dims + n_section_dims + 1

        sampler = qmc.LatinHypercube(d=n_dims, seed=self.seed)
        samples = sampler.random(n=self.n_samples)
        n_jobs = max(1, min(self.n_jobs, self.n_samples))
        n_chunks = min(self.n_samples, max(n_jobs * 4, 1))
        chunk_sizes = np.full(n_chunks, self.n_samples // n_chunks, dtype=int)
        chunk_sizes[: self.n_samples % n_chunks] += 1

        chunks = []
        start = 0
        for chunk_size in chunk_sizes:
            end = start + int(chunk_size)
            chunks.append((start, samples[start:end]))
            start = end

        if n_jobs == 1:
            specs = []
            with tqdm(total=self.n_samples, desc="Generating LHS samples") as pbar:
                for start_index, chunk in chunks:
                    chunk_specs = _generate_design_chunk(
                        chunk,
                        start_index,
                        self.layout_strategy,
                        self.seed,
                        self.GEOMETRY_RANGES,
                        self.LOAD_OPTIONS,
                        self.GROUP_COUNT_RANGE,
                    )
                    specs.extend(chunk_specs)
                    pbar.update(len(chunk_specs))
            return specs

        chunk_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
            delayed(_generate_design_chunk)(
                chunk,
                start_index,
                self.layout_strategy,
                self.seed,
                self.GEOMETRY_RANGES,
                self.LOAD_OPTIONS,
                self.GROUP_COUNT_RANGE,
            )
            for start_index, chunk in chunks
        )
        return [spec for chunk_specs in chunk_results for spec in chunk_specs]

    def collect(self, output_dir: str = "results/surrogate_data") -> str:
        """
        Collect training data via parallel FEA.
        Processes in batches to avoid memory issues.

        Returns: Path to saved dataset
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"Generating {self.n_samples} LHS samples...")
        specs = self._generate_lhs_samples()

        print(f"Running FEA in parallel (n_jobs={self.n_jobs})...")
        start_time = time.time()

        # Process in batches to avoid memory issues
        batch_size = 5000
        all_results = []

        for batch_start in range(0, len(specs), batch_size):
            batch_end = min(batch_start + batch_size, len(specs))
            batch_specs = specs[batch_start:batch_end]

            print(
                f"Processing batch {batch_start//batch_size + 1}/{(len(specs) + batch_size - 1)//batch_size} "
                f"(samples {batch_start}-{batch_end-1})..."
            )

            batch_results = Parallel(n_jobs=self.n_jobs, backend="loky", verbose=10)(
                delayed(_run_single_spec_fea)(spec, self.layout_strategy) for spec in batch_specs
            )

            all_results.extend(batch_results)

            # Clear memory
            del batch_results

        results = all_results

        elapsed = time.time() - start_time
        print(
            f"FEA completed in {elapsed:.1f}s ({elapsed/len(specs):.2f}s per sample with {self.n_jobs} workers)"
        )

        # Build dataset
        X = []
        y_sigma = []
        y_delta = []
        masses = []
        valid_count = 0

        sample_specs = []
        for spec, res in zip(specs, results):
            # print(f"Processing sample: {spec}")
            # print(f"  Result: {res}")
            u_sigma, u_delta, mass, success = res
            if success:
                design = spec.build_design()
                X.append(design_to_group_feature_vector(design))
                y_sigma.append(u_sigma)
                y_delta.append(u_delta)
                masses.append(mass)
                sample_specs.append(spec.to_dict())
                valid_count += 1

        X = np.array(X)
        y_sigma = np.array(y_sigma)
        y_delta = np.array(y_delta)
        masses = np.array(masses)

        print(f"Valid samples: {valid_count}/{len(specs)} ({100*valid_count/len(specs):.1f}%)")

        # Save dataset
        dataset = {
            "X": X,
            "y_sigma": y_sigma,
            "y_delta": y_delta,
            "masses": masses,
            "feature_names": group_feature_names(self.layout_strategy),
            "feature_format": "group_v1",
            "sample_spec_format": "structural_design_v1",
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
            "feature_names": group_feature_names(self.layout_strategy),
            "feature_format": "group_v1",
            "sample_spec_format": "structural_design_v1",
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
