from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.surrogate_data import SampleSpec
from src.optimization.design_space import (
    design_to_feature_vector,
    design_to_group_feature_vector,
    design_to_group_feature_vector_v2,
    feature_names,
    group_feature_names,
    group_feature_names_v2,
)


def _feature_names(layout_strategy: str, feature_format: str) -> list[str]:
    if feature_format == "legacy":
        return feature_names(layout_strategy)
    if feature_format == "group_v1":
        return group_feature_names(layout_strategy)
    if feature_format == "group_v2":
        return group_feature_names_v2(layout_strategy)
    raise ValueError(f"Unsupported feature_format: {feature_format}")


def _design_to_features(design, feature_format: str) -> np.ndarray:
    if feature_format == "legacy":
        return design_to_feature_vector(design)
    if feature_format == "group_v1":
        return design_to_group_feature_vector(design)
    if feature_format == "group_v2":
        return design_to_group_feature_vector_v2(design)
    raise ValueError(f"Unsupported feature_format: {feature_format}")


def _build_feature_chunk(specs: list[dict], feature_format: str) -> np.ndarray:
    rows = []
    for item in specs:
        design = SampleSpec.from_dict(item).build_design()
        rows.append(_design_to_features(design, feature_format))
    return np.array(rows, dtype=np.float32)


def _chunks(items: list[dict], chunk_size: int) -> list[list[dict]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def build_feature_matrix(
    dataset: dict,
    feature_format: str,
    n_jobs: int = 1,
    chunk_size: int = 1000,
    limit: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    specs = dataset.get("sample_specs")
    if not specs:
        raise ValueError("Dataset does not contain sample_specs; features cannot be rebuilt.")

    layout_strategy = dataset["layout_strategy"]
    names = _feature_names(layout_strategy, feature_format)
    specs = specs[:limit] if limit is not None else specs
    batches = _chunks(specs, chunk_size)

    if n_jobs == 1:
        arrays = [
            _build_feature_chunk(batch, feature_format)
            for batch in tqdm(batches, desc="Rebuilding feature chunks")
        ]
    else:
        arrays = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
            delayed(_build_feature_chunk)(batch, feature_format)
            for batch in batches
        )
    return np.vstack(arrays).astype(np.float32), names


def main():
    parser = argparse.ArgumentParser(description="Rebuild surrogate dataset features from saved sample specs.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--feature-format", default="group_v1", choices=["legacy", "group_v1", "group_v2"])
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.dataset, "rb") as file:
        dataset = pickle.load(file)

    X, names = build_feature_matrix(
        dataset,
        args.feature_format,
        n_jobs=args.n_jobs,
        chunk_size=args.chunk_size,
        limit=args.limit,
    )
    rebuilt = dict(dataset)
    rebuilt["X"] = X
    rebuilt["feature_names"] = names
    rebuilt["feature_format"] = args.feature_format
    if args.limit is not None:
        n = len(X)
        for key in ["y_sigma", "y_delta", "masses", "sample_specs"]:
            if key in rebuilt:
                rebuilt[key] = rebuilt[key][:n]
        rebuilt["n_samples"] = n

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        pickle.dump(rebuilt, file)

    print(f"Rebuilt features: {X.shape}, format={args.feature_format}")
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()
