from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.surrogate_data import SampleSpec
from src.optimization.design_space import (
    design_to_feature_vector,
    design_to_group_feature_vector,
    feature_names,
    group_feature_names,
)


def build_feature_matrix(dataset: dict, feature_format: str) -> tuple[np.ndarray, list[str]]:
    specs = dataset.get("sample_specs")
    if not specs:
        raise ValueError("Dataset does not contain sample_specs; features cannot be rebuilt.")

    layout_strategy = dataset["layout_strategy"]
    designs = [SampleSpec.from_dict(item).build_design() for item in specs]

    if feature_format == "legacy":
        X = np.array([design_to_feature_vector(design) for design in designs], dtype=np.float32)
        return X, feature_names(layout_strategy)
    if feature_format == "group_v1":
        X = np.array([design_to_group_feature_vector(design) for design in designs], dtype=np.float32)
        return X, group_feature_names(layout_strategy)
    raise ValueError(f"Unsupported feature_format: {feature_format}")


def main():
    parser = argparse.ArgumentParser(description="Rebuild surrogate dataset features from saved sample specs.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--feature-format", default="group_v1", choices=["legacy", "group_v1"])
    args = parser.parse_args()

    with open(args.dataset, "rb") as file:
        dataset = pickle.load(file)

    X, names = build_feature_matrix(dataset, args.feature_format)
    rebuilt = dict(dataset)
    rebuilt["X"] = X
    rebuilt["feature_names"] = names
    rebuilt["feature_format"] = args.feature_format

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        pickle.dump(rebuilt, file)

    print(f"Rebuilt features: {X.shape}, format={args.feature_format}")
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()
