from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.surrogate_data import DataCollector, SampleSpec
from src.optimization.design_space import design_to_group_feature_vector


def main():
    parser = argparse.ArgumentParser(description="Recreate deterministic sample specs for an old grouped dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--check-rows", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-5)
    args = parser.parse_args()

    with open(args.dataset, "rb") as file:
        dataset = pickle.load(file)

    if dataset.get("sample_specs"):
        print("Dataset already contains sample_specs.")
        specs = dataset["sample_specs"]
    else:
        collector = DataCollector(
            n_samples=int(dataset["n_samples"]),
            n_jobs=args.n_jobs,
            seed=args.seed,
            layout_strategy=dataset["layout_strategy"],
        )
        specs = [spec.to_dict() for spec in collector._generate_lhs_samples()]

    check_rows = min(args.check_rows, len(specs), len(dataset["X"]))
    for i in range(check_rows):
        design = SampleSpec.from_dict(specs[i]).build_design()
        x = design_to_group_feature_vector(design)
        if not np.allclose(x, dataset["X"][i], rtol=args.tol, atol=args.tol):
            diff = float(np.max(np.abs(x - dataset["X"][i])))
            raise ValueError(f"Regenerated sample specs do not match dataset X at row {i}; max diff={diff}")

    rebuilt = dict(dataset)
    rebuilt["sample_specs"] = specs
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        pickle.dump(rebuilt, file)

    print(f"Attached sample_specs: {len(specs)}")
    print(f"Validated first {check_rows} rows against existing X.")
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()
