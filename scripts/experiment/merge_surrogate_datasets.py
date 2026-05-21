from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Merge surrogate datasets with the same feature schema.")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    datasets = []
    for path in args.datasets:
        with open(path, "rb") as file:
            datasets.append(pickle.load(file))

    first = datasets[0]
    for dataset in datasets[1:]:
        if dataset["feature_format"] != first["feature_format"]:
            raise ValueError("All datasets must have the same feature_format.")
        if dataset["layout_strategy"] != first["layout_strategy"]:
            raise ValueError("All datasets must have the same layout_strategy.")
        if dataset["feature_names"] != first["feature_names"]:
            raise ValueError("All datasets must have identical feature_names.")

    merged = dict(first)
    for key in ["X", "y_sigma", "y_delta", "masses"]:
        merged[key] = np.concatenate([dataset[key] for dataset in datasets], axis=0)

    if all("sample_specs" in dataset for dataset in datasets):
        merged["sample_specs"] = [
            spec
            for dataset in datasets
            for spec in dataset["sample_specs"]
        ]

    merged["n_samples"] = int(len(merged["X"]))
    merged["source_datasets"] = args.datasets

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        pickle.dump(merged, file)

    print(f"Merged {len(datasets)} datasets: {merged['X'].shape}")
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()
