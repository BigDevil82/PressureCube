from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment.analyze_sigma_errors import split_indices
from scripts.experiment.train_sigma_feature_experiment import build_features
from src.analysis.surrogate_model import ResNetRegressor


GEOM_SUFFIXES = {
    "beam_count",
    "beam_count_norm",
    "length_sum_norm",
    "length_mean_norm",
    "length_max_norm",
    "x_mean",
    "y_mean",
    "z_mean",
    "x_std",
    "y_std",
    "z_std",
    "edge_proximity_mean",
    "edge_proximity_min",
    "corner_proximity_mean",
    "corner_proximity_min",
    "ori_x_frac",
    "ori_y_frac",
    "ori_z_frac",
    "face_horizontal_frac",
    "face_longitudinal_frac",
    "face_end_frac",
    "face_other_frac",
    "length_per_area",
    "length_per_Ix",
    "pressure_length_per_area",
}


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
    }


def load_model_payload(model_path: Path) -> dict:
    with model_path.open("rb") as file:
        return pickle.load(file)


def prepare_features(dataset: dict, payload: dict) -> tuple[np.ndarray, list[str]]:
    config = payload.get("config", {})
    feature_set = config.get("feature_set", "auto")
    if feature_set == "auto" and len(payload.get("feature_names", [])) == len(dataset["feature_names"]):
        feature_set = "base"
    return build_features(
        dataset["X"],
        dataset["feature_names"],
        feature_set,
        dataset.get("feature_format"),
    )


def predict_from_payload(
    X: np.ndarray,
    payload: dict,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    config = payload.get("config", {})
    model_cfg = payload.get("resnet_config", {})
    model = ResNetRegressor(
        X.shape[1],
        hidden_dim=model_cfg.get("hidden_dim", config.get("hidden_dim", 384)),
        n_blocks=model_cfg.get("resnet_blocks", config.get("resnet_blocks", 4)),
        dropout=model_cfg.get("dropout", config.get("dropout", 0.08)),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    X_norm = (X - payload["X_mean"]) / payload["X_std"]
    preds = []
    with torch.no_grad():
        for start in range(0, len(X_norm), batch_size):
            batch = torch.FloatTensor(X_norm[start : start + batch_size]).to(device)
            pred = model(batch).cpu().numpy() * payload["y_std"] + payload["y_mean"]
            preds.append(pred)
    return np.concatenate(preds)


def feature_groups(feature_names: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {
        "global_geometry_load": _indices(
            feature_names,
            ["L", "W", "H", "volume", "floor_area", "aspect_ratio", "pressure_kPa", "total_load_MN"],
        ),
        "topology": _indices(
            feature_names,
            ["main_beam_spacing", "support_beam_max_spacing", "add_support_beam", "n_main_beams"],
        ),
    }

    for prefix in ["frame", "main", "sec", "brace"]:
        count_cols = [i for i, name in enumerate(feature_names) if name in {f"{prefix}_n_groups", f"{prefix}_n_groups_norm"}]
        slot_section_cols = []
        slot_geom_cols = []
        stat_cols = []

        for i, name in enumerate(feature_names):
            if name.startswith((f"{prefix}_area_", f"{prefix}_Ix_", f"{prefix}_Iy_", f"{prefix}_weakest_")):
                stat_cols.append(i)
                continue

            match = re.match(rf"^{prefix}_g\d+_(.+)$", name)
            if not match:
                continue
            suffix = match.group(1)
            if suffix in GEOM_SUFFIXES:
                slot_geom_cols.append(i)
            else:
                slot_section_cols.append(i)

        groups[f"{prefix}_group_count"] = count_cols
        groups[f"{prefix}_section_slots"] = slot_section_cols
        groups[f"{prefix}_section_stats"] = stat_cols
        groups[f"{prefix}_group_geometry"] = slot_geom_cols

    return {name: cols for name, cols in groups.items() if cols}


def _indices(feature_names: list[str], names: list[str]) -> list[int]:
    index = {name: i for i, name in enumerate(feature_names)}
    return [index[name] for name in names if name in index]


def grouped_permutation(
    X: np.ndarray,
    y: np.ndarray,
    baseline_pred: np.ndarray,
    payload: dict,
    groups: dict[str, list[int]],
    device: torch.device,
    batch_size: int,
    rng: np.random.Generator,
    repeats: int,
) -> list[dict]:
    base = metrics(y, baseline_pred)
    rows = []
    for group_name, cols in groups.items():
        rmse_values = []
        mae_values = []
        for _ in range(repeats):
            X_perm = X.copy()
            order = rng.permutation(len(X_perm))
            X_perm[:, cols] = X_perm[order][:, cols]
            pred = predict_from_payload(X_perm, payload, device, batch_size)
            item = metrics(y, pred)
            rmse_values.append(item["rmse"])
            mae_values.append(item["mae"])
        rows.append(
            {
                "group": group_name,
                "n_features": len(cols),
                "perm_rmse": float(np.mean(rmse_values)),
                "perm_rmse_delta": float(np.mean(rmse_values) - base["rmse"]),
                "perm_mae": float(np.mean(mae_values)),
                "perm_mae_delta": float(np.mean(mae_values) - base["mae"]),
            }
        )
    return sorted(rows, key=lambda item: item["perm_rmse_delta"], reverse=True)


def individual_permutation(
    X: np.ndarray,
    y: np.ndarray,
    baseline_pred: np.ndarray,
    payload: dict,
    feature_names: list[str],
    candidate_cols: list[int],
    device: torch.device,
    batch_size: int,
    rng: np.random.Generator,
) -> list[dict]:
    base = metrics(y, baseline_pred)
    rows = []
    for col in candidate_cols:
        X_perm = X.copy()
        X_perm[:, col] = X_perm[rng.permutation(len(X_perm)), col]
        pred = predict_from_payload(X_perm, payload, device, batch_size)
        item = metrics(y, pred)
        rows.append(
            {
                "feature": feature_names[col],
                "perm_rmse_delta": float(item["rmse"] - base["rmse"]),
                "perm_mae_delta": float(item["mae"] - base["mae"]),
            }
        )
    return sorted(rows, key=lambda item: item["perm_rmse_delta"], reverse=True)


def pick_freeze_sobol_morris(
    X: np.ndarray,
    payload: dict,
    groups: dict[str, list[int]],
    device: torch.device,
    batch_size: int,
    rng: np.random.Generator,
    n_pairs: int,
) -> list[dict]:
    a_idx = rng.integers(0, len(X), size=n_pairs)
    b_idx = rng.integers(0, len(X), size=n_pairs)
    XA = X[a_idx]
    XB = X[b_idx]
    fA = predict_from_payload(XA, payload, device, batch_size)
    fB = predict_from_payload(XB, payload, device, batch_size)
    variance = float(np.var(np.concatenate([fA, fB]), ddof=1)) + 1e-12

    rows = []
    for group_name, cols in groups.items():
        X_swap = XA.copy()
        X_swap[:, cols] = XB[:, cols]
        f_swap = predict_from_payload(X_swap, payload, device, batch_size)
        diff = f_swap - fA
        total_index = float(np.mean(diff ** 2) / (2.0 * variance))
        first_index = float(np.mean(fB * (f_swap - fA)) / variance)
        rows.append(
            {
                "group": group_name,
                "n_features": len(cols),
                "sobol_first_pick_freeze": first_index,
                "sobol_total_pick_freeze": total_index,
                "morris_mu": float(np.mean(diff)),
                "morris_mu_star": float(np.mean(np.abs(diff))),
                "morris_sigma": float(np.std(diff)),
            }
        )
    return sorted(rows, key=lambda item: item["sobol_total_pick_freeze"], reverse=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Grouped sensitivity analysis for sigma surrogate features.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test", choices=["test", "val", "train", "all"])
    parser.add_argument("--max-rows", type=int, default=12000)
    parser.add_argument("--sobol-pairs", type=int, default=6000)
    parser.add_argument("--permutation-repeats", type=int, default=3)
    parser.add_argument("--individual-top-groups", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    with open(args.dataset, "rb") as file:
        dataset = pickle.load(file)
    payload = load_model_payload(Path(args.model))
    X_all, names = prepare_features(dataset, payload)
    y_all = dataset["y_sigma"].astype(np.float32)

    keep = split_indices(len(X_all), payload.get("config", {}), args.split)
    rng = np.random.default_rng(args.seed)
    if args.max_rows and len(keep) > args.max_rows:
        keep = rng.choice(keep, size=args.max_rows, replace=False)

    X = X_all[keep]
    y = y_all[keep]
    groups = feature_groups(names)
    baseline_pred = predict_from_payload(X, payload, device, args.batch_size)
    baseline = metrics(y, baseline_pred)

    permutation_rows = grouped_permutation(
        X, y, baseline_pred, payload, groups, device, args.batch_size, rng, args.permutation_repeats
    )
    sobol_morris_rows = pick_freeze_sobol_morris(
        X, payload, groups, device, args.batch_size, rng, min(args.sobol_pairs, len(X))
    )
    top_group_names = [row["group"] for row in permutation_rows[: args.individual_top_groups]]
    candidate_cols = sorted({col for name in top_group_names for col in groups[name]})
    individual_rows = individual_permutation(
        X, y, baseline_pred, payload, names, candidate_cols, device, args.batch_size, rng
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": args.dataset,
        "model": args.model,
        "split": args.split,
        "n_rows": int(len(X)),
        "baseline": baseline,
        "notes": [
            "Features are strongly correlated engineered variables.",
            "Sobol/Morris results are grouped empirical pick-freeze approximations, not strict independent-variable variance decomposition.",
            "Use permutation for model dependence and pick-freeze Sobol/Morris for grouped sensitivity screening.",
        ],
        "groups": {name: len(cols) for name, cols in groups.items()},
        "permutation": permutation_rows,
        "sobol_morris": sobol_morris_rows,
        "individual_permutation": individual_rows,
    }
    with output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    write_csv(output.with_name(output.stem + "_permutation.csv"), permutation_rows)
    write_csv(output.with_name(output.stem + "_sobol_morris.csv"), sobol_morris_rows)
    write_csv(output.with_name(output.stem + "_individual.csv"), individual_rows)

    print("Baseline:", baseline)
    print("\nTop permutation groups:")
    for row in permutation_rows[:10]:
        print(f"  {row['group']}: dRMSE={row['perm_rmse_delta']:.4f}, dMAE={row['perm_mae_delta']:.4f}")
    print("\nTop Sobol/Morris groups:")
    for row in sobol_morris_rows[:10]:
        print(
            f"  {row['group']}: ST={row['sobol_total_pick_freeze']:.4f}, "
            f"mu*={row['morris_mu_star']:.4f}, sigma={row['morris_sigma']:.4f}"
        )
    print("\nTop individual features within important groups:")
    for row in individual_rows[:15]:
        print(f"  {row['feature']}: dRMSE={row['perm_rmse_delta']:.4f}, dMAE={row['perm_mae_delta']:.4f}")
    print(f"\nSaved report to: {output}")


if __name__ == "__main__":
    main()
