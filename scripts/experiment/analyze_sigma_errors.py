from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment.train_sigma_feature_experiment import build_features
from src.analysis.surrogate_model import ResNetRegressor


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)) if len(y) > 1 else float("nan"),
        "bias": float(np.mean(pred - y)),
        "p95_abs_err": float(np.quantile(np.abs(pred - y), 0.95)),
    }


def load_predictions(dataset_path: Path, model_path: Path, device: torch.device, batch_size: int, target: str | None = None):
    with dataset_path.open("rb") as file:
        data = pickle.load(file)
    with model_path.open("rb") as file:
        payload = pickle.load(file)

    config = payload.get("config", {})
    X, feature_names = build_features(
        data["X"],
        data["feature_names"],
        config.get("feature_set", "auto"),
        data.get("feature_format"),
    )
    target = target or payload.get("config", {}).get("target", "sigma")
    y_key = "y_sigma" if target == "sigma" else "y_delta"
    y = data[y_key].astype(np.float32)

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
            if payload.get("config", {}).get("target_transform") == "log1p":
                pred = np.expm1(pred)
            preds.append(pred)

    return X, y, np.concatenate(preds), feature_names, payload


def split_indices(n: int, config: dict, split: str) -> np.ndarray:
    if split == "all":
        return np.arange(n)

    rng = np.random.default_rng(config.get("split_seed") or config.get("seed", 42))
    indices = rng.permutation(n)
    n_test = max(1, int(n * config.get("test_ratio", 0.1)))
    n_val = max(1, int(n * config.get("val_ratio", 0.1)))
    if n_test + n_val >= n:
        n_test = max(1, n // 5)
        n_val = max(1, n // 5)

    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]
    train_samples = config.get("train_samples")
    if train_samples is not None and train_samples < len(train_idx):
        train_idx = rng.choice(train_idx, size=train_samples, replace=False)

    if split == "test":
        return test_idx
    if split == "val":
        return val_idx
    if split == "train":
        return train_idx
    raise ValueError(f"Unsupported split: {split}")


def quantile_slices(
    name: str,
    values: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
    n_bins: int,
) -> list[dict]:
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) <= 2:
        return []

    rows = []
    for i in range(len(edges) - 1):
        if i == len(edges) - 2:
            mask = (values >= edges[i]) & (values <= edges[i + 1])
        else:
            mask = (values >= edges[i]) & (values < edges[i + 1])
        if mask.sum() < 2:
            continue
        item = regression_metrics(y[mask], pred[mask])
        item["feature"] = name
        item["range"] = [float(edges[i]), float(edges[i + 1])]
        rows.append(item)
    return rows


def categorical_slices(
    name: str,
    values: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
) -> list[dict]:
    rows = []
    for value in sorted(np.unique(values)):
        mask = values == value
        if mask.sum() < 2:
            continue
        item = regression_metrics(y[mask], pred[mask])
        item["feature"] = name
        item["value"] = float(value)
        rows.append(item)
    return rows


def top_correlations(
    X: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
    feature_names: list[str],
    top_k: int,
) -> dict[str, list[dict]]:
    residual = pred - y
    abs_err = np.abs(residual)
    out = {"abs_error": [], "signed_error": []}
    for target_name, target in [("abs_error", abs_err), ("signed_error", residual)]:
        rows = []
        for i, name in enumerate(feature_names):
            x = X[:, i]
            if np.std(x) < 1e-12:
                continue
            corr = float(np.corrcoef(x, target)[0, 1])
            if np.isfinite(corr):
                rows.append({"feature": name, "corr": corr, "abs_corr": abs(corr)})
        rows.sort(key=lambda item: item["abs_corr"], reverse=True)
        out[target_name] = rows[:top_k]
    return out


def main():
    parser = argparse.ArgumentParser(description="Analyze surrogate residuals without retraining.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--n-bins", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--split", default="test", choices=["test", "val", "train", "all"])
    parser.add_argument("--target", default=None, choices=["sigma", "delta"])
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    X, y, pred, feature_names, payload = load_predictions(
        Path(args.dataset),
        Path(args.model),
        device,
        args.batch_size,
        args.target,
    )
    keep = split_indices(len(X), payload.get("config", {}), args.split)
    X = X[keep]
    y = y[keep]
    pred = pred[keep]
    index = {name: i for i, name in enumerate(feature_names)}

    slice_features = [
        "L",
        "W",
        "H",
        "pressure_kPa",
        "main_beam_spacing",
        "total_load_MN",
        "frame_n_groups",
        "main_n_groups",
        "sec_n_groups",
        "brace_n_groups",
    ]
    categorical_features = [
        name for name in slice_features if name.endswith("_n_groups") or name == "add_support_beam"
    ]

    slices = []
    for name in slice_features:
        if name not in index:
            continue
        values = X[:, index[name]]
        if name in categorical_features:
            slices.extend(categorical_slices(name, values, y, pred))
        else:
            slices.extend(quantile_slices(name, values, y, pred, args.n_bins))

    group_cols = [index[name] for name in feature_names if name.endswith("_n_groups")]
    if group_cols:
        total_groups = X[:, group_cols].sum(axis=1)
        slices.extend(categorical_slices("total_n_groups", total_groups, y, pred))

    worst_idx = np.argsort(np.abs(pred - y))[-args.top_k :][::-1]
    worst = [
        {
            "index": int(i),
            "y": float(y[i]),
            "pred": float(pred[i]),
            "error": float(pred[i] - y[i]),
            "abs_error": float(abs(pred[i] - y[i])),
        }
        for i in worst_idx
    ]

    report = {
        "split": args.split,
        "overall": regression_metrics(y, pred),
        "slices": sorted(slices, key=lambda item: item["mae"], reverse=True),
        "feature_error_correlations": top_correlations(X, y, pred, feature_names, args.top_k),
        "worst_samples": worst,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("Overall:", json.dumps(report["overall"], ensure_ascii=False))
    print("\nWorst slices by MAE:")
    for row in report["slices"][:10]:
        key = row.get("range", row.get("value"))
        print(f"  {row['feature']}={key}: n={row['n']}, mae={row['mae']:.4f}, r2={row['r2']:.4f}, bias={row['bias']:.4f}")
    print(f"\nSaved report to: {output}")


if __name__ == "__main__":
    main()
