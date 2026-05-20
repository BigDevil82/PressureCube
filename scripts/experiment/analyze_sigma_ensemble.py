from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment.analyze_sigma_errors import regression_metrics, split_indices
from scripts.experiment.train_sigma_feature_experiment import build_features
from src.analysis.surrogate_model import ResNetRegressor


def predict_model(dataset: dict, model_path: Path, device: torch.device, batch_size: int) -> np.ndarray:
    with model_path.open("rb") as file:
        payload = pickle.load(file)

    config = payload.get("config", {})
    feature_set = config.get("feature_set", "auto")
    if feature_set == "auto" and len(payload.get("feature_names", [])) == len(dataset["feature_names"]):
        feature_set = "base"
    X, _ = build_features(
        dataset["X"],
        dataset["feature_names"],
        feature_set,
        dataset.get("feature_format"),
    )
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate averaged u_sigma model predictions.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test", choices=["test", "val", "train", "all"])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    with open(args.dataset, "rb") as file:
        dataset = pickle.load(file)

    preds = [
        predict_model(dataset, Path(model_path), device, args.batch_size)
        for model_path in args.models
    ]
    y = dataset["y_sigma"].astype(np.float32)
    split_config = {"split_seed": args.split_seed, "seed": args.split_seed, "val_ratio": 0.1, "test_ratio": 0.1}
    keep = split_indices(len(y), split_config, args.split)

    rows = {}
    for model_path, pred in zip(args.models, preds):
        rows[Path(model_path).stem] = regression_metrics(y[keep], pred[keep])
    ensemble_pred = np.mean(np.vstack(preds), axis=0)
    rows["ensemble"] = regression_metrics(y[keep], ensemble_pred[keep])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset": args.dataset,
                "models": args.models,
                "split": args.split,
                "split_seed": args.split_seed,
                "metrics": rows,
            },
            file,
            indent=2,
        )

    for name, item in rows.items():
        print(f"{name}: MAE={item['mae']:.4f}, RMSE={item['rmse']:.4f}, R2={item['r2']:.4f}")
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()
