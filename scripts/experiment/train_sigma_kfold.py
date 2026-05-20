from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment.train_sigma_feature_experiment import (
    SigmaExperimentConfig,
    build_features,
    metrics,
    train_sigma_resnet,
)
from src.analysis.surrogate_model import KFoldConfig


def split_holdout(n: int, test_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_test = max(1, int(n * test_ratio))
    return indices[n_test:], indices[:n_test]


def predict(model, X: np.ndarray, y_mean: float, y_std: float, device: torch.device, batch_size: int) -> np.ndarray:
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[start : start + batch_size]).to(device)
            pred = model(batch).cpu().numpy() * y_std + y_mean
            preds.append(pred)
    return np.concatenate(preds)


def main():
    parser = argparse.ArgumentParser(description="Train u_sigma with true K-fold CV plus holdout test.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--feature-set", default="base", choices=["auto", "base", "stress_v1", "group_interactions"])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--resnet-blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=180)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--loss", default="smooth_l1", choices=["smooth_l1", "mse"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    with open(args.dataset, "rb") as file:
        data = pickle.load(file)

    config = SigmaExperimentConfig(
        feature_set=args.feature_set,
        hidden_dim=args.hidden_dim,
        resnet_blocks=args.resnet_blocks,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        test_ratio=args.test_ratio,
        seed=args.split_seed,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
        loss=args.loss,
    )

    X, feature_names = build_features(
        data["X"],
        data["feature_names"],
        config.feature_set,
        data.get("feature_format"),
    )
    y = data["y_sigma"].astype(np.float32)

    trainval_idx, test_idx = split_holdout(len(X), config.test_ratio, config.split_seed or config.seed)
    X_trainval = X[trainval_idx]
    y_trainval = y[trainval_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    X_mean = X_trainval.mean(axis=0)
    X_std = X_trainval.std(axis=0) + 1e-8
    y_mean = float(y_trainval.mean())
    y_std = float(y_trainval.std() + 1e-8)

    X_trainval_norm = (X_trainval - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std
    y_trainval_norm = (y_trainval - y_mean) / y_std

    kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.split_seed)
    models = []
    fold_metrics = []
    test_preds = []

    print(
        f"Training u_sigma {args.n_folds}-fold: trainval={len(trainval_idx)}, "
        f"test={len(test_idx)}, features={X.shape[1]}, device={device}"
    )

    for fold, (train_local, val_local) in enumerate(kfold.split(X_trainval_norm), start=1):
        print(f"\n--- Fold {fold}/{args.n_folds} ---")
        torch.manual_seed(args.model_seed + fold - 1)
        np.random.seed(args.model_seed + fold - 1)

        X_train = X_trainval_norm[train_local]
        y_train = y_trainval_norm[train_local]
        X_val = X_trainval_norm[val_local]
        y_val_norm = y_trainval_norm[val_local]
        y_val = y_trainval[val_local]

        model = train_sigma_resnet(X_train, y_train, X_val, y_val_norm, config, device)
        models.append(model)

        val_pred = predict(model, X_val, y_mean, y_std, device, args.batch_size)
        test_pred = predict(model, X_test_norm, y_mean, y_std, device, args.batch_size)
        test_preds.append(test_pred)
        fold_result = {
            "fold": fold,
            "val": metrics(y_val, val_pred),
            "test": metrics(y_test, test_pred),
        }
        fold_metrics.append(fold_result)
        print(
            f"Fold {fold}: val MAE={fold_result['val']['mae']:.4f}, "
            f"test MAE={fold_result['test']['mae']:.4f}, test R2={fold_result['test']['r2']:.4f}"
        )

    ensemble_pred = np.mean(np.vstack(test_preds), axis=0)
    test_result = metrics(y_test, ensemble_pred)

    payload = {
        "strategy": "sigma_kfold_resnet",
        "config": asdict(config),
        "dataset": args.dataset,
        "layout_strategy": data.get("layout_strategy"),
        "feature_names": feature_names,
        "model_states": [model.state_dict() for model in models],
        "X_mean": X_mean,
        "X_std": X_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "fold_metrics": fold_metrics,
        "metrics": {"test": test_result},
        "resnet_config": asdict(KFoldConfig(
            n_folds=args.n_folds,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            lr=config.lr,
            batch_size=config.batch_size,
            max_epochs=config.max_epochs,
            patience=config.patience,
            resnet_blocks=config.resnet_blocks,
        )),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        pickle.dump(payload, file)

    info = {
        "strategy": payload["strategy"],
        "config": payload["config"],
        "dataset": payload["dataset"],
        "layout_strategy": payload["layout_strategy"],
        "n_features": len(feature_names),
        "fold_metrics": fold_metrics,
        "metrics": payload["metrics"],
    }
    with output.with_suffix(".json").open("w", encoding="utf-8") as file:
        json.dump(info, file, indent=2)

    print("\n=== K-Fold Holdout Test ===")
    print(
        f"Ensemble: MAE={test_result['mae']:.4f}, RMSE={test_result['rmse']:.4f}, "
        f"R2={test_result['r2']:.4f}"
    )
    print(f"Saved model to: {output}")
    print(f"Saved metrics to: {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
