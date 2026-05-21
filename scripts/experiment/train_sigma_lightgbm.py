from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment.train_sigma_feature_experiment import build_features


@dataclass
class LGBMSigmaConfig:
    feature_set: str = "reduced_v2"
    n_estimators: int = 8000
    learning_rate: float = 0.025
    num_leaves: int = 96
    max_depth: int = -1
    min_child_samples: int = 80
    subsample: float = 0.9
    colsample_bytree: float = 0.85
    reg_alpha: float = 0.05
    reg_lambda: float = 1.0
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 42
    model_seed: int = 42
    early_stopping_rounds: int = 250


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def split_indices(n: int, config: LGBMSigmaConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.split_seed)
    indices = rng.permutation(n)
    n_test = max(1, int(n * config.test_ratio))
    n_val = max(1, int(n * config.val_ratio))
    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]
    return train_idx, val_idx, test_idx


def make_model(config: LGBMSigmaConfig) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        max_depth=config.max_depth,
        min_child_samples=config.min_child_samples,
        subsample=config.subsample,
        subsample_freq=1,
        colsample_bytree=config.colsample_bytree,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        random_state=config.model_seed,
        n_jobs=-1,
        verbosity=-1,
    )


def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: LGBMSigmaConfig,
) -> lgb.LGBMRegressor:
    model = make_model(config)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(config.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    return model


def importance_rows(model: lgb.LGBMRegressor, feature_names: list[str]) -> list[dict]:
    booster = model.booster_
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    gain_sum = float(gain.sum()) + 1e-12
    rows = [
        {
            "rank": rank,
            "feature": feature_names[idx],
            "gain": float(gain[idx]),
            "gain_pct": float(gain[idx] / gain_sum),
            "split": int(split[idx]),
        }
        for rank, idx in enumerate(np.argsort(gain)[::-1], start=1)
    ]
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def topk_experiments(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    ranked_features: list[dict],
    config: LGBMSigmaConfig,
    topk_values: list[int],
) -> list[dict]:
    rows = []
    for topk in topk_values:
        if topk >= X.shape[1]:
            continue
        cols = [int(row["_col"]) for row in ranked_features[:topk]]
        model = train_lgbm(X[train_idx][:, cols], y[train_idx], X[val_idx][:, cols], y[val_idx], config)
        val_pred = model.predict(X[val_idx][:, cols])
        test_pred = model.predict(X[test_idx][:, cols])
        rows.append(
            {
                "topk": int(topk),
                "best_iteration": int(model.best_iteration_ or config.n_estimators),
                "val": metrics(y[val_idx], val_pred),
                "test": metrics(y[test_idx], test_pred),
            }
        )
        print(
            f"Top-{topk}: val R2={rows[-1]['val']['r2']:.4f}, "
            f"test R2={rows[-1]['test']['r2']:.4f}, test MAE={rows[-1]['test']['mae']:.4f}"
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM u_sigma model and rank features.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--feature-set", default="reduced_v2", choices=["base", "reduced_v1", "reduced_v2", "reduced_v3"])
    parser.add_argument("--n-estimators", type=int, default=8000)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--num-leaves", type=int, default=96)
    parser.add_argument("--min-child-samples", type=int, default=80)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.85)
    parser.add_argument("--reg-alpha", type=float, default=0.05)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--topk", default="200,150,120,100,75,50")
    args = parser.parse_args()

    config = LGBMSigmaConfig(
        feature_set=args.feature_set,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
    )

    with open(args.dataset, "rb") as file:
        data = pickle.load(file)
    X, names = build_features(data["X"], data["feature_names"], config.feature_set, data.get("feature_format"))
    y = data["y_sigma"].astype(np.float32)
    train_idx, val_idx, test_idx = split_indices(len(X), config)

    print(
        f"Training LightGBM u_sigma: train={len(train_idx)}, val={len(val_idx)}, "
        f"test={len(test_idx)}, features={X.shape[1]}, feature_set={config.feature_set}"
    )
    model = train_lgbm(X[train_idx], y[train_idx], X[val_idx], y[val_idx], config)
    val_pred = model.predict(X[val_idx])
    test_pred = model.predict(X[test_idx])
    result = {"val": metrics(y[val_idx], val_pred), "test": metrics(y[test_idx], test_pred)}

    booster = model.booster_
    gain = booster.feature_importance(importance_type="gain")
    order = np.argsort(gain)[::-1]
    rows = importance_rows(model, names)
    name_to_col = {name: i for i, name in enumerate(names)}
    for row in rows:
        row["_col"] = int(name_to_col[row["feature"]])

    topk_values = [int(item) for item in args.topk.split(",") if item.strip()]
    topk_results = topk_experiments(X, y, train_idx, val_idx, test_idx, rows, config, topk_values)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategy": "sigma_lightgbm",
        "config": asdict(config),
        "dataset": args.dataset,
        "layout_strategy": data.get("layout_strategy"),
        "feature_names": names,
        "model": model,
        "metrics": result,
        "topk_results": topk_results,
    }
    with output.open("wb") as file:
        pickle.dump(payload, file)

    importance_path = output.with_name(output.stem + "_importance.csv")
    write_csv(importance_path, [{k: v for k, v in row.items() if k != "_col"} for row in rows])

    info_path = output.with_suffix(".json")
    with info_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "strategy": payload["strategy"],
                "config": payload["config"],
                "dataset": payload["dataset"],
                "layout_strategy": payload["layout_strategy"],
                "n_features": len(names),
                "best_iteration": int(model.best_iteration_ or config.n_estimators),
                "metrics": result,
                "top_features": [{k: v for k, v in row.items() if k != "_col"} for row in rows[:50]],
                "topk_results": topk_results,
            },
            file,
            indent=2,
        )

    print("\n=== LightGBM u_sigma Results ===")
    print(f"Val : MAE={result['val']['mae']:.4f}, RMSE={result['val']['rmse']:.4f}, R2={result['val']['r2']:.4f}")
    print(f"Test: MAE={result['test']['mae']:.4f}, RMSE={result['test']['rmse']:.4f}, R2={result['test']['r2']:.4f}")
    print("\nTop features:")
    for row in rows[:20]:
        print(f"  {row['rank']:2d}. {row['feature']}: gain_pct={row['gain_pct']:.4f}, split={row['split']}")
    print(f"Saved model to: {output}")
    print(f"Saved metrics to: {info_path}")
    print(f"Saved importance to: {importance_path}")


if __name__ == "__main__":
    main()
