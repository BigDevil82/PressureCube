from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.surrogate_model import KFoldConfig, ResNetRegressor


EPS = 1e-8


@dataclass
class SigmaExperimentConfig:
    feature_set: str = "auto"
    hidden_dim: int = 384
    resnet_blocks: int = 4
    dropout: float = 0.08
    lr: float = 1e-3
    batch_size: int = 512
    max_epochs: int = 180
    patience: int = 25
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    split_seed: int | None = None
    model_seed: int | None = None
    weight_decay: float = 1e-4
    train_samples: int | None = None
    loss: str = "smooth_l1"


def _idx(feature_names: list[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(feature_names)}


def _col(X: np.ndarray, index: dict[str, int], name: str) -> np.ndarray:
    return X[:, index[name]]


def build_features(
    X: np.ndarray,
    feature_names: list[str],
    feature_set: str,
    feature_format: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    if feature_set == "auto":
        feature_set = "base" if str(feature_format).startswith("group_v") else "stress_v1"

    if feature_set == "base":
        return X.astype(np.float32), list(feature_names)
    if feature_set == "group_interactions":
        return build_group_interaction_features(X, feature_names)
    if feature_set != "stress_v1":
        raise ValueError(f"Unsupported feature_set: {feature_set}")

    index = _idx(feature_names)
    extras = []
    names = []

    def add(name: str, value: np.ndarray):
        extras.append(value.astype(np.float32))
        names.append(name)

    L = _col(X, index, "L")
    W = _col(X, index, "W")
    H = _col(X, index, "H")
    floor_area = _col(X, index, "floor_area")
    pressure = _col(X, index, "pressure_kPa")
    total_load = _col(X, index, "total_load_MN")
    spacing = _col(X, index, "main_beam_spacing")
    n_main = _col(X, index, "n_main_beams")

    areas = np.column_stack([_col(X, index, name) for name in [
        "frame_area_mean",
        "main_area_mean",
        "sec_area_mean",
        "brace_area_mean",
    ]])
    ixs = np.column_stack([_col(X, index, name) for name in [
        "frame_Ix_mean",
        "main_Ix_mean",
        "sec_Ix_mean",
        "brace_Ix_mean",
    ]])
    iys = np.column_stack([_col(X, index, name) for name in [
        "frame_Iy_mean",
        "main_Iy_mean",
        "sec_Iy_mean",
        "brace_Iy_mean",
    ]])
    type_names = ["frame", "main", "sec", "brace"]

    # Log transforms help the network separate small-section/high-demand tails.
    for base_name in [
        "volume",
        "floor_area",
        "total_load_MN",
        "total_area",
        "avg_Ix",
        "min_Ix",
        "stiffness_load_ratio",
        "strength_load_ratio",
    ]:
        add(f"log1p_{base_name}", np.log1p(np.maximum(_col(X, index, base_name), 0.0)))

    for i, type_name in enumerate(type_names):
        area = areas[:, i]
        ix = ixs[:, i]
        iy = iys[:, i]
        add(f"{type_name}_load_per_area", total_load / (area + EPS))
        add(f"{type_name}_load_per_Ix", total_load / (ix + EPS))
        add(f"{type_name}_load_per_Iy", total_load / (iy + EPS))
        add(f"{type_name}_pressure_area_demand", pressure * floor_area / (area + EPS))
        add(f"{type_name}_Ix_per_area", ix / (area + EPS))
        add(f"{type_name}_Iy_per_area", iy / (area + EPS))
        add(f"{type_name}_spacing_load_per_Ix", total_load * spacing / (ix + EPS))
        add(f"{type_name}_main_count_load_per_Ix", total_load * n_main / (ix + EPS))

    for stat_name, values in [("area", areas), ("Ix", ixs), ("Iy", iys)]:
        mean = values.mean(axis=1)
        std = values.std(axis=1)
        min_v = values.min(axis=1)
        max_v = values.max(axis=1)
        add(f"{stat_name}_std", std)
        add(f"{stat_name}_cv", std / (mean + EPS))
        add(f"{stat_name}_min_to_mean", min_v / (mean + EPS))
        add(f"{stat_name}_max_to_mean", max_v / (mean + EPS))
        add(f"{stat_name}_min_to_max", min_v / (max_v + EPS))

    pair_defs = [
        ("frame_main", 0, 1),
        ("main_sec", 1, 2),
        ("sec_brace", 2, 3),
        ("frame_sec", 0, 2),
    ]
    for pair_name, a, b in pair_defs:
        add(f"{pair_name}_area_ratio", areas[:, a] / (areas[:, b] + EPS))
        add(f"{pair_name}_Ix_ratio", ixs[:, a] / (ixs[:, b] + EPS))
        add(f"{pair_name}_Iy_ratio", iys[:, a] / (iys[:, b] + EPS))

    add("L_over_spacing", L / (spacing + EPS))
    add("W_over_spacing", W / (spacing + EPS))
    add("H_over_spacing", H / (spacing + EPS))
    add("pressure_times_L", pressure * L)
    add("pressure_times_W", pressure * W)
    add("pressure_times_H", pressure * H)
    add("load_times_aspect", total_load * (L / (W + EPS)))

    X_extra = np.column_stack(extras)
    X_aug = np.concatenate([X, X_extra], axis=1)
    X_aug = np.nan_to_num(X_aug, nan=0.0, posinf=1e8, neginf=-1e8)
    return X_aug.astype(np.float32), list(feature_names) + names


def build_group_interaction_features(
    X: np.ndarray,
    feature_names: list[str],
    max_groups_per_type: int = 3,
) -> tuple[np.ndarray, list[str]]:
    index = _idx(feature_names)
    extras = []
    names = []

    def add(name: str, value: np.ndarray):
        extras.append(value.astype(np.float32))
        names.append(name)

    def stack(prefix: str, suffix: str) -> np.ndarray:
        return np.column_stack([
            _col(X, index, f"{prefix}_g{i}_{suffix}")
            for i in range(1, max_groups_per_type + 1)
        ])

    for prefix in ["frame", "main", "sec", "brace"]:
        active = stack(prefix, "active") > 0.5
        weight = stack(prefix, "beam_count_norm")
        weight = np.where(active, weight, 0.0)
        weight_sum = weight.sum(axis=1, keepdims=True) + EPS

        area = stack(prefix, "area")
        ix = stack(prefix, "Ix")
        iy = stack(prefix, "Iy")
        z = stack(prefix, "z_mean")
        edge = stack(prefix, "edge_proximity_mean")
        corner = stack(prefix, "corner_proximity_mean")
        length = stack(prefix, "length_sum_norm")

        demand_names = [
            "load_per_area",
            "load_per_Ix",
            "pressure_area_demand",
            "length_per_area",
            "length_per_Ix",
            "pressure_length_per_area",
        ]
        for suffix in demand_names:
            values = np.where(active, stack(prefix, suffix), np.nan)
            add(f"{prefix}_{suffix}_max_group", np.nanmax(values, axis=1))
            add(f"{prefix}_{suffix}_std_group", np.nan_to_num(np.nanstd(values, axis=1), nan=0.0))

        for prop_name, prop in [("area", area), ("Ix", ix), ("Iy", iy)]:
            prop = np.where(active, prop, 0.0)
            mean = (prop * weight).sum(axis=1) / weight_sum[:, 0]
            var = (((prop - mean[:, None]) ** 2) * weight).sum(axis=1) / weight_sum[:, 0]
            std = np.sqrt(np.maximum(var, 0.0))
            add(f"{prefix}_{prop_name}_beam_weighted_mean", mean)
            add(f"{prefix}_{prop_name}_beam_weighted_cv", std / (mean + EPS))

        weakest_idx = np.nanargmin(np.where(active, area, np.nan), axis=1)
        strongest_idx = np.nanargmax(np.where(active, area, np.nan), axis=1)
        rows = np.arange(len(X))
        for label, idxs in [("weakest", weakest_idx), ("strongest", strongest_idx)]:
            add(f"{prefix}_{label}_beam_count_norm", weight[rows, idxs])
            add(f"{prefix}_{label}_length_sum_norm", length[rows, idxs])
            add(f"{prefix}_{label}_z_mean", z[rows, idxs])
            add(f"{prefix}_{label}_edge_proximity_mean", edge[rows, idxs])
            add(f"{prefix}_{label}_corner_proximity_mean", corner[rows, idxs])

        ix_mean = (ix * weight).sum(axis=1) / weight_sum[:, 0]
        for geo_name, geo in [("z", z), ("edge", edge), ("corner", corner), ("length", length)]:
            geo_mean = (geo * weight).sum(axis=1) / weight_sum[:, 0]
            cov = ((ix - ix_mean[:, None]) * (geo - geo_mean[:, None]) * weight).sum(axis=1) / weight_sum[:, 0]
            add(f"{prefix}_Ix_{geo_name}_weighted_cov", cov)

    X_extra = np.column_stack(extras)
    X_aug = np.concatenate([X, X_extra], axis=1)
    X_aug = np.nan_to_num(X_aug, nan=0.0, posinf=1e8, neginf=-1e8)
    return X_aug.astype(np.float32), list(feature_names) + names


def train_sigma_resnet(
    X_train: np.ndarray,
    y_train_norm: np.ndarray,
    X_val: np.ndarray,
    y_val_norm: np.ndarray,
    config: SigmaExperimentConfig,
    device: torch.device,
) -> ResNetRegressor:
    model = ResNetRegressor(
        X_train.shape[1],
        hidden_dim=config.hidden_dim,
        n_blocks=config.resnet_blocks,
        dropout=config.dropout,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)
    criterion = torch.nn.MSELoss() if config.loss == "mse" else torch.nn.SmoothL1Loss()

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train_norm)),
        batch_size=config.batch_size,
        shuffle=True,
    )
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val_norm).to(device)

    best_state = None
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(X_val_t), y_val_t).item())
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}: train_loss={np.mean(train_losses):.5f}, "
                f"val_loss={val_loss:.5f}, best={best_val_loss:.5f}"
            )

        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    return model


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    parser = argparse.ArgumentParser(description="Single-model u_sigma feature experiment.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--feature-set", default="auto", choices=["auto", "base", "stress_v1", "group_interactions"])
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--resnet-blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=180)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--model-seed", type=int, default=None)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--loss", default="smooth_l1", choices=["smooth_l1", "mse"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = SigmaExperimentConfig(
        feature_set=args.feature_set,
        hidden_dim=args.hidden_dim,
        resnet_blocks=args.resnet_blocks,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
        train_samples=args.train_samples,
        loss=args.loss,
    )

    split_seed = config.split_seed if config.split_seed is not None else config.seed
    model_seed = config.model_seed if config.model_seed is not None else config.seed

    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    with open(args.dataset, "rb") as file:
        data = pickle.load(file)

    X, feature_names = build_features(
        data["X"],
        data["feature_names"],
        config.feature_set,
        data.get("feature_format"),
    )
    y = data["y_sigma"].astype(np.float32)

    n = len(X)
    rng = np.random.default_rng(split_seed)
    indices = rng.permutation(n)
    n_test = max(1, int(n * config.test_ratio))
    n_val = max(1, int(n * config.val_ratio))
    if n_test + n_val >= n:
        n_test = max(1, n // 5)
        n_val = max(1, n // 5)
    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]
    if config.train_samples is not None and config.train_samples < len(train_idx):
        train_idx = rng.choice(train_idx, size=config.train_samples, replace=False)

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8
    y_mean = float(y_train.mean())
    y_std = float(y_train.std() + 1e-8)

    X_train_norm = (X_train - X_mean) / X_std
    X_val_norm = (X_val - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std
    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm = (y_val - y_mean) / y_std

    print(
        f"Training u_sigma only: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}, "
        f"features={X.shape[1]}, feature_set={config.feature_set}, device={device}"
    )

    model = train_sigma_resnet(X_train_norm, y_train_norm, X_val_norm, y_val_norm, config, device)
    model.eval()
    with torch.no_grad():
        val_pred = model(torch.FloatTensor(X_val_norm).to(device)).cpu().numpy() * y_std + y_mean
        test_pred = model(torch.FloatTensor(X_test_norm).to(device)).cpu().numpy() * y_std + y_mean

    result = {
        "val": metrics(y_val, val_pred),
        "test": metrics(y_test, test_pred),
    }
    print("\n=== u_sigma Results ===")
    print(
        f"Val : MAE={result['val']['mae']:.4f}, RMSE={result['val']['rmse']:.4f}, R2={result['val']['r2']:.4f}"
    )
    print(
        f"Test: MAE={result['test']['mae']:.4f}, RMSE={result['test']['rmse']:.4f}, R2={result['test']['r2']:.4f}"
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategy": "sigma_feature_experiment_resnet",
        "config": asdict(config),
        "dataset": args.dataset,
        "layout_strategy": data.get("layout_strategy"),
        "feature_names": feature_names,
        "model_state": model.state_dict(),
        "X_mean": X_mean,
        "X_std": X_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "metrics": result,
        "resnet_config": asdict(KFoldConfig(
            n_folds=1,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            lr=config.lr,
            batch_size=config.batch_size,
            max_epochs=config.max_epochs,
            patience=config.patience,
            resnet_blocks=config.resnet_blocks,
        )),
    }
    with output.open("wb") as file:
        pickle.dump(payload, file)

    info_path = output.with_suffix(".json")
    with info_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "strategy": payload["strategy"],
                "config": payload["config"],
                "dataset": payload["dataset"],
                "layout_strategy": payload["layout_strategy"],
                "n_features": len(feature_names),
                "metrics": payload["metrics"],
            },
            file,
            indent=2,
        )
    print(f"Saved model to: {output}")
    print(f"Saved metrics to: {info_path}")


if __name__ == "__main__":
    main()
