"""
Advanced surrogate model training with a single strategy: K-Fold + ResNet.
"""

import pickle
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

from analysis.surrogate_data import DesignConfig, SectionLibrary, compute_mass_fast, detect_feature_format


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class ResNetRegressor(nn.Module):
    """ResNet-style regressor with skip connections."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_blocks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.output(x).squeeze(-1)


@dataclass
class KFoldConfig:
    """Configuration for K-Fold + ResNet training."""

    n_folds: int = 5
    hidden_dim: int = 256
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 20
    resnet_blocks: int = 3


class KFoldTrainer:
    """K-Fold trainer with ResNet ensemble prediction."""

    def __init__(self, config: KFoldConfig, device: str = "auto"):
        self.config = config

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.models_sigma: List[nn.Module] = []
        self.models_delta: List[nn.Module] = []

        self.X_mean: Optional[np.ndarray] = None
        self.X_std: Optional[np.ndarray] = None
        self.y_sigma_mean: float = 0.0
        self.y_sigma_std: float = 1.0
        self.y_delta_mean: float = 0.0
        self.y_delta_std: float = 1.0

        self.fold_metrics: List[Dict] = []

    def _create_model(self, input_dim: int) -> nn.Module:
        return ResNetRegressor(
            input_dim,
            hidden_dim=self.config.hidden_dim,
            n_blocks=self.config.resnet_blocks,
            dropout=self.config.dropout,
        )

    def _normalize_X(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.X_mean = X.mean(axis=0)
            self.X_std = X.std(axis=0) + 1e-8
        return (X - self.X_mean) / self.X_std

    def _normalize_y(self, y: np.ndarray, target: str, fit: bool = False) -> np.ndarray:
        if fit:
            if target == "sigma":
                self.y_sigma_mean = float(y.mean())
                self.y_sigma_std = float(y.std()) + 1e-8
            else:
                self.y_delta_mean = float(y.mean())
                self.y_delta_std = float(y.std()) + 1e-8

        if target == "sigma":
            return (y - self.y_sigma_mean) / self.y_sigma_std
        return (y - self.y_delta_mean) / self.y_delta_std

    def _denormalize_y(self, y: np.ndarray, target: str) -> np.ndarray:
        if target == "sigma":
            return y * self.y_sigma_std + self.y_sigma_mean
        return y * self.y_delta_std + self.y_delta_mean

    def _train_single_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> nn.Module:
        input_dim = X_train.shape[1]
        model = self._create_model(input_dim).to(self.device)

        optimizer = optim.AdamW(model.parameters(), lr=self.config.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
        criterion = nn.SmoothL1Loss()

        X_train_norm = self._normalize_X(X_train)
        X_val_norm = self._normalize_X(X_val)

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_norm),
            torch.FloatTensor(y_train),
        )
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)

        X_val_t = torch.FloatTensor(X_val_norm).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for _ in range(self.config.max_epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.patience:
                break

        model.load_state_dict(best_state)
        return model

    def fit(self, X: np.ndarray, y_sigma: np.ndarray, y_delta: np.ndarray):
        """Train with K-Fold cross validation."""
        print(f"Training with {self.config.n_folds}-Fold ResNet CV on {self.device}...")

        self._normalize_X(X, fit=True)
        self._normalize_y(y_sigma, "sigma", fit=True)
        self._normalize_y(y_delta, "delta", fit=True)

        kfold = KFold(n_splits=self.config.n_folds, shuffle=True, random_state=42)

        self.models_sigma = []
        self.models_delta = []
        self.fold_metrics = []

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
            print(f"\n--- Fold {fold_idx + 1}/{self.config.n_folds} ---")

            X_train, X_val = X[train_idx], X[val_idx]
            y_sigma_train, y_sigma_val = y_sigma[train_idx], y_sigma[val_idx]
            y_delta_train, y_delta_val = y_delta[train_idx], y_delta[val_idx]

            print("  Training u_sigma...")
            y_sigma_train_norm = self._normalize_y(y_sigma_train, "sigma")
            y_sigma_val_norm = self._normalize_y(y_sigma_val, "sigma")
            model_sigma = self._train_single_fold(X_train, y_sigma_train_norm, X_val, y_sigma_val_norm)
            self.models_sigma.append(model_sigma)

            print("  Training u_delta...")
            y_delta_train_norm = self._normalize_y(y_delta_train, "delta")
            y_delta_val_norm = self._normalize_y(y_delta_val, "delta")
            model_delta = self._train_single_fold(X_train, y_delta_train_norm, X_val, y_delta_val_norm)
            self.models_delta.append(model_delta)

            pred_sigma, pred_delta = self._predict_single_fold(X_val, fold_idx)
            fold_metrics = self._compute_metrics(y_sigma_val, pred_sigma, y_delta_val, pred_delta)
            self.fold_metrics.append(fold_metrics)

            print(
                f"  Fold {fold_idx + 1} - sigma MAE: {fold_metrics['sigma_mae']:.4f}, "
                f"delta MAE: {fold_metrics['delta_mae']:.4f}"
            )

        avg_sigma_mae = np.mean([m["sigma_mae"] for m in self.fold_metrics])
        avg_delta_mae = np.mean([m["delta_mae"] for m in self.fold_metrics])
        std_sigma_mae = np.std([m["sigma_mae"] for m in self.fold_metrics])
        std_delta_mae = np.std([m["delta_mae"] for m in self.fold_metrics])

        print(f"\n=== K-Fold CV Results ===")
        print(f"  u_sigma MAE: {avg_sigma_mae:.4f} ± {std_sigma_mae:.4f}")
        print(f"  u_delta MAE: {avg_delta_mae:.4f} ± {std_delta_mae:.4f}")

    def _predict_single_fold(self, X: np.ndarray, fold_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        X_norm = self._normalize_X(X)
        X_t = torch.FloatTensor(X_norm).to(self.device)

        self.models_sigma[fold_idx].eval()
        self.models_delta[fold_idx].eval()

        with torch.no_grad():
            pred_sigma = self.models_sigma[fold_idx](X_t).cpu().numpy()
            pred_delta = self.models_delta[fold_idx](X_t).cpu().numpy()

        pred_sigma = self._denormalize_y(pred_sigma, "sigma")
        pred_delta = self._denormalize_y(pred_delta, "delta")
        return pred_sigma, pred_delta

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Ensemble prediction: average of all fold models."""
        X_norm = self._normalize_X(X)
        X_t = torch.FloatTensor(X_norm).to(self.device)

        all_sigma = []
        all_delta = []

        for model_sigma, model_delta in zip(self.models_sigma, self.models_delta):
            model_sigma.eval()
            model_delta.eval()

            with torch.no_grad():
                pred_sigma = model_sigma(X_t).cpu().numpy()
                pred_delta = model_delta(X_t).cpu().numpy()

            all_sigma.append(pred_sigma)
            all_delta.append(pred_delta)

        ensemble_sigma = np.mean(all_sigma, axis=0)
        ensemble_delta = np.mean(all_delta, axis=0)

        ensemble_sigma = self._denormalize_y(ensemble_sigma, "sigma")
        ensemble_delta = self._denormalize_y(ensemble_delta, "delta")
        return ensemble_sigma, ensemble_delta

    def _compute_metrics(
        self,
        y_sigma_true: np.ndarray,
        y_sigma_pred: np.ndarray,
        y_delta_true: np.ndarray,
        y_delta_pred: np.ndarray,
    ) -> Dict:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        return {
            "sigma_mae": mean_absolute_error(y_sigma_true, y_sigma_pred),
            "sigma_rmse": np.sqrt(mean_squared_error(y_sigma_true, y_sigma_pred)),
            "sigma_r2": r2_score(y_sigma_true, y_sigma_pred),
            "delta_mae": mean_absolute_error(y_delta_true, y_delta_pred),
            "delta_rmse": np.sqrt(mean_squared_error(y_delta_true, y_delta_pred)),
            "delta_r2": r2_score(y_delta_true, y_delta_pred),
        }

    def get_state(self) -> Dict:
        """Get state for serialization."""
        return {
            "config": {
                "n_folds": self.config.n_folds,
                "hidden_dim": self.config.hidden_dim,
                "dropout": self.config.dropout,
                "resnet_blocks": self.config.resnet_blocks,
            },
            "models_sigma_state": [m.state_dict() for m in self.models_sigma],
            "models_delta_state": [m.state_dict() for m in self.models_delta],
            "X_mean": self.X_mean,
            "X_std": self.X_std,
            "y_sigma_mean": self.y_sigma_mean,
            "y_sigma_std": self.y_sigma_std,
            "y_delta_mean": self.y_delta_mean,
            "y_delta_std": self.y_delta_std,
            "input_dim": self.models_sigma[0].input_proj[0].in_features,
            "fold_metrics": self.fold_metrics,
        }

    def load_state(self, state: Dict):
        """Load from serialized state."""
        cfg = state["config"]
        hidden_dim = cfg.get("hidden_dim")
        if hidden_dim is None:
            hidden_dims = cfg.get("hidden_dims", [256])
            hidden_dim = hidden_dims[0]

        self.config = KFoldConfig(
            n_folds=cfg["n_folds"],
            hidden_dim=hidden_dim,
            dropout=cfg["dropout"],
            resnet_blocks=cfg["resnet_blocks"],
        )

        self.X_mean = state["X_mean"]
        self.X_std = state["X_std"]
        self.y_sigma_mean = state["y_sigma_mean"]
        self.y_sigma_std = state["y_sigma_std"]
        self.y_delta_mean = state["y_delta_mean"]
        self.y_delta_std = state["y_delta_std"]
        self.fold_metrics = state["fold_metrics"]

        input_dim = state["input_dim"]

        self.models_sigma = []
        self.models_delta = []

        for sigma_state, delta_state in zip(state["models_sigma_state"], state["models_delta_state"]):
            model_sigma = self._create_model(input_dim).to(self.device)
            model_delta = self._create_model(input_dim).to(self.device)

            model_sigma.load_state_dict(sigma_state)
            model_delta.load_state_dict(delta_state)

            self.models_sigma.append(model_sigma)
            self.models_delta.append(model_delta)


class AdvancedSurrogateModel:
    """Advanced surrogate model using only K-Fold + ResNet strategy."""

    def __init__(self):
        self.strategy = "kfold_resnet"
        self.trainer: Optional[KFoldTrainer] = None
        self.is_trained = False
        self.use_physics_features = True
        self.section_lib = SectionLibrary("src/data/steel_sections_large.json")

    def train(
        self,
        dataset_path: str,
        test_ratio: float = 0.1,
        use_physics_features: bool = True,
        **kwargs,
    ) -> Dict:
        """Train the model."""
        with open(dataset_path, "rb") as f:
            data = pickle.load(f)

        X = data["X"]
        y_sigma = data["y_sigma"]
        y_delta = data["y_delta"]

        feature_format = detect_feature_format(X)
        print(f"Detected feature format: {feature_format} ({X.shape[1]} dimensions)")

        if use_physics_features and feature_format == "index":
            raise ValueError("Index format is deprecated. Please convert to physics features first.")

        self.use_physics_features = use_physics_features

        n = len(X)
        n_test = int(n * test_ratio)
        indices = np.random.permutation(n)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        X_train, X_test = X[train_idx], X[test_idx]
        y_sigma_train, y_sigma_test = y_sigma[train_idx], y_sigma[test_idx]
        y_delta_train, y_delta_test = y_delta[train_idx], y_delta[test_idx]

        print(f"Training on {len(X_train)} samples, holdout test: {len(X_test)} samples")

        valid_keys = {f.name for f in fields(KFoldConfig)}
        cfg_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        config = KFoldConfig(**cfg_kwargs)

        self.trainer = KFoldTrainer(config)
        self.trainer.fit(X_train, y_sigma_train, y_delta_train)
        self.is_trained = True

        pred_sigma, pred_delta = self.trainer.predict(X_test)

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        metrics = {
            "sigma_mae": mean_absolute_error(y_sigma_test, pred_sigma),
            "sigma_rmse": np.sqrt(mean_squared_error(y_sigma_test, pred_sigma)),
            "sigma_r2": r2_score(y_sigma_test, pred_sigma),
            "delta_mae": mean_absolute_error(y_delta_test, pred_delta),
            "delta_rmse": np.sqrt(mean_squared_error(y_delta_test, pred_delta)),
            "delta_r2": r2_score(y_delta_test, pred_delta),
        }

        print("\n=== Final Test Results ===")
        print(
            f"  u_sigma: MAE={metrics['sigma_mae']:.4f}, RMSE={metrics['sigma_rmse']:.4f}, R2={metrics['sigma_r2']:.4f}"
        )
        print(
            f"  u_delta: MAE={metrics['delta_mae']:.4f}, RMSE={metrics['delta_rmse']:.4f}, R2={metrics['delta_r2']:.4f}"
        )

        return metrics

    def predict(self, config: DesignConfig) -> Tuple[float, float]:
        """Predict for a single config."""
        X = config.to_feature_vector(use_physics=self.use_physics_features).reshape(1, -1)
        pred_sigma, pred_delta = self.trainer.predict(X)
        return float(pred_sigma[0]), float(pred_delta[0])

    def predict_batch(self, configs: List[DesignConfig]) -> Tuple[np.ndarray, np.ndarray]:
        """Batch prediction."""
        X = np.array([cfg.to_feature_vector(use_physics=self.use_physics_features) for cfg in configs])
        return self.trainer.predict(X)

    def get_mass(self, config: DesignConfig) -> float:
        """Fast mass calculation."""
        return compute_mass_fast(config, self.section_lib)

    def save(self, path: str):
        """Save model."""
        data = {
            "strategy": self.strategy,
            "use_physics_features": self.use_physics_features,
            "trainer_state": self.trainer.get_state(),
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Model saved to: {path}")

    @classmethod
    def load(cls, path: str) -> "AdvancedSurrogateModel":
        """Load model."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        strategy = data.get("strategy", "kfold_resnet")
        if strategy != "kfold_resnet":
            raise ValueError(f"Unsupported model strategy: {strategy}. Only 'kfold_resnet' is supported.")

        model = cls()
        model.use_physics_features = data["use_physics_features"]
        model.trainer = KFoldTrainer(KFoldConfig())
        model.trainer.load_state(data["trainer_state"])

        model.is_trained = True
        print("Model loaded: strategy=kfold_resnet")
        return model


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Advanced Surrogate Model Training (KFold + ResNet only)")
    parser.add_argument("--dataset", type=str, default="results/surrogate_data/dataset.pkl")
    parser.add_argument("--output", type=str, default="results/surrogate_data/model_advanced.pkl")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--resnet_blocks", type=int, default=3)

    args = parser.parse_args()

    model = AdvancedSurrogateModel()
    model.train(
        args.dataset,
        n_folds=args.n_folds,
        hidden_dim=args.hidden_dim,
        resnet_blocks=args.resnet_blocks,
    )
    model.save(args.output)


if __name__ == "__main__":
    main()
