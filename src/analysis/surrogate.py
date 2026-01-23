"""
Surrogate Model for FEA Acceleration.

Trains a machine learning model to predict structural responses (stress ratio, displacement ratio)
from design parameters, replacing expensive FEA simulations during optimization.

Key features:
1. Data collection via Latin Hypercube Sampling (LHS) with parallel FEA
2. XGBoost/LightGBM for fast, accurate predictions
3. Fast mass calculation without FEA
4. Drop-in replacement interface for optimization loops
"""

import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from joblib import Parallel, delayed
from scipy.stats import qmc
from torch.utils.data import DataLoader, TensorDataset

from src.analysis.opensees_analysis import OpenSeesSolver
from src.analysis.post_processor import ResultProcessor
from src.structure.mesh_generator import MeshGenerator
from src.structure.section_library import SectionLibrary, SectionType
from src.structure.structural_model import StructuralModel, StructuralParameters

# ============================================================================
# Feature Engineering
# ============================================================================

# Global section library for feature extraction
_SECTION_LIB: Optional[SectionLibrary] = None


def get_section_lib() -> SectionLibrary:
    """Get or create singleton section library."""
    global _SECTION_LIB
    if _SECTION_LIB is None:
        _SECTION_LIB = SectionLibrary("src/data/steel_sections_large.json")
    return _SECTION_LIB


@dataclass
class DesignConfig:
    """A complete design configuration for surrogate model."""

    # Geometry
    L: float
    W: float
    H: float
    pressure: float  # in Pa

    # Topology
    main_beam_spacing: float
    num_secondary_width: int
    num_secondary_height: int
    add_support_beam: bool

    # Sections (indices)
    frame_section_index: int
    main_beam_section_index: int
    secondary_beam_section_index: int
    brace_section_index: int

    def to_feature_vector(self, use_physics: bool = True) -> np.ndarray:
        """
        Convert to feature vector for ML model.

        Args:
            use_physics: If True, use physical section properties instead of indices
        """
        if use_physics:
            return self._to_physics_features()
        else:
            return self._to_index_features()

    def _to_index_features(self) -> np.ndarray:
        """Original index-based features."""
        return np.array(
            [
                self.L,
                self.W,
                self.H,
                self.pressure / 1000.0,  # 保存时用的kPa
                self.main_beam_spacing,
                float(self.num_secondary_width),
                float(self.num_secondary_height),
                float(self.add_support_beam),
                float(self.frame_section_index),
                float(self.main_beam_section_index),
                float(self.secondary_beam_section_index),
                float(self.brace_section_index),
            ],
            dtype=np.float32,
        )

    def _to_physics_features(self) -> np.ndarray:
        """
        Physics-based features using section properties.

        Features:
        - Geometry: L, W, H, volume (L*W*H)
        - Load: pressure, total_load (pressure * L * W)
        - Topology: spacing, n_sec_w, n_sec_h, add_support, n_main_beams, n_sec_beams
        - Section properties (for each of 4 types): area, Ix, Iy
        - Derived: total_area, avg_Ix, stiffness_ratio
        """
        lib = get_section_lib()
        sec_types = [SectionType.BOX, SectionType.H_MAIN, SectionType.H_SEC, SectionType.CHANNEL]
        indices = [
            self.frame_section_index,
            self.main_beam_section_index,
            self.secondary_beam_section_index,
            self.brace_section_index,
        ]

        # Get section properties
        sections = [lib.get_section(st, idx) for st, idx in zip(sec_types, indices)]

        # Geometry features
        volume = self.L * self.W * self.H
        floor_area = self.L * self.W
        aspect_ratio = self.L / self.W if self.W > 0 else 1.0

        # Load features
        pressure_kPa = self.pressure / 1000.0  # Pa -> kPa
        total_load = self.pressure * floor_area  # Total pressure force (N)

        # Topology features
        n_main_beams = int(self.L / self.main_beam_spacing) + 1

        # Section physical properties (scaled for numerical stability)
        # area in cm², Ix in cm⁴ for better numerical range
        areas = [s.area * 1e4 for s in sections]  # m² -> cm²
        Ixs = [s.Ix * 1e8 for s in sections]  # m⁴ -> cm⁴
        Iys = [s.Iy * 1e8 for s in sections]

        # Derived features
        total_area = sum(areas)
        avg_Ix = np.mean(Ixs)
        min_Ix = min(Ixs)
        max_Ix = max(Ixs)
        Ix_ratio = min_Ix / (max_Ix + 1e-10)  # Stiffness balance

        # Stiffness-to-load ratio (proxy for displacement)
        stiffness_load_ratio = avg_Ix / (total_load / 1e6 + 1e-10)

        # Strength proxy: area / load
        strength_load_ratio = total_area / (total_load / 1e6 + 1e-10)

        features = [
            # Geometry (6)
            self.L,
            self.W,
            self.H,
            volume,
            floor_area,
            aspect_ratio,
            # Load (2)
            pressure_kPa,
            total_load / 1e6,  # Scale to MN
            # Topology (5)
            self.main_beam_spacing,
            float(self.num_secondary_width),
            float(self.num_secondary_height),
            float(self.add_support_beam),
            float(n_main_beams),
            # Section areas (4)
            areas[0],
            areas[1],
            areas[2],
            areas[3],
            # Section Ix (4)
            Ixs[0],
            Ixs[1],
            Ixs[2],
            Ixs[3],
            # Section Iy (4)
            Iys[0],
            Iys[1],
            Iys[2],
            Iys[3],
            # Derived (6)
            total_area,
            avg_Ix,
            min_Ix,
            Ix_ratio,
            stiffness_load_ratio,
            strength_load_ratio,
        ]

        return np.array(features, dtype=np.float32)

    @staticmethod
    def feature_names(use_physics: bool = True) -> List[str]:
        if use_physics:
            return [
                # Geometry
                "L",
                "W",
                "H",
                "volume",
                "floor_area",
                "aspect_ratio",
                # Load
                "pressure_kPa",
                "total_load_MN",
                # Topology
                "main_beam_spacing",
                "num_secondary_width",
                "num_secondary_height",
                "add_support_beam",
                "n_main_beams",
                # Section areas
                "frame_area",
                "main_area",
                "sec_area",
                "brace_area",
                # Section Ix
                "frame_Ix",
                "main_Ix",
                "sec_Ix",
                "brace_Ix",
                # Section Iy
                "frame_Iy",
                "main_Iy",
                "sec_Iy",
                "brace_Iy",
                # Derived
                "total_area",
                "avg_Ix",
                "min_Ix",
                "Ix_ratio",
                "stiffness_load_ratio",
                "strength_load_ratio",
            ]
        else:
            return [
                "L",
                "W",
                "H",
                "pressure_kPa",
                "main_beam_spacing",
                "num_secondary_width",
                "num_secondary_height",
                "add_support_beam",
                "frame_idx",
                "main_beam_idx",
                "secondary_beam_idx",
                "brace_idx",
            ]


def compute_mass_fast(config: DesignConfig, section_lib: SectionLibrary) -> float:
    """
    Fast mass calculation without FEA.
    Only builds the structural model and computes mass.
    """
    params = StructuralParameters(
        length=config.L,
        width=config.W,
        height=config.H,
        main_beam_spacing=config.main_beam_spacing,
        num_secondary_width=config.num_secondary_width,
        num_secondary_height=config.num_secondary_height,
        add_support_beam=config.add_support_beam,
        frame_section_index=config.frame_section_index,
        main_beam_section_index=config.main_beam_section_index,
        secondary_beam_section_index=config.secondary_beam_section_index,
        brace_section_index=config.brace_section_index,
    )
    model = StructuralModel(params, section_lib)
    model.generate()
    return model.calculate_mass_kg()


def convert_index_to_physics_features(X_index: np.ndarray) -> np.ndarray:
    """
    Convert old index-based features (12-dim) to new physics-based features (31-dim).

    This allows reusing existing datasets collected with the old feature format.

    Input format (12-dim):
        [L, W, H, pressure_kPa, main_beam_spacing, num_secondary_width,
         num_secondary_height, add_support_beam, frame_idx, main_idx, sec_idx, brace_idx]

    Output format (31-dim):
        Physics-based features with section properties instead of indices.

    Args:
        X_index: Array of shape (n_samples, 12) with index-based features

    Returns:
        Array of shape (n_samples, 31) with physics-based features
    """
    n_samples = X_index.shape[0]
    X_physics = []

    for i in range(n_samples):
        x = X_index[i]

        # Parse old features
        L = float(x[0])
        W = float(x[1])
        H = float(x[2])
        pressure_kPa = float(x[3])
        main_beam_spacing = float(x[4])
        num_secondary_width = int(x[5])
        num_secondary_height = int(x[6])
        add_support_beam = bool(x[7])
        frame_idx = int(x[8])
        main_idx = int(x[9])
        sec_idx = int(x[10])
        brace_idx = int(x[11])

        # Create DesignConfig and convert to physics features
        cfg = DesignConfig(
            L=L,
            W=W,
            H=H,
            pressure=pressure_kPa * 1000.0,  # Convert back to Pa
            main_beam_spacing=main_beam_spacing,
            num_secondary_width=num_secondary_width,
            num_secondary_height=num_secondary_height,
            add_support_beam=add_support_beam,
            frame_section_index=frame_idx,
            main_beam_section_index=main_idx,
            secondary_beam_section_index=sec_idx,
            brace_section_index=brace_idx,
        )

        X_physics.append(cfg._to_physics_features())

    return np.array(X_physics, dtype=np.float32)


def detect_feature_format(X: np.ndarray) -> str:
    """
    Detect whether features are in old index format (12-dim) or new physics format (31-dim).

    Returns: "index" or "physics"
    """
    n_features = X.shape[1]
    if n_features == 12:
        return "index"
    elif n_features == 31:
        return "physics"
    else:
        raise ValueError(f"Unknown feature format with {n_features} dimensions. Expected 12 or 31.")


# ============================================================================
# Data Collection
# ============================================================================


def _run_single_fea(model: StructuralModel, pressure: float) -> Tuple[float, float, float, bool]:
    """
    Run single FEA in isolated process.
    Returns: (u_sigma, u_delta, mass, success)
    """
    try:
        mass = model.calculate_mass_kg()
        mesh_generator = MeshGenerator(model, 0.5)
        model = mesh_generator.generate()

        solver = OpenSeesSolver(model)
        combo_strength = {"DL": 1.3, "LL": 1.5, "PL": 1.25}
        combo_service = {"DL": 1.0, "LL": 1.0, "PL": 1.0}

        res_strength = solver.run_load_combination(combo_strength, pressure_pa=pressure)
        strength_check = ResultProcessor(model).run_gb50017_check(res_strength)
        u_sigma = strength_check.max_stress_ratio

        res_service = solver.run_load_combination(combo_service, pressure_pa=pressure)
        disp_check = ResultProcessor(model).run_gb50017_check(res_service)
        u_delta = disp_check.max_deflection_ratio

        return (u_sigma, u_delta, mass, True)

    except Exception as e:
        print(f"FEA failed: {e}")
        return (0.0, 0.0, 0.0, False)


class DataCollector:
    """
    Collect training data for surrogate model using Latin Hypercube Sampling.
    """

    # Default parameter ranges
    GEOMETRY_RANGES = {
        "L": (6.0, 12.0),
        "W": (3.0, 4.5),
        "H": (3.0, 4.5),
    }

    TOPOLOGY_OPTIONS = {
        "main_beam_spacing": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        "num_secondary_width": [2, 3, 4, 5, 6],
        "num_secondary_height": [2, 3, 4, 5, 6],
        "add_support_beam": [True, False],
    }

    LOAD_OPTIONS = {
        "pressure_range": (20.0, 40.0),  # in kPa
    }

    def __init__(
        self,
        n_samples: int = 2000,
        n_jobs: int = 4,
        seed: int = 42,
    ):
        self.n_samples = n_samples
        self.n_jobs = n_jobs
        self.seed = seed

        # Get section counts
        self.section_lib = SectionLibrary("src/data/steel_sections_large.json")
        self.sec_types = [SectionType.BOX, SectionType.H_MAIN, SectionType.H_SEC, SectionType.CHANNEL]
        self.sec_counts = [self.section_lib.section_counts[t] for t in self.sec_types]

    def _generate_lhs_samples(self) -> List[DesignConfig]:
        """Generate samples using Latin Hypercube Sampling."""

        # Dimensions: L, W, H, spacing_idx, sec_w_idx, sec_h_idx, support_idx, 4 section indices, pressure
        n_dims = 12

        sampler = qmc.LatinHypercube(d=n_dims, seed=self.seed)
        samples = sampler.random(n=self.n_samples)  # [0, 1] range

        configs = []
        for i in range(self.n_samples):
            s = samples[i]

            # Map to actual values
            L = self.GEOMETRY_RANGES["L"][0] + s[0] * (
                self.GEOMETRY_RANGES["L"][1] - self.GEOMETRY_RANGES["L"][0]
            )
            W = self.GEOMETRY_RANGES["W"][0] + s[1] * (
                self.GEOMETRY_RANGES["W"][1] - self.GEOMETRY_RANGES["W"][0]
            )
            H = self.GEOMETRY_RANGES["H"][0] + s[2] * (
                self.GEOMETRY_RANGES["H"][1] - self.GEOMETRY_RANGES["H"][0]
            )

            # Topology (discrete)
            spacing_idx = int(s[3] * len(self.TOPOLOGY_OPTIONS["main_beam_spacing"]))
            spacing_idx = min(spacing_idx, len(self.TOPOLOGY_OPTIONS["main_beam_spacing"]) - 1)
            spacing = self.TOPOLOGY_OPTIONS["main_beam_spacing"][spacing_idx]

            sec_w_idx = int(s[4] * len(self.TOPOLOGY_OPTIONS["num_secondary_width"]))
            sec_w_idx = min(sec_w_idx, len(self.TOPOLOGY_OPTIONS["num_secondary_width"]) - 1)
            sec_w = self.TOPOLOGY_OPTIONS["num_secondary_width"][sec_w_idx]

            sec_h_idx = int(s[5] * len(self.TOPOLOGY_OPTIONS["num_secondary_height"]))
            sec_h_idx = min(sec_h_idx, len(self.TOPOLOGY_OPTIONS["num_secondary_height"]) - 1)
            sec_h = self.TOPOLOGY_OPTIONS["num_secondary_height"][sec_h_idx]

            add_support = s[6] > 0.5

            # Section indices
            sections = [
                int(s[7] * self.sec_counts[0]),
                int(s[8] * self.sec_counts[1]),
                int(s[9] * self.sec_counts[2]),
                int(s[10] * self.sec_counts[3]),
            ]
            # Clamp to valid range
            sections = [min(sec, cnt - 1) for sec, cnt in zip(sections, self.sec_counts)]

            pressure = self.LOAD_OPTIONS["pressure_range"][0] + s[11] * (
                self.LOAD_OPTIONS["pressure_range"][1] - self.LOAD_OPTIONS["pressure_range"][0]
            )

            configs.append(
                DesignConfig(
                    L=round(L, 2),
                    W=round(W, 2),
                    H=round(H, 2),
                    pressure=float(round(pressure, 2)) * 1000.0,  # convert to Pa
                    main_beam_spacing=spacing,
                    num_secondary_width=sec_w,
                    num_secondary_height=sec_h,
                    add_support_beam=add_support,
                    frame_section_index=sections[0],
                    main_beam_section_index=sections[1],
                    secondary_beam_section_index=sections[2],
                    brace_section_index=sections[3],
                )
            )

        return configs

    def collect(self, output_dir: str = "results/surrogate_data") -> str:
        """
        Collect training data via parallel FEA.
        Processes in batches to avoid memory issues.

        Returns: Path to saved dataset
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"Generating {self.n_samples} LHS samples...")
        configs = self._generate_lhs_samples()

        print(f"Running FEA in parallel (n_jobs={self.n_jobs})...")
        start_time = time.time()

        # Process in batches to avoid memory issues
        batch_size = 5000
        all_results = []

        for batch_start in range(0, self.n_samples, batch_size):
            batch_end = min(batch_start + batch_size, self.n_samples)
            batch_configs = configs[batch_start:batch_end]

            print(
                f"Processing batch {batch_start//batch_size + 1}/{(self.n_samples + batch_size - 1)//batch_size} "
                f"(samples {batch_start}-{batch_end-1})..."
            )

            models = []
            for cfg in batch_configs:
                model = StructuralModel(
                    StructuralParameters(
                        length=cfg.L,
                        width=cfg.W,
                        height=cfg.H,
                        main_beam_spacing=cfg.main_beam_spacing,
                        num_secondary_width=cfg.num_secondary_width,
                        num_secondary_height=cfg.num_secondary_height,
                        add_support_beam=cfg.add_support_beam,
                        frame_section_index=cfg.frame_section_index,
                        main_beam_section_index=cfg.main_beam_section_index,
                        secondary_beam_section_index=cfg.secondary_beam_section_index,
                        brace_section_index=cfg.brace_section_index,
                    ),
                    self.section_lib,
                )
                model.generate()
                models.append(model)

            batch_results = Parallel(n_jobs=self.n_jobs, backend="loky", verbose=10)(
                delayed(_run_single_fea)(model, cfg.pressure) for model, cfg in zip(models, batch_configs)
            )

            all_results.extend(batch_results)

            # Clear memory
            del models
            del batch_results

        results = all_results

        elapsed = time.time() - start_time
        print(
            f"FEA completed in {elapsed:.1f}s ({elapsed/self.n_samples:.2f}s per sample with {self.n_jobs} workers)"
        )

        # Build dataset
        X = []
        y_sigma = []
        y_delta = []
        masses = []
        valid_count = 0

        for cfg, res in zip(configs, results):
            # print(f"Processing sample: {cfg}")
            # print(f"  Result: {res}")
            u_sigma, u_delta, mass, success = res
            if success:
                X.append(cfg.to_feature_vector())
                y_sigma.append(u_sigma)
                y_delta.append(u_delta)
                masses.append(mass)
                valid_count += 1

        X = np.array(X)
        y_sigma = np.array(y_sigma)
        y_delta = np.array(y_delta)
        masses = np.array(masses)

        print(f"Valid samples: {valid_count}/{self.n_samples} ({100*valid_count/self.n_samples:.1f}%)")

        # Save dataset
        dataset = {
            "X": X,
            "y_sigma": y_sigma,
            "y_delta": y_delta,
            "masses": masses,
            "feature_names": DesignConfig.feature_names(),
            "n_samples": valid_count,
            "collection_time": elapsed,
        }

        save_path = os.path.join(output_dir, "dataset.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(dataset, f)

        # Also save as JSON for inspection
        json_data = {
            "n_samples": valid_count,
            "collection_time": elapsed,
            "feature_names": DesignConfig.feature_names(),
            "y_sigma_stats": {
                "min": float(y_sigma.min()),
                "max": float(y_sigma.max()),
                "mean": float(y_sigma.mean()),
            },
            "y_delta_stats": {
                "min": float(y_delta.min()),
                "max": float(y_delta.max()),
                "mean": float(y_delta.mean()),
            },
        }
        with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
            json.dump(json_data, f, indent=2)

        print(f"Dataset saved to: {save_path}")
        return save_path


# ============================================================================
# Neural Network Components
# ============================================================================


class MLPRegressor(nn.Module):
    """Multi-layer perceptron for regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        # Output layer (single value)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class MLPSurrogate:
    """
    MLP-based surrogate model with normalization and early stopping.
    Predicts both u_sigma and u_delta using separate networks.
    """

    def __init__(
        self,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 200,
        patience: int = 20,
        device: str = "auto",
    ):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_sigma: Optional[MLPRegressor] = None
        self.model_delta: Optional[MLPRegressor] = None

        # Normalization parameters
        self.X_mean: Optional[np.ndarray] = None
        self.X_std: Optional[np.ndarray] = None
        self.y_sigma_mean: float = 0.0
        self.y_sigma_std: float = 1.0
        self.y_delta_mean: float = 0.0
        self.y_delta_std: float = 1.0

    def _normalize_X(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.X_mean = X.mean(axis=0)
            self.X_std = X.std(axis=0) + 1e-8
        return (X - self.X_mean) / self.X_std

    def _normalize_y(self, y: np.ndarray, target: str, fit: bool = False) -> np.ndarray:
        if fit:
            if target == "sigma":
                self.y_sigma_mean = y.mean()
                self.y_sigma_std = y.std() + 1e-8
            else:
                self.y_delta_mean = y.mean()
                self.y_delta_std = y.std() + 1e-8

        if target == "sigma":
            return (y - self.y_sigma_mean) / self.y_sigma_std
        else:
            return (y - self.y_delta_mean) / self.y_delta_std

    def _denormalize_y(self, y: np.ndarray, target: str) -> np.ndarray:
        if target == "sigma":
            return y * self.y_sigma_std + self.y_sigma_mean
        else:
            return y * self.y_delta_std + self.y_delta_mean

    def _train_single_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        target: str,
    ) -> MLPRegressor:
        """Train a single MLP model with early stopping."""
        input_dim = X_train.shape[1]
        model = MLPRegressor(input_dim, self.hidden_dims, self.dropout).to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
        criterion = nn.MSELoss()

        # Normalize
        X_train_norm = self._normalize_X(X_train, fit=(target == "sigma"))
        y_train_norm = self._normalize_y(y_train, target, fit=True)
        X_val_norm = self._normalize_X(X_val)
        y_val_norm = self._normalize_y(y_val, target)

        # Create dataloaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_norm),
            torch.FloatTensor(y_train_norm),
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        X_val_t = torch.FloatTensor(X_val_norm).to(self.device)
        y_val_t = torch.FloatTensor(y_val_norm).to(self.device)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.max_epochs):
            # Training
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(X_batch)

            train_loss /= len(train_dataset)

            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

            if patience_counter >= self.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # Load best model
        model.load_state_dict(best_state)
        return model

    def fit(
        self,
        X_train: np.ndarray,
        y_sigma_train: np.ndarray,
        y_delta_train: np.ndarray,
        X_val: np.ndarray,
        y_sigma_val: np.ndarray,
        y_delta_val: np.ndarray,
    ):
        """Train both sigma and delta models."""
        print(f"Training MLP on {self.device}...")

        print("Training u_sigma model...")
        self.model_sigma = self._train_single_model(X_train, y_sigma_train, X_val, y_sigma_val, "sigma")

        print("Training u_delta model...")
        self.model_delta = self._train_single_model(X_train, y_delta_train, X_val, y_delta_val, "delta")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict both u_sigma and u_delta."""
        X_norm = self._normalize_X(X)
        X_t = torch.FloatTensor(X_norm).to(self.device)

        self.model_sigma.eval()
        self.model_delta.eval()

        with torch.no_grad():
            pred_sigma = self.model_sigma(X_t).cpu().numpy()
            pred_delta = self.model_delta(X_t).cpu().numpy()

        # Denormalize
        pred_sigma = self._denormalize_y(pred_sigma, "sigma")
        pred_delta = self._denormalize_y(pred_delta, "delta")

        return pred_sigma, pred_delta

    def get_state(self) -> Dict:
        """Get state for serialization."""
        return {
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "model_sigma_state": self.model_sigma.state_dict() if self.model_sigma else None,
            "model_delta_state": self.model_delta.state_dict() if self.model_delta else None,
            "X_mean": self.X_mean,
            "X_std": self.X_std,
            "y_sigma_mean": self.y_sigma_mean,
            "y_sigma_std": self.y_sigma_std,
            "y_delta_mean": self.y_delta_mean,
            "y_delta_std": self.y_delta_std,
            "input_dim": self.model_sigma.network[0].in_features if self.model_sigma else None,
        }

    def load_state(self, state: Dict):
        """Load state from serialization."""
        self.hidden_dims = state["hidden_dims"]
        self.dropout = state["dropout"]
        self.X_mean = state["X_mean"]
        self.X_std = state["X_std"]
        self.y_sigma_mean = state["y_sigma_mean"]
        self.y_sigma_std = state["y_sigma_std"]
        self.y_delta_mean = state["y_delta_mean"]
        self.y_delta_std = state["y_delta_std"]

        input_dim = state["input_dim"]
        self.model_sigma = MLPRegressor(input_dim, self.hidden_dims, self.dropout).to(self.device)
        self.model_delta = MLPRegressor(input_dim, self.hidden_dims, self.dropout).to(self.device)

        self.model_sigma.load_state_dict(state["model_sigma_state"])
        self.model_delta.load_state_dict(state["model_delta_state"])


# ============================================================================
# Surrogate Model
# ============================================================================


class SurrogateModel:
    """
    Surrogate model for predicting structural responses.

    Predicts:
    - u_sigma: max stress ratio
    - u_delta: max displacement ratio

    Does NOT predict mass (use compute_mass_fast instead).
    """

    def __init__(self, model_type: str = "xgb"):
        """
        Args:
            model_type: "xgb" for XGBoost, "lgb" for LightGBM, "mlp" for neural network
        """
        self.model_type = model_type
        self.model_sigma = None
        self.model_delta = None
        self.mlp_surrogate: Optional[MLPSurrogate] = None  # For MLP model
        self.is_trained = False
        self.use_physics_features = True  # Default to physics features

        # For fast mass calculation
        self.section_lib = SectionLibrary("src/data/steel_sections_large.json")

    def train(
        self,
        dataset_path: str,
        test_ratio: float = 0.1,
        use_physics_features: bool = True,
        **model_params,
    ) -> Dict[str, float]:
        """
        Train surrogate model on collected data.

        Args:
            dataset_path: Path to dataset pickle file
            test_ratio: Fraction of data to use for testing
            use_physics_features: If True, convert to physics features if needed

        Returns: Dictionary with training metrics
        """
        # Load dataset
        with open(dataset_path, "rb") as f:
            data = pickle.load(f)

        X = data["X"]
        y_sigma = data["y_sigma"]
        y_delta = data["y_delta"]

        # Auto-detect and convert feature format if needed
        feature_format = detect_feature_format(X)
        print(f"Detected feature format: {feature_format} ({X.shape[1]} dimensions)")
        # print(f"first sample features: {X[0]}")
        # print(f"first sample target: u_sigma={y_sigma[0]}, u_delta={y_delta[0]}")
        # exit(0)

        if use_physics_features and feature_format == "index":
            print("Converting index features to physics features...")
            X = convert_index_to_physics_features(X)
            print(f"Converted to {X.shape[1]} physics features")

        self.use_physics_features = use_physics_features
        # print(f"first sample features: {X[0].tolist()}")

        # Train/test split
        n = len(X)
        n_test = int(n * test_ratio)
        indices = np.random.permutation(n)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        X_train, X_test = X[train_idx], X[test_idx]
        y_sigma_train, y_sigma_test = y_sigma[train_idx], y_sigma[test_idx]
        y_delta_train, y_delta_test = y_delta[train_idx], y_delta[test_idx]

        print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")

        if self.model_type == "mlp":
            # Train MLP model
            self.mlp_surrogate = MLPSurrogate(**model_params)
            self.mlp_surrogate.fit(
                X_train,
                y_sigma_train,
                y_delta_train,
                X_test,
                y_sigma_test,
                y_delta_test,
            )
            pred_sigma_test, pred_delta_test = self.mlp_surrogate.predict(X_test)
        else:
            # Default model parameters for XGBoost
            default_params = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": 42,
            }
            default_params.update(model_params)

            self.model_sigma = xgb.XGBRegressor(**default_params)
            self.model_delta = xgb.XGBRegressor(**default_params)

            # Train models
            print("Training u_sigma model...")
            self.model_sigma.fit(X_train, y_sigma_train)

            print("Training u_delta model...")
            self.model_delta.fit(X_train, y_delta_train)

            # Evaluate
            pred_sigma_test = self.model_sigma.predict(X_test)
            pred_delta_test = self.model_delta.predict(X_test)

        self.is_trained = True

        # Metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        metrics = {
            "sigma_mae": mean_absolute_error(y_sigma_test, pred_sigma_test),
            "sigma_rmse": np.sqrt(mean_squared_error(y_sigma_test, pred_sigma_test)),
            "sigma_r2": r2_score(y_sigma_test, pred_sigma_test),
            "delta_mae": mean_absolute_error(y_delta_test, pred_delta_test),
            "delta_rmse": np.sqrt(mean_squared_error(y_delta_test, pred_delta_test)),
            "delta_r2": r2_score(y_delta_test, pred_delta_test),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

        print("\nTraining Results:")
        print(
            f"  u_sigma: MAE={metrics['sigma_mae']:.4f}, RMSE={metrics['sigma_rmse']:.4f}, R2={metrics['sigma_r2']:.4f}"
        )
        print(
            f"  u_delta: MAE={metrics['delta_mae']:.4f}, RMSE={metrics['delta_rmse']:.4f}, R2={metrics['delta_r2']:.4f}"
        )

        return metrics

    def predict(self, config: DesignConfig) -> Tuple[float, float]:
        """
        Predict structural responses for a design configuration.

        Returns: (u_sigma, u_delta)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        X = config.to_feature_vector(use_physics=self.use_physics_features).reshape(1, -1)

        if self.model_type == "mlp":
            pred_sigma, pred_delta = self.mlp_surrogate.predict(X)
            return float(pred_sigma[0]), float(pred_delta[0])
        else:
            u_sigma = float(self.model_sigma.predict(X)[0])
            u_delta = float(self.model_delta.predict(X)[0])
            return u_sigma, u_delta

    def predict_batch(self, configs: List[DesignConfig]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch prediction for multiple configurations.

        Returns: (u_sigma_array, u_delta_array)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        X = np.array([cfg.to_feature_vector(use_physics=self.use_physics_features) for cfg in configs])

        if self.model_type == "mlp":
            return self.mlp_surrogate.predict(X)
        else:
            u_sigma = self.model_sigma.predict(X)
            u_delta = self.model_delta.predict(X)
            return u_sigma, u_delta

    def get_mass(self, config: DesignConfig) -> float:
        """Fast mass calculation without FEA."""
        return compute_mass_fast(config, self.section_lib)

    def evaluate(self, config: DesignConfig) -> Dict[str, float]:
        """
        Full evaluation: predict responses + compute mass.
        Drop-in replacement for FEA-based evaluation.

        Returns: Dict with u_sigma, u_delta, mass, feasible
        """
        u_sigma, u_delta = self.predict(config)
        mass = self.get_mass(config)
        feasible = (u_sigma <= 1.0) and (u_delta <= 1.0)

        return {
            "u_sigma": u_sigma,
            "u_delta": u_delta,
            "mass": mass,
            "feasible": feasible,
        }

    def save(self, path: str):
        """Save trained model."""
        if not self.is_trained:
            raise RuntimeError("Model not trained.")

        data = {
            "model_type": self.model_type,
            "use_physics_features": self.use_physics_features,
        }

        if self.model_type == "mlp":
            data["mlp_state"] = self.mlp_surrogate.get_state()
        else:
            data["model_sigma"] = self.model_sigma
            data["model_delta"] = self.model_delta

        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Model saved to: {path}")

    @classmethod
    def load(cls, path: str) -> "SurrogateModel":
        """Load trained model."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        model = cls(model_type=data["model_type"])
        model.use_physics_features = data.get("use_physics_features", True)

        if data["model_type"] == "mlp":
            model.mlp_surrogate = MLPSurrogate()
            model.mlp_surrogate.load_state(data["mlp_state"])
        else:
            model.model_sigma = data["model_sigma"]
            model.model_delta = data["model_delta"]

        model.is_trained = True
        print(
            f"Model loaded from: {path} (type={data['model_type']}, physics_features={model.use_physics_features})"
        )
        return model


# ============================================================================
# CLI
# ============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Surrogate Model for FEA Acceleration")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Data collection
    collect_parser = subparsers.add_parser("collect", help="Collect training data")
    collect_parser.add_argument("--n_samples", type=int, default=2000, help="Number of samples")
    collect_parser.add_argument("--n_jobs", type=int, default=4, help="Parallel workers")
    collect_parser.add_argument("--output_dir", type=str, default="results/surrogate_data")
    collect_parser.add_argument("--seed", type=int, default=42)

    # Train model
    train_parser = subparsers.add_parser("train", help="Train surrogate model")
    train_parser.add_argument("--dataset", type=str, default="results/surrogate_data/dataset.pkl")
    train_parser.add_argument("--model_type", type=str, default="xgb", choices=["xgb", "lgb", "mlp"])
    train_parser.add_argument("--output", type=str, default="results/surrogate_data/model.pkl")

    args = parser.parse_args()

    if args.command == "collect":
        collector = DataCollector(
            n_samples=args.n_samples,
            n_jobs=args.n_jobs,
            seed=args.seed,
        )
        collector.collect(output_dir=args.output_dir)

    elif args.command == "train":
        model = SurrogateModel(model_type=args.model_type)
        metrics = model.train(args.dataset)
        model.save(args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
