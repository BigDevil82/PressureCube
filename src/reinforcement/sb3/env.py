"""
Steel Section Optimization Environment for Stable-Baselines3.

Compatible with MaskablePPO from sb3-contrib.
Supports both FEA and Surrogate evaluation modes.
"""

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.heuristic.evaluator import StructuralEvaluator
from src.structure.section_library import SectionLibrary, SectionType


class SteelSectionOptEnvSB3(gym.Env):
    """
    Steel section optimization environment for SB3.

    State: 12-dim vector [L/12, W/4.5, H/4.5, i_f/max, i_m/max, i_s/max, i_b/max,
                          u_sigma, u_delta, m/m0, argmax_sigma/3, argmax_delta/3]
    Action: 81 discrete actions (4 section types × 3 deltas each, 3^4=81)
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        config = config or {}

        # Evaluation mode
        eval_mode = config.get("eval_mode", "fea")
        surrogate_path = config.get("surrogate_path", None)

        # Initialize evaluator
        self.evaluator = StructuralEvaluator(
            mode=eval_mode,
            surrogate_path=surrogate_path,
        )

        # Section library
        self.section_lib = SectionLibrary("src/data/steel_sections_large.json")
        self.sec_types = [SectionType.BOX, SectionType.H_MAIN, SectionType.H_SEC, SectionType.CHANNEL]
        self.sec_counts = [self.section_lib.section_counts[t] for t in self.sec_types]
        self.max_indices = np.array(self.sec_counts, dtype=np.int64) - 1

        # Action space: 81 discrete actions
        self.n_actions = 81
        self.action_space = spaces.Discrete(self.n_actions)

        # Observation space: 10-dim vector
        self.obs_dim = 10
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # Episode parameters
        self.T = int(config.get("T", 30))
        self.K_no_improve = int(config.get("K_no_improve", 5))
        self.eps_improve = float(config.get("eps_improve", 1e-3))

        # Reward parameters
        self.alpha = float(config.get("alpha", 20.0))
        self.beta = float(config.get("beta", 1.0))
        self.c_feas = float(config.get("c_feas", 1.0))
        self.penalty_feas_to_infeas = float(config.get("penalty_feas_to_infeas", 2.0))
        self.penalty_noop = float(config.get("penalty_noop", 0.05))

        # Geometry config (fixed or randomized)
        self.fixed_geometry = config.get("fixed_geometry", None)  # (L, W, H) or None
        self.pressure = float(config.get("pressure", 30000.0))

        # Topology config
        self.main_beam_spacing = float(config.get("main_beam_spacing", 0.7))
        self.num_secondary_width = int(config.get("num_secondary_width", 4))
        self.num_secondary_height = int(config.get("num_secondary_height", 4))
        self.add_support_beam = bool(config.get("add_support_beam", True))

        # Runtime state
        self.idx: np.ndarray = None
        self.L: float = None
        self.W: float = None
        self.H: float = None
        self.m0: float = None
        self.prev_v: float = None
        self.prev_mnorm: float = None
        self.no_improve: int = 0
        self.t: int = 0
        self._current_metrics: Dict = {}

    # ==================== Action Encoding ====================

    @staticmethod
    def decode_action(a: int) -> np.ndarray:
        """Decode action 0..80 to delta array [-1,0,+1]^4."""
        ds = []
        x = int(a)
        for _ in range(4):
            ds.append(x % 3)
            x //= 3
        ds = ds[::-1]
        return np.array([d - 1 for d in ds], dtype=np.int64)

    @staticmethod
    def encode_action(deltas: np.ndarray) -> int:
        """Encode delta array to action index."""
        digits = [int(d + 1) for d in deltas]
        a = 0
        for dig in digits:
            a = a * 3 + dig
        return int(a)

    def action_masks(self) -> np.ndarray:
        """Return valid action mask for MaskablePPO."""
        mask = np.ones(self.n_actions, dtype=bool)
        for a in range(self.n_actions):
            d = self.decode_action(a)
            new_idx = self.idx + d
            if np.any(new_idx < 0) or np.any(new_idx > self.max_indices):
                mask[a] = False
        return mask

    # ==================== Environment Core ====================

    def _sample_geometry(self, rng: np.random.Generator) -> Tuple[float, float, float]:
        """Sample random geometry."""
        L = float(round(rng.choice(np.arange(6.0, 12.0, 0.2)), 2))
        W = float(round(rng.choice(np.arange(3.0, 4.5, 0.3)), 2))
        H = float(round(rng.choice(np.arange(3.0, 4.5, 0.3)), 2))
        return L, W, H

    def _evaluate(self) -> Dict:
        """Run evaluation and return metrics."""
        result = self.evaluator.evaluate(
            L=self.L,
            W=self.W,
            H=self.H,
            pressure=self.pressure,
            main_beam_spacing=self.main_beam_spacing,
            num_secondary_width=self.num_secondary_width,
            num_secondary_height=self.num_secondary_height,
            add_support_beam=self.add_support_beam,
            sections=self.idx.tolist(),
        )
        return {
            "mass": result.mass,
            "u_sigma": result.u_sigma,
            "u_delta": result.u_delta,
            "feasible": result.feasible,
        }

    def _calc_m0(self) -> float:
        """Calculate mass with max sections (for normalization)."""
        result = self.evaluator.get_mass_only(
            L=self.L,
            W=self.W,
            H=self.H,
            main_beam_spacing=self.main_beam_spacing,
            num_secondary_width=self.num_secondary_width,
            num_secondary_height=self.num_secondary_height,
            add_support_beam=self.add_support_beam,
            sections=self.max_indices.tolist(),
        )
        return result

    def _build_obs(self, metrics: Dict) -> np.ndarray:
        """Build observation vector."""
        i_norm = np.array(
            [
                self.idx[0] / max(1, self.max_indices[0]),
                self.idx[1] / max(1, self.max_indices[1]),
                self.idx[2] / max(1, self.max_indices[2]),
                self.idx[3] / max(1, self.max_indices[3]),
            ],
            dtype=np.float32,
        )

        # argmax: which section type has max violation
        type_metrics = np.zeros((4, 2), dtype=np.float32)  # placeholder
        argmax_sigma = 0
        argmax_delta = 0

        obs = np.array(
            [
                float(self.L / 12.0),
                float(self.W / 4.5),
                float(self.H / 4.5),
                float(i_norm[0]),
                float(i_norm[1]),
                float(i_norm[2]),
                float(i_norm[3]),
                float(metrics["u_sigma"]),
                float(metrics["u_delta"]),
                float(metrics["mass"] / (self.m0 + 1e-9)),
                # float(argmax_sigma / 3.0),
                # float(argmax_delta / 3.0),
            ],
            dtype=np.float32,
        )
        return obs

    @staticmethod
    def _violation(u_sigma: float, u_delta: float) -> float:
        """Calculate constraint violation."""
        return max(0.0, u_sigma - 1.0) + max(0.0, u_delta - 1.0)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        options = options or {}
        rng = np.random.default_rng(seed) if seed is not None else self.np_random

        # Initialize sections
        if options.get("init_sections") is not None:
            self.idx = np.asarray(options["init_sections"], dtype=np.int64).copy()
        else:
            # Random initialization
            self.idx = np.array(
                [rng.integers(0, self.sec_counts[i]) for i in range(4)],
                dtype=np.int64,
            )

        # Geometry
        if self.fixed_geometry is not None:
            self.L, self.W, self.H = self.fixed_geometry
        elif options.get("geometry") is not None:
            self.L, self.W, self.H = options["geometry"]
        else:
            self.L, self.W, self.H = self._sample_geometry(rng)

        # Evaluate initial state
        metrics = self._evaluate()
        self._current_metrics = metrics

        # Calculate m0 for normalization
        self.m0 = self._calc_m0()

        # Initialize tracking variables
        self.prev_v = self._violation(metrics["u_sigma"], metrics["u_delta"])
        self.prev_mnorm = float(metrics["mass"] / (self.m0 + 1e-9))
        self.no_improve = 0
        self.t = 0

        obs = self._build_obs(metrics)
        info = {
            **metrics,
            "mnorm": self.prev_mnorm,
            "violation": self.prev_v,
            "action_mask": self.action_masks(),
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Decode action
        deltas = self.decode_action(int(action))

        # Check validity
        new_idx = self.idx + deltas
        if np.any(new_idx < 0) or np.any(new_idx > self.max_indices):
            # Invalid action (should not happen with masking)
            obs = self._build_obs(self._current_metrics)
            info = {"action_mask": self.action_masks(), "illegal_action": True}
            return obs, -1.0, False, False, info

        # Apply action
        self.idx = new_idx
        self.t += 1

        # Evaluate
        metrics = self._evaluate()
        self._current_metrics = metrics

        u_sigma = float(metrics["u_sigma"])
        u_delta = float(metrics["u_delta"])
        v = self._violation(u_sigma, u_delta)
        m = float(metrics["mass"])
        mnorm = float(m / (self.m0 + 1e-9))

        feasible_prev = self.prev_v <= 0.0
        feasible_now = v <= 0.0

        # Calculate reward
        if v > 0.0:
            # Infeasible: reward improvement in violation
            r = self.alpha * (self.prev_v - v) - self.beta * (mnorm - self.prev_mnorm)
        else:
            # Feasible: constant reward minus mass increase
            r = self.c_feas - self.beta * (mnorm - self.prev_mnorm)

        # Penalty for leaving feasible region
        if feasible_prev and not feasible_now:
            r -= self.penalty_feas_to_infeas

        # Penalty for no-op
        if np.all(deltas == 0):
            r -= self.penalty_noop

        # Early stopping: feasible and no improvement
        if feasible_now and (mnorm >= self.prev_mnorm - self.eps_improve):
            self.no_improve += 1
        else:
            self.no_improve = 0

        terminated = (self.t >= self.T) or (feasible_now and self.no_improve >= self.K_no_improve)
        truncated = False

        # Update state
        self.prev_v = v
        self.prev_mnorm = mnorm

        obs = self._build_obs(metrics)
        info = {
            **metrics,
            "mnorm": mnorm,
            "violation": v,
            "action_mask": self.action_masks(),
        }

        return obs, float(r), terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
