"""
Steel Section Optimization Environment for Stable-Baselines3.

Compatible with MaskablePPO from sb3-contrib.
Supports both FEA and Surrogate evaluation modes.

Extended version: optimizes both section indices (4) and topology parameters (4).
"""

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.analysis.evaluator import StructuralEvaluator
from src.optimization.design_space import get_layout_adapter, get_section_lib
from src.structure.section_library import SectionType


class SteelSectionOptEnvSB3(gym.Env):
    """
    Steel section optimization environment for SB3.

    Optimizes 8 variables:
    - 4 section indices: frame, main, secondary, brace
    - 4 topology parameters: main_beam_spacing, num_secondary_width/height, add_support_beam

    Action space: MultiDiscrete([3,3,3,3, 3,3,3, 2])
    - First 4: section delta {-1, 0, +1}
    - Next 3: topology delta {-1, 0, +1} for spacing/num_secondary
    - Last 1: support beam toggle {0=keep, 1=toggle}

    Observation space: ~31-dim vector with geometry, section properties, topology, and performance metrics.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        config = config or {}

        self.layout_strategy = config.get("layout_strategy", "research")

        # Evaluation mode
        eval_mode = config.get("eval_mode", "fea")
        surrogate_path = config.get("surrogate_path", None)

        # Initialize evaluator
        self.evaluator = StructuralEvaluator(
            mode=eval_mode,
            surrogate_path=surrogate_path,
            mesh_size=config.get("mesh_size", 0.5),
        )

        # Layout and section library
        self.adapter = get_layout_adapter(self.layout_strategy)
        self.section_lib = get_section_lib(self.layout_strategy)
        self.sec_types = [SectionType.BOX, SectionType.H_MAIN, SectionType.H_SEC, SectionType.CHANNEL]
        self.sec_counts = [self.section_lib.section_counts[t] for t in self.sec_types]
        self.max_indices = np.array(self.sec_counts, dtype=np.int64) - 1

        specs = {spec.name: spec for spec in self.adapter.topology_specs()}
        self.spacing_options = np.array(list(specs["main_beam_spacing"].choices), dtype=np.float64)
        self.support_spacing_options = np.array(
            list(specs["support_beam_max_spacing"].choices),
            dtype=np.float64,
        ) if "support_beam_max_spacing" in specs else np.array([0.0], dtype=np.float64)
        self.num_secondary_range = self._int_range(specs, "num_secondary_width", default=(0, 0))

        # Precompute normalization ranges for section properties
        self._compute_section_ranges()

        # Action space: MultiDiscrete
        # [sec_frame, sec_main, sec_sec, sec_brace, spacing, num_w, num_h, support]
        # First 7: delta {-1, 0, +1} -> 3 choices
        # Last 1: toggle {0=keep, 1=toggle} -> 2 choices
        self.action_space = spaces.MultiDiscrete([3, 3, 3, 3, 3, 3, 3, 2])

        # Observation space: extended with physics features
        # [geometry(3) + section_idx_norm(4) + section_physics(12) + topology(4) + pressure(1) + performance(3)] = 27
        self.obs_dim = 27
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

        # Runtime state
        self.idx: np.ndarray = None  # Section indices [4]
        self.spacing_idx: int = 0  # Index into spacing_options
        self.support_spacing_idx: int = 0
        self.num_secondary_width: int = 4
        self.num_secondary_height: int = 4
        self.add_support_beam: bool = True
        self.L: float = None
        self.W: float = None
        self.H: float = None
        self.m0: float = None
        self.prev_v: float = None
        self.prev_mnorm: float = None
        self.no_improve: int = 0
        self.t: int = 0
        self._current_metrics: Dict = {}

    def _compute_section_ranges(self):
        """Precompute min/max for section properties for normalization."""
        self.section_ranges = {}
        for sec_type in self.sec_types:
            sections = [
                self.section_lib.get_section(sec_type, i)
                for i in range(self.section_lib.section_counts[sec_type])
            ]
            areas = [s.area for s in sections]
            Ixs = [s.Ix for s in sections]
            Iys = [s.Iy for s in sections]
            self.section_ranges[sec_type] = {
                "area": (min(areas), max(areas)),
                "Ix": (min(Ixs), max(Ixs)),
                "Iy": (min(Iys), max(Iys)),
            }

        # Global max for normalization (use max across all types)
        self.max_area = max(r["area"][1] for r in self.section_ranges.values())
        self.max_Ix = max(r["Ix"][1] for r in self.section_ranges.values())
        self.max_Iy = max(r["Iy"][1] for r in self.section_ranges.values())

    @property
    def main_beam_spacing(self) -> float:
        """Current main beam spacing value."""
        return float(self.spacing_options[self.spacing_idx])

    @property
    def support_beam_max_spacing(self) -> float:
        return float(self.support_spacing_options[self.support_spacing_idx])

    # ==================== Action Encoding ====================

    @staticmethod
    def decode_action(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Decode MultiDiscrete action to deltas.

        Args:
            action: np.ndarray of shape (8,) with values:
                    [0-3]: section deltas (0,1,2 -> -1,0,+1)
                    [4-6]: topology deltas (0,1,2 -> -1,0,+1)
                    [7]: support beam toggle (0=keep, 1=toggle)

        Returns:
            section_deltas: np.ndarray of shape (4,) with values {-1, 0, +1}
            topology_deltas: np.ndarray of shape (3,) with values {-1, 0, +1}
            support_toggle: int (0 or 1)
        """
        section_deltas = np.array([action[i] - 1 for i in range(4)], dtype=np.int64)
        topology_deltas = np.array([action[i] - 1 for i in range(4, 7)], dtype=np.int64)
        support_toggle = int(action[7])
        return section_deltas, topology_deltas, support_toggle

    def action_masks(self) -> np.ndarray:
        """
        Return valid action mask for MaskablePPO with MultiDiscrete.

        For MultiDiscrete, mask is a flat array of size sum(nvec).
        Each action dimension has its own mask segment.
        """
        nvec = self.action_space.nvec  # [3,3,3,3, 3,3,3, 2]
        mask = np.ones(sum(nvec), dtype=bool)
        offset = 0

        # Section masks (first 4 dimensions, each size 3)
        for i in range(4):
            for delta_choice in range(3):  # 0,1,2 -> -1,0,+1
                delta = delta_choice - 1
                new_idx = self.idx[i] + delta
                if new_idx < 0 or new_idx > self.max_indices[i]:
                    mask[offset + delta_choice] = False
            offset += nvec[i]

        # Spacing mask (dimension 4, size 3)
        for delta_choice in range(3):
            delta = delta_choice - 1
            new_spacing_idx = self.spacing_idx + delta
            if new_spacing_idx < 0 or new_spacing_idx >= len(self.spacing_options):
                mask[offset + delta_choice] = False
        offset += nvec[4]

        # Layout-specific topology slot 1.
        for delta_choice in range(3):
            delta = delta_choice - 1
            if self.layout_strategy == "research":
                new_val = self.num_secondary_width + delta
                invalid = new_val < self.num_secondary_range[0] or new_val > self.num_secondary_range[1]
            else:
                new_val = self.support_spacing_idx + delta
                invalid = new_val < 0 or new_val >= len(self.support_spacing_options)
            if invalid:
                mask[offset + delta_choice] = False
        offset += nvec[5]

        # Layout-specific topology slot 2. Engineering has no third topology
        # delta while preserving the old 8-dim action space.
        for delta_choice in range(3):
            delta = delta_choice - 1
            if self.layout_strategy == "research":
                new_val = self.num_secondary_height + delta
                invalid = new_val < self.num_secondary_range[0] or new_val > self.num_secondary_range[1]
            else:
                invalid = delta != 0
            if invalid:
                mask[offset + delta_choice] = False
        offset += nvec[6]

        # support_beam toggle (dimension 7, size 2) - always valid
        # mask[offset:offset+2] stays True

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
        result = self.evaluator.evaluate(self._build_design(self.idx.tolist()))
        return {
            "mass": result.mass,
            "u_sigma": result.u_sigma,
            "u_delta": result.u_delta,
            "feasible": result.feasible,
        }

    def _calc_m0(self) -> float:
        """Calculate mass with max sections (for normalization)."""
        return self.evaluator.get_mass_only(self._build_design(self.max_indices.tolist()))

    def _build_design(self, sections: list[int]):
        return self.adapter.build_design_from_sections(
            geometry={"L": self.L, "W": self.W, "H": self.H},
            pressure=self.pressure,
            topology=self._topology_for_design(),
            sections=sections,
        )

    def _get_section_properties(self) -> Tuple[List[float], List[float], List[float]]:
        """Get current section properties for observation."""
        areas, Ixs, Iys = [], [], []
        for i, sec_type in enumerate(self.sec_types):
            section = self.section_lib.get_section(sec_type, int(self.idx[i]))
            areas.append(section.area)
            Ixs.append(section.Ix)
            Iys.append(section.Iy)
        return areas, Ixs, Iys

    def _build_obs(self, metrics: Dict) -> np.ndarray:
        """
        Build observation vector with physics features.

        Components (26-dim total):
        - Geometry (3): L, W, H normalized
        - Section indices (4): normalized by max index
        - Section physics (12): area, Ix, Iy for each of 4 section types
        - Topology (4): spacing, num_w, num_h, add_support
        - Performance (3): u_sigma, u_delta, mnorm
        """
        # Section index normalization
        idx_norm = np.array(
            [self.idx[i] / max(1, self.max_indices[i]) for i in range(4)],
            dtype=np.float32,
        )

        # Section physical properties
        areas, Ixs, Iys = self._get_section_properties()
        areas_norm = [a / (self.max_area + 1e-12) for a in areas]
        Ixs_norm = [ix / (self.max_Ix + 1e-12) for ix in Ixs]
        Iys_norm = [iy / (self.max_Iy + 1e-12) for iy in Iys]

        # Topology normalization. The shape stays fixed for old policies:
        # research = spacing, num_w, num_h, support
        # engineering = spacing, support spacing, placeholder, support
        spacing_norm = self.spacing_idx / (len(self.spacing_options) - 1)
        if self.layout_strategy == "research":
            num_w_norm = self._norm_int(self.num_secondary_width, self.num_secondary_range)
            num_h_norm = self._norm_int(self.num_secondary_height, self.num_secondary_range)
        else:
            num_w_norm = self._norm_index(self.support_spacing_idx, len(self.support_spacing_options))
            num_h_norm = 0.0

        obs = np.array(
            [
                # Geometry (3)
                float(self.L / 12.0),
                float(self.W / 4.5),
                float(self.H / 4.5),
                # Section indices (4)
                float(idx_norm[0]),
                float(idx_norm[1]),
                float(idx_norm[2]),
                float(idx_norm[3]),
                # Section areas (4)
                float(areas_norm[0]),
                float(areas_norm[1]),
                float(areas_norm[2]),
                float(areas_norm[3]),
                # Section Ix (4)
                float(Ixs_norm[0]),
                float(Ixs_norm[1]),
                float(Ixs_norm[2]),
                float(Ixs_norm[3]),
                # Section Iy (4)
                float(Iys_norm[0]),
                float(Iys_norm[1]),
                float(Iys_norm[2]),
                float(Iys_norm[3]),
                # Topology (4)
                float(spacing_norm),
                float(num_w_norm),
                float(num_h_norm),
                float(self.add_support_beam),
                # pressure
                float(self.pressure / 40000.0),
                # Performance (3)
                float(metrics["u_sigma"]),
                float(metrics["u_delta"]),
                float(metrics["mass"] / (self.m0 + 1e-9)),
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

        # Initialize topology
        if options.get("init_topology") is not None:
            topo = options["init_topology"]
            self.spacing_idx = self._choice_idx(
                self.spacing_options,
                topo.get("main_beam_spacing", topo.get("spacing")),
                topo.get("spacing_idx"),
                rng,
            )
            if self.layout_strategy == "research":
                self.num_secondary_width = int(
                    topo.get("num_secondary_width", self._sample_int(rng, self.num_secondary_range))
                )
                self.num_secondary_height = int(
                    topo.get("num_secondary_height", self._sample_int(rng, self.num_secondary_range))
                )
            else:
                self.support_spacing_idx = self._choice_idx(
                    self.support_spacing_options,
                    topo.get("support_beam_max_spacing"),
                    topo.get("support_spacing_idx"),
                    rng,
                )
            self.add_support_beam = bool(topo.get("add_support_beam", True))
        else:
            # Random topology initialization
            self.spacing_idx = int(rng.integers(0, len(self.spacing_options)))
            if self.layout_strategy == "research":
                self.num_secondary_width = self._sample_int(rng, self.num_secondary_range)
                self.num_secondary_height = self._sample_int(rng, self.num_secondary_range)
            else:
                self.support_spacing_idx = int(rng.integers(0, len(self.support_spacing_options)))
            self.add_support_beam = bool(rng.choice([True, False]))

        # Geometry
        if self.fixed_geometry is not None:
            self.L, self.W, self.H = self.fixed_geometry
        elif options.get("geometry") is not None:
            self.L, self.W, self.H = options["geometry"]
        else:
            self.L, self.W, self.H = self._sample_geometry(rng)

        # pressure
        if options.get("pressure") is not None:
            self.pressure = options["pressure"]
        else:
            self.pressure = rng.integers(20, 41) * 1000.0  # 20kPa to 40kPa

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
            "topology": self._topology_info(),
        }
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action and return next state.

        Args:
            action: np.ndarray of shape (8,) from MultiDiscrete action space
        """
        # Decode action
        section_deltas, topology_deltas, support_toggle = self.decode_action(action)

        # Apply section changes
        new_idx = self.idx + section_deltas
        # Clip to valid range (should be valid due to masking, but safety check)
        new_idx = np.clip(new_idx, 0, self.max_indices)
        self.idx = new_idx

        # Apply topology changes
        new_spacing_idx = self.spacing_idx + topology_deltas[0]
        new_spacing_idx = int(np.clip(new_spacing_idx, 0, len(self.spacing_options) - 1))
        self.spacing_idx = new_spacing_idx

        if self.layout_strategy == "research":
            new_num_w = self.num_secondary_width + topology_deltas[1]
            new_num_w = int(np.clip(new_num_w, self.num_secondary_range[0], self.num_secondary_range[1]))
            self.num_secondary_width = new_num_w

            new_num_h = self.num_secondary_height + topology_deltas[2]
            new_num_h = int(np.clip(new_num_h, self.num_secondary_range[0], self.num_secondary_range[1]))
            self.num_secondary_height = new_num_h
        else:
            new_support_idx = self.support_spacing_idx + topology_deltas[1]
            self.support_spacing_idx = int(np.clip(new_support_idx, 0, len(self.support_spacing_options) - 1))

        if support_toggle == 1:
            self.add_support_beam = not self.add_support_beam

        self.t += 1

        # Evaluate
        metrics = self._evaluate()
        self._current_metrics = metrics

        u_sigma = float(metrics["u_sigma"])
        u_delta = float(metrics["u_delta"])
        v = self._violation(u_sigma, u_delta)
        m = float(metrics["mass"])
        mnorm = float(m / (self.m0 + 1e-9))

        feasible_now = v <= 0.0

        k_vio = 1.0
        k_mass = 4

        def calculate_potential(v: float, mnorm: float) -> float:
            return k_vio * v + k_mass * mnorm

        phi_prev = calculate_potential(self.prev_v, self.prev_mnorm)
        phi_curr = calculate_potential(v, mnorm)

        # Reward = (Potential Decrease)
        r = phi_prev - phi_curr

        # Penalty for no-op (all deltas are zero and no toggle)
        is_noop = np.all(section_deltas == 0) and np.all(topology_deltas == 0) and support_toggle == 0
        if is_noop:
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
            "topology": self._topology_info(),
        }

        return obs, float(r), terminated, truncated, info

    @staticmethod
    def _int_range(specs: dict, name: str, default: tuple[int, int]) -> tuple[int, int]:
        spec = specs.get(name)
        if spec is None:
            return default
        return int(spec.low), int(spec.high)

    @staticmethod
    def _sample_int(rng: np.random.Generator, value_range: tuple[int, int]) -> int:
        return int(rng.integers(value_range[0], value_range[1] + 1))

    @staticmethod
    def _norm_index(index: int, count: int) -> float:
        return float(index / max(count - 1, 1))

    @staticmethod
    def _norm_int(value: int, value_range: tuple[int, int]) -> float:
        low, high = value_range
        return float((value - low) / max(high - low, 1))

    @staticmethod
    def _choice_idx(
        choices: np.ndarray,
        value,
        index,
        rng: np.random.Generator,
    ) -> int:
        if index is not None:
            return int(np.clip(index, 0, len(choices) - 1))
        if value is None:
            return int(rng.integers(0, len(choices)))
        return int(np.argmin(np.abs(choices - float(value))))

    def _topology_for_design(self) -> dict:
        topology = {
            "main_beam_spacing": self.main_beam_spacing,
            "add_support_beam": self.add_support_beam,
        }
        if self.layout_strategy == "research":
            topology.update(
                {
                    "num_secondary_width": self.num_secondary_width,
                    "num_secondary_height": self.num_secondary_height,
                }
            )
        else:
            topology["support_beam_max_spacing"] = self.support_beam_max_spacing
        return topology

    def _topology_info(self) -> dict:
        if self.layout_strategy == "research":
            return {
                "spacing": self.main_beam_spacing,
                "num_secondary_width": self.num_secondary_width,
                "num_secondary_height": self.num_secondary_height,
                "add_support_beam": self.add_support_beam,
            }
        return {
            "main_beam_spacing": self.main_beam_spacing,
            "support_beam_max_spacing": self.support_beam_max_spacing,
            "add_support_beam": self.add_support_beam,
        }

    def render(self):
        pass

    def close(self):
        pass
