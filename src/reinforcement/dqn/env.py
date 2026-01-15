import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.analysis.opensees_analysis import AnalysisResults, OpenSeesSolver
from src.analysis.post_processor import ResultProcessor
from src.structure.enums import BeamType
from src.structure.mesh_generator import MeshGenerator
from src.structure.section_library import SectionLibrary, SectionType
from src.structure.structural_model import BeamResponse, StructuralModel, StructuralParameters
from src.utils.logger import Logger


class SteelSectionOptEnv(gym.Env):
    """
    离散RL版本：只优化 4 类构件截面（frame/main/sec/brace）
    动作：81 个并行微调 (Δf,Δm,Δs,Δb), Δ∈{-1,0,+1}
    观测：固定维度向量（便于DQN/MLP）
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # --- libraries / logger (只加载一次) ---
        self.section_lib = SectionLibrary("src/data/steel_sections_large.json")

        # 4类截面数量（你的 env.py 就是这么取的）  :contentReference[oaicite:1]{index=1}
        self.sec_types = [SectionType.BOX, SectionType.H_MAIN, SectionType.H_SEC, SectionType.CHANNEL]
        self.sec_counts = [self.section_lib.section_counts[t] for t in self.sec_types]
        self.max_indices = np.array(self.sec_counts, dtype=np.int64) - 1  # 每类最大 index

        # --- action space: 81 discrete actions ---
        self.n_actions = 81
        self.action_space = spaces.Discrete(self.n_actions)

        # --- observation space: 12-dim vector (float32) ---
        # [L/12, W/4.5, H/4.5,
        #  i_f/imax_f, i_m/imax_m, i_s/imax_s, i_b/imax_b,
        #  u_sigma, u_delta,
        #  m/m0,
        #  argmax_sigma/3, argmax_delta/3]
        self.obs_dim = 12
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # --- hyperparams ---
        self.T = int(self.config.get("T", 30))
        self.K_no_improve = int(self.config.get("K_no_improve", 5))
        self.eps_improve = float(self.config.get("eps_improve", 1e-3))

        # reward params (推荐的两段增量式)
        self.alpha = float(self.config.get("alpha", 20.0))  # 违约改善权重
        self.beta = float(self.config.get("beta", 1.0))  # 质量增量权重
        self.c_feas = float(self.config.get("c_feas", 1.0))  # 可行域常数奖励
        self.penalty_feas_to_infeas = float(self.config.get("penalty_feas_to_infeas", 2.0))
        self.penalty_noop = float(self.config.get("penalty_noop", 0.05))

        # --- runtime cache ---
        self.current_params: StructuralParameters | None = None
        self.current_design_model: StructuralModel | None = None
        self.current_fem_model = None

        self.pressure = None
        self.idx = None  # (4,) 当前截面 index
        self.m0 = None  # reset时“全最大截面”质量
        self.prev_v = None
        self.prev_mnorm = None
        self.no_improve = 0
        self.t = 0

    # ---------------- action encoding ----------------
    @staticmethod
    def decode_action(a: int) -> np.ndarray:
        """0..80 -> 4 digits base-3 -> Δ∈{-1,0,+1}^4"""
        ds = []
        x = int(a)
        for _ in range(4):
            ds.append(x % 3)
            x //= 3
        ds = ds[::-1]
        return np.array([d - 1 for d in ds], dtype=np.int64)

    @staticmethod
    def encode_action(deltas: np.ndarray) -> int:
        digits = [int(d + 1) for d in deltas]  # -1,0,1 -> 0,1,2
        a = 0
        for dig in digits:
            a = a * 3 + dig
        return int(a)

    def _get_action_mask(self) -> np.ndarray:
        """当前 idx 下，哪些动作不会越界"""
        mask = np.ones(self.n_actions, dtype=bool)
        for a in range(self.n_actions):
            d = self.decode_action(a)
            new_idx = self.idx + d
            if np.any(new_idx < 0) or np.any(new_idx > self.max_indices):
                mask[a] = False
        return mask

    # ---------------- reset sampling ----------------
    def _sample_geometry(self, rng: np.random.Generator):
        L = float(rng.choice(np.arange(6.0, 12.0, 0.2)))
        W = float(rng.choice(np.arange(3.0, 4.5, 0.3)))
        H = float(rng.choice(np.arange(3.0, 4.5, 0.3)))
        return L, W, H

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        rng = np.random.default_rng(seed) if seed is not None else self.np_random

        # 1) 截面初始：默认全最大
        if options.get("init_sections") is not None:
            self.idx = np.asarray(options["init_sections"], dtype=np.int64).copy()
        else:
            self.idx = self.max_indices.copy()

        # 2) 几何随机（你现有的参数化方式） :contentReference[oaicite:2]{index=2}
        L, W, H = self._sample_geometry(rng)

        # 3) 荷载：默认30kPa；也可以按你的旧env随机（20~40kPa） :contentReference[oaicite:3]{index=3}
        if options.get("pressure") is not None:
            self.pressure = float(options["pressure"])
        else:
            if bool(self.config.get("random_pressure", False)):
                self.pressure = float(rng.choice(np.arange(20000.0, 40000.0, 4000.0)))
            else:
                self.pressure = float(self.config.get("default_pressure", 30000.0))

        # 4) 固定拓扑控制参数（按你现在的设定）
        self.current_params = StructuralParameters(
            length=L,
            width=W,
            height=H,
            main_beam_spacing=float(self.config.get("main_beam_spacing", 0.7)),
            num_secondary_width=int(self.config.get("num_secondary_width", 4)),
            num_secondary_height=int(self.config.get("num_secondary_height", 4)),
            add_support_beam=bool(self.config.get("add_support_beam", True)),
            frame_section_index=int(self.idx[0]),
            main_beam_section_index=int(self.idx[1]),
            secondary_beam_section_index=int(self.idx[2]),
            brace_section_index=int(self.idx[3]),
        )

        # 5) 跑一次仿真，得到基准 m0、利用率
        out = self._run_simulation_and_collect()

        self.m0 = float(out["mass"])
        self.prev_v = self._violation(out["u_sigma"], out["u_delta"])
        self.prev_mnorm = float(out["mass"] / (self.m0 + 1e-9))
        self.no_improve = 0
        self.t = 0

        obs = self._build_obs(out)
        info = {
            **out,
            "action_mask": self._get_action_mask(),
        }
        return obs, info

    def step(self, action):
        # 1) 解码动作
        a = int(action)
        deltas = self.decode_action(a)

        # 2) 如果动作非法（越界），给惩罚并不更新（也可以直接mask避免发生）
        new_idx = self.idx + deltas
        if np.any(new_idx < 0) or np.any(new_idx > self.max_indices):
            obs = self._build_obs(self._run_simulation_and_collect())  # 不改也重新取一次状态
            info = {"action_mask": self._get_action_mask(), "illegal_action": True}
            return obs, -1.0, False, False, info

        # 3) 应用动作：更新截面 index + 写回 params
        self.idx = new_idx
        self.current_params.frame_section_index = int(self.idx[0])
        self.current_params.main_beam_section_index = int(self.idx[1])
        self.current_params.secondary_beam_section_index = int(self.idx[2])
        self.current_params.brace_section_index = int(self.idx[3])

        self.t += 1

        # 4) 分析 + 指标
        out = self._run_simulation_and_collect()

        u_sigma = float(out["u_sigma"])
        u_delta = float(out["u_delta"])
        v = self._violation(u_sigma, u_delta)

        m = float(out["mass"])
        mnorm = float(m / (self.m0 + 1e-9))

        feasible_prev = self.prev_v <= 0.0
        feasible_now = v <= 0.0

        # 5) 奖励（两段增量）
        if v > 0.0:
            r = self.alpha * (self.prev_v - v) - self.beta * (mnorm - self.prev_mnorm)
        else:
            r = self.c_feas - self.beta * (mnorm - self.prev_mnorm)

        if feasible_prev and (not feasible_now):
            r -= self.penalty_feas_to_infeas

        if np.all(deltas == 0):
            r -= self.penalty_noop

        # 6) early stop：可行 & 无改进
        if feasible_now and (mnorm >= self.prev_mnorm - self.eps_improve):
            self.no_improve += 1
        else:
            self.no_improve = 0

        terminated = (self.t >= self.T) or (feasible_now and self.no_improve >= self.K_no_improve)
        truncated = False

        # 7) 更新缓存
        self.prev_v = v
        self.prev_mnorm = mnorm

        obs = self._build_obs(out)
        info = {
            **out,
            "mnorm": mnorm,
            "violation": v,
            "action_mask": self._get_action_mask(),
        }
        return obs, float(r), terminated, truncated, info

    # ---------------- helper: metrics / obs ----------------
    @staticmethod
    def _violation(u_sigma: float, u_delta: float) -> float:
        return max(0.0, u_sigma - 1.0) + max(0.0, u_delta - 1.0)

    def _build_obs(self, out: dict) -> np.ndarray:
        p = self.current_params
        # 注意：不同类型的imax可能不一样，所以分别归一化
        i_norm = np.array(
            [
                self.idx[0] / max(1, self.max_indices[0]),
                self.idx[1] / max(1, self.max_indices[1]),
                self.idx[2] / max(1, self.max_indices[2]),
                self.idx[3] / max(1, self.max_indices[3]),
            ],
            dtype=np.float32,
        )

        obs = np.array(
            [
                float(p.length / 12.0),
                float(p.width / 4.5),
                float(p.height / 4.5),
                float(i_norm[0]),
                float(i_norm[1]),
                float(i_norm[2]),
                float(i_norm[3]),
                float(out["u_sigma"]),
                float(out["u_delta"]),
                float(out["mass"] / (self.m0 + 1e-9)),
                float(out["argmax_sigma"] / 3.0),
                float(out["argmax_delta"] / 3.0),
            ],
            dtype=np.float32,
        )
        return obs

    # ---------------- core: reuse your simulation pipeline ----------------
    def _run_simulation_and_collect(self) -> dict:
        """
        复用你 env.py 的核心流程：
        A DesignModel -> B FEM mesh -> C OpenSees -> D map back -> (E graph 可省略) -> F info
        """
        # A. Design model
        self.current_design_model = StructuralModel(self.current_params, self.section_lib)
        self.current_design_model.generate()

        # B. FEM mesh
        mesh_gen = MeshGenerator(self.current_design_model, max_mesh_size=0.5)
        self.current_fem_model = mesh_gen.generate()

        # C. OpenSees analysis (2 combinations)
        solver = OpenSeesSolver(self.current_fem_model)
        combo_strength = {"DL": 1.3, "LL": 1.5, "PL": 1.25}
        combo_service = {"DL": 1.0, "LL": 1.0, "PL": 1.0}
        res_strength = solver.run_load_combination(combo_strength, pressure_pa=self.pressure)
        res_service = solver.run_load_combination(combo_service, pressure_pa=self.pressure)

        # D. Map back to design
        self._map_results_to_design(res_strength, res_service)

        # F. Collect metrics
        total_mass = float(self.current_design_model.calculate_mass_kg())

        # 统计：全结构 max ratio + 控制类别（按你的 BeamResponse 聚合逻辑） :contentReference[oaicite:4]{index=4}
        max_stress = 0.0
        max_disp = 0.0

        type_max_s = np.zeros(4, dtype=np.float32)
        type_max_d = np.zeros(4, dtype=np.float32)

        for b in self.current_design_model.beams.values():
            if b.response is None:
                continue
            sr = float(b.response.max_stress_ratio)
            dr = float(b.response.max_disp_ratio)
            max_stress = max(max_stress, sr)
            max_disp = max(max_disp, dr)

            if b.beam_type == BeamType.FRAME:
                k = 0
            elif b.beam_type == BeamType.MAIN:
                k = 1
            elif b.beam_type == BeamType.SECONDARY:
                k = 2
            elif b.beam_type == BeamType.BRACE:
                k = 3
            else:
                raise ValueError(f"Unknown beam type: {b.beam_type}")

            type_max_s[k] = max(type_max_s[k], sr)
            type_max_d[k] = max(type_max_d[k], dr)

        argmax_sigma = int(type_max_s.argmax()) if type_max_s.max() > 0 else 0
        argmax_delta = int(type_max_d.argmax()) if type_max_d.max() > 0 else 0

        return {
            "mass": total_mass,
            "u_sigma": float(max_stress),
            "u_delta": float(max_disp),
            "argmax_sigma": argmax_sigma,
            "argmax_delta": argmax_delta,
            "pressure": float(self.pressure),
        }

    def _map_results_to_design(self, res_strength: AnalysisResults, res_service: AnalysisResults):
        """
        直接复制/等价实现你 env.py 里的聚合逻辑：单元结果 -> parent beam
        你原实现里使用 parent_id + processor._check_single_beam_stress + 位移比 L/250
        :contentReference[oaicite:5]{index=5}
        """
        processor = ResultProcessor(self.current_fem_model)

        agg_stress = {}
        agg_disp = {}

        for fem_bid, fem_beam in self.current_fem_model.beams.items():
            parent_id = getattr(fem_beam, "parent_id", None)
            if parent_id is None:
                continue

            forces = res_strength.beam_forces.get(fem_bid)
            if forces is not None:
                ratio, _, _, _ = processor._check_single_beam_stress(fem_bid, forces)
                agg_stress[parent_id] = max(agg_stress.get(parent_id, 0.0), float(ratio))

            d1 = res_service.displacements.get(fem_beam.start_node, np.zeros(6))
            d2 = res_service.displacements.get(fem_beam.end_node, np.zeros(6))
            disp_val = max(np.linalg.norm(d1[:3]), np.linalg.norm(d2[:3]))

            # L/250：L取“构件长度”（你现有做法就是这样） :contentReference[oaicite:6]{index=6}
            # L_mm = self.current_design_model.beam_manager.get_beam_length(parent_id) * 1000.0
            L_mm = self.current_params.width * 1000.0  # 你的 env.py 里是这么取的
            limit = L_mm / 250.0
            agg_disp[parent_id] = max(agg_disp.get(parent_id, 0.0), float(disp_val / (limit + 1e-12)))

        for bid, beam in self.current_design_model.beams.items():
            s_ratio = float(agg_stress.get(bid, 0.0))
            d_ratio = float(agg_disp.get(bid, 0.0))
            beam.response = BeamResponse(
                max_stress_ratio=s_ratio,
                max_disp_ratio=d_ratio,
                is_violated=(s_ratio > 1.0 or d_ratio > 1.0),
            )
