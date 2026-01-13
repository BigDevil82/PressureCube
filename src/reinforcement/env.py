import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from src.analysis.opensees_analysis import AnalysisResults, OpenSeesSolver
from src.analysis.post_processor import ResultProcessor
from src.graph.structural_graph import BeamAsNodeGraph
from src.structure.mesh_generator import MeshGenerator
from src.structure.section_library import SectionLibrary, SectionType
from src.structure.structural_model import BeamResponse, BeamType, StructuralModel, StructuralParameters
from src.utils.logger import Logger


class StructuralEnv(gym.Env):
    """
    三维钢框架结构优化环境

    Observation:
        PyG Data 对象 (Beam-as-Node graph)，包含几何、物理属性和当前的应力/位移状态。

    Action:
        Box(8,), 范围 [-1, 1]。映射到 8 个设计变量。
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # 1. 初始化资源库 (只加载一次)
        self.section_lib = SectionLibrary("src/data/steel_sections_large.json")
        self.logger = Logger("structural_env.log", disable=False)
        self.sec_counts = [
            self.section_lib.section_counts[t]
            for t in [SectionType.BOX, SectionType.H_MAIN, SectionType.H_SEC, SectionType.CHANNEL]
        ]

        # 2. 定义动作空间 (归一化到 [-1, 1] 以便于 RL 训练)
        # 0: main_spacing (float)
        # 1: num_sec_width (int)
        # 2: num_sec_height (int)
        # 3: add_support (bool > 0)
        # 4-7: section indices (int)

        self.action_space = gym.spaces.MultiDiscrete(self.sec_counts)

        # 3. 定义观察空间
        # 由于 PyG 图结构大小可变，标准 Gym Space 难以描述。
        # 这里为了兼容性定义为一个空的 Dict，实际上 step 会返回 Data 对象
        self.observation_space = spaces.Dict({})

        # 增量 reward 超参数
        self.delta_scale = float(self.config.get("delta_scale", 20.0))  # 放大奖励，学习更稳
        self.infeasible_penalty = float(self.config.get("infeasible_penalty", -1.0))
        self.noop_penalty = float(self.config.get("noop_penalty", -0.01))

        # 状态缓存
        self.current_design_model = None
        self.current_fem_model = None
        self.last_obs = None
        self.last_info = None
        self.prev_mass = None  # 新增：记录上一步的质量

        # === Reward hyper-params (can be overridden by env_config) ===
        self.target_ratio = float(self.config.get("target_ratio", 0.95))  # 想让 max ratio 贴近的目标
        self.w_mass = float(self.config.get("w_mass", 1.0))  # 质量项权重
        self.w_delta = float(self.config.get("w_delta", 0.3))  # “本步比上步更轻”的奖励
        self.w_slack = float(self.config.get("w_slack", 2.0))  # 可行域内“太保守”的惩罚
        self.w_viol = float(self.config.get("w_viol", 50.0))  # 违约惩罚（平方）
        self.feasible_bonus = float(self.config.get("feasible_bonus", 0.5))
        self.infeasible_base = float(self.config.get("infeasible_base", -1.0))
        self.clip_reward = float(self.config.get("clip_reward", 5.0))

    def _sync_indices_to_params(self):
        self.current_params.frame_section_index = int(self.current_section_indices[0])
        self.current_params.main_beam_section_index = int(self.current_section_indices[1])
        self.current_params.secondary_beam_section_index = int(self.current_section_indices[2])
        self.current_params.brace_section_index = int(self.current_section_indices[3])

    def _mass_norm(self, mass_kg: float) -> float:
        min_mass = self.min_volume * 7850.0
        max_mass = self.max_volume * 7850.0
        return float((mass_kg - min_mass) / (max_mass - min_mass + 1e-6))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}

        rng = self.np_random

        if "init_sections" in options and options["init_sections"] is not None:
            self.current_section_indices = np.asarray(options["init_sections"], dtype=np.int64)
        else:
            self.current_section_indices = np.array([c - 1 for c in self.sec_counts], dtype=np.int64)

        self.current_params = StructuralParameters(
            length=float(rng.choice(np.arange(6.0, 12.0, 0.2))),
            width=float(rng.choice(np.arange(3.0, 4.5, 0.3))),
            height=float(rng.choice(np.arange(3.0, 4.5, 0.3))),
            main_beam_spacing=0.7,
            num_secondary_width=4,
            num_secondary_height=4,
            add_support_beam=True,
            frame_section_index=int(self.current_section_indices[0]),
            main_beam_section_index=int(self.current_section_indices[1]),
            secondary_beam_section_index=int(self.current_section_indices[2]),
            brace_section_index=int(self.current_section_indices[3]),
        )

        if "pressure" in options and options["pressure"] is not None:
            self.pressure = float(options["pressure"])
        else:
            self.pressure = float(rng.choice(np.arange(20000.0, 40000.0, 4000.0)))

        self._calculate_volume_bounds()

        # 执行完整的建模-分析-构图流程
        obs, info = self._run_simulation()
        self.prev_mass = info["mass"]  # 初始化质量记录
        self.stress_ratio_prev = info["max_stress_ratio"]
        self.disp_ratio_prev = info["max_disp_ratio"]
        self.last_obs = obs
        self.last_info = info

        return obs, info

    def step(self, action):
        # gymnasium 可能传 numpy 标量
        if isinstance(action, (np.ndarray,)):
            action = int(action.item())
        else:
            action = int(action)

        # STOP：直接结束回合（不给额外奖励，已通过前面 shrink 累积获得）
        if action == 4:
            return self.last_obs, 0.0, True, False, self.last_info

        # 备份旧状态（避免 infeasible 时再跑一次仿真）
        old_indices = self.current_section_indices.copy()
        old_obs = self.last_obs
        old_info = self.last_info
        old_mass_norm = self._mass_norm(float(self.prev_mass))

        # 选到已经不能减的类型：小惩罚，不终止
        if self.current_section_indices[action] <= 0:
            return self.last_obs, self.noop_penalty, False, False, self.last_info

        # 提议 shrink
        self.current_section_indices[action] -= 1
        self._sync_indices_to_params()

        obs, info = self._run_simulation()
        sr = float(info["max_stress_ratio"])
        dr = float(info["max_disp_ratio"])
        feasible = max(sr, dr) <= 1.0

        if not feasible:
            # 回滚并终止：表示触碰约束边界
            self.current_section_indices = old_indices
            self._sync_indices_to_params()
            return old_obs, self.infeasible_penalty, True, False, old_info

        # 可行：奖励 = 变轻的归一化增量（放大 delta_scale）
        new_mass = float(info["mass"])
        new_mass_norm = self._mass_norm(new_mass)
        reward = self.delta_scale * (old_mass_norm - new_mass_norm)

        # 更新缓存
        self.prev_mass = new_mass
        self.last_obs = obs
        self.last_info = info

        return obs, float(reward), False, False, info

    def _apply_action_to_params(self, action):
        """动作是 4 个离散 index：frame/main/sec/brace"""
        max_indices = np.array(self.sec_counts, dtype=np.int64) - 1

        action = np.asarray(action).reshape(-1)
        if action.shape[0] != 4:
            raise ValueError(f"Action must have 4 dims, got {action.shape}")

        action = action.astype(np.int64)
        action = np.clip(action, 0, max_indices).astype(np.int64)

        frame_idx, main_idx, sec_idx, brace_idx = action.tolist()

        self.current_params.frame_section_index = frame_idx
        self.current_params.main_beam_section_index = main_idx
        self.current_params.secondary_beam_section_index = sec_idx
        self.current_params.brace_section_index = brace_idx

        # 1. 几何参数
        # self.current_params.main_beam_spacing = map_val(act[0], 0.75, 1.5)  # 0.75m ~ 1.5m
        # self.current_params.num_secondary_width = int(map_val(act[1], 2.0, 6.0))
        # self.current_params.num_secondary_height = int(map_val(act[2], 2.0, 6.0))
        # self.current_params.add_support_beam = act[3] > 0

    def _run_simulation(self):
        """核心流程：建模 -> 网格 -> 分析 -> 结果映射 -> 图生成"""

        # A. 生成粗模型 (Design Model)
        self.current_design_model = StructuralModel(self.current_params, self.section_lib)
        self.current_design_model.generate()

        # B. 生成细网格 (FEM Model)
        # 注意：这里需要 MeshGenerator 记录映射关系，以便后续回传结果
        mesh_gen = MeshGenerator(self.current_design_model, max_mesh_size=0.5)
        self.current_fem_model = mesh_gen.generate()

        # C. OpenSees 分析
        solver = OpenSeesSolver(self.current_fem_model)

        # 1. 定义荷载组合
        combo_strength = {"DL": 1.3, "LL": 1.5, "PL": 1.25}  # 承载能力极限状态
        combo_service = {"DL": 1.0, "LL": 1.0, "PL": 1.0}  # 正常使用极限状态

        # 2. 直接运行荷载组合分析
        res_strength = solver.run_load_combination(combo_strength, pressure_pa=self.pressure)
        res_service = solver.run_load_combination(combo_service, pressure_pa=self.pressure)

        # D. 规范校核 (获取 Ratio)
        # 关键步骤：将 FEM 结果映射回 Design Model
        self._map_results_to_design(res_strength, res_service)

        # E. 生成图 (Observation)
        # StructuralGraph 会读取 beam.response 来构建特征
        graph_gen = BeamAsNodeGraph(self.current_design_model, current_pressure=self.pressure)
        obs_data = graph_gen.to_pyg_data()

        # F. 收集 Info (用于 Reward)
        total_mass = self.current_design_model.calculate_mass_kg()  # 粗模型计算质量即可

        # 统计整体违约情况
        max_stress_r = 0
        max_disp_r = 0
        for b in self.current_design_model.beams.values():
            if b.response:
                max_stress_r = max(max_stress_r, b.response.max_stress_ratio)
                max_disp_r = max(max_disp_r, b.response.max_disp_ratio)

        info = {
            "mass": total_mass,
            "max_stress_ratio": max_stress_r,
            "max_disp_ratio": max_disp_r,
            "analysis_failed": False,
        }
        if max_disp_r > 50:
            self.log_state()

        return obs_data, info

    def _map_results_to_design(self, res_strength: AnalysisResults, res_service: AnalysisResults):
        """
        将 FEM 单元的分析结果（应力、位移）聚合回原始梁。
        聚合策略：取子单元中的最大值 (Conservative)。
        """
        # 1. 实例化后处理器用于计算 Ratio (不进行全盘 Check，只利用其计算逻辑)
        processor = ResultProcessor(self.current_fem_model)

        # 初始化聚合器
        agg_stress = {}  # old_id -> max_ratio
        agg_disp = {}  # old_id -> max_ratio

        # 遍历 FEM 模型的所有梁
        for fem_bid, fem_beam in self.current_fem_model.beams.items():
            # 这里假设我们通过某种方式知道它属于哪个原始梁
            # 为了实现方便，建议在 MeshGenerator._mesh_single_beam 中：
            # new_beam.tag = beam.id (利用 tag 或新字段存储原始ID)
            parent_id = fem_beam.parent_id

            if parent_id is not None:
                # 计算该 FEM 单元的 Stress Ratio
                # 这里直接调用 Processor 的单梁校核逻辑 (需要稍微拆解 Processor)
                # 简化：直接取内力算应力
                forces = res_strength.beam_forces.get(fem_bid)
                if forces is not None:
                    ratio, _, _, _ = processor._check_single_beam_stress(fem_bid, forces)
                    agg_stress[parent_id] = max(agg_stress.get(parent_id, 0), ratio)

                # 计算位移 (取两个节点的平均或最大)
                d1 = res_service.displacements.get(fem_beam.start_node, np.zeros(6))
                d2 = res_service.displacements.get(fem_beam.end_node, np.zeros(6))
                disp_val = max(np.linalg.norm(d1[:3]), np.linalg.norm(d2[:3]))

                # 归一化位移 (L/400) - 这里 L 取主梁间距作为参考
                limit = self.current_params.width * 1000 / 250.0
                agg_disp[parent_id] = max(agg_disp.get(parent_id, 0), disp_val / limit)

        # 将聚合结果写回 Design Model
        for bid, beam in self.current_design_model.beams.items():
            s_ratio = agg_stress.get(bid, 0.0)
            d_ratio = agg_disp.get(bid, 0.0)

            beam.response = BeamResponse(
                max_stress_ratio=s_ratio, max_disp_ratio=d_ratio, is_violated=(s_ratio > 1.0 or d_ratio > 1.0)
            )

    def _calculate_reward(self, info):
        """
        目标：在满足 sr<=1 & dr<=1 的前提下尽量减重，同时避免“过度保守”(ratio 长期 0.3)
        """
        mass = float(info["mass"])
        sr = float(info["max_stress_ratio"])
        dr = float(info["max_disp_ratio"])

        # --- 1) 质量归一化（0=最轻, 1=最重），并转成“越轻越大”的 reward_mass ---
        min_mass = self.min_volume * 7850.0
        max_mass = self.max_volume * 7850.0
        denom = (max_mass - min_mass) + 1e-6
        mass_norm = (mass - min_mass) / denom
        mass_norm = float(np.clip(mass_norm, 0.0, 1.0))
        reward_mass = 1.0 - mass_norm  # [0,1]，越轻越大

        # --- 2) 违约惩罚（平方，既能压住大违约，又不至于对小违约“过度恐惧”）---
        viol_s = max(0.0, sr - 1.0)
        viol_d = max(0.0, dr - 1.0)
        penalty_viol = self.w_viol * (viol_s * viol_s + viol_d * viol_d)

        feasible = (sr <= 1.0) and (dr <= 1.0)

        # --- 3) 可行域“贴边”惩罚：ratio 太小 => 罚（促使从 0.3 往 0.9~1.0 靠近）---
        # 只在可行域内启用，避免违约时策略被“贴边惩罚”误导
        if feasible:
            slack_s = max(0.0, self.target_ratio - sr)
            slack_d = max(0.0, self.target_ratio - dr)
            penalty_slack = self.w_slack * (slack_s * slack_s + slack_d * slack_d)
        else:
            penalty_slack = 0.0

        # --- 4) 逐步减重奖励：鼓励在 20-step 内“越改越轻”（你 env 里已经有 prev_mass 缓存）---
        if self.prev_mass is None:
            delta_norm = 0.0
        else:
            delta_norm = (float(self.prev_mass) - mass) / denom  # >0 表示变轻
            delta_norm = float(np.clip(delta_norm, -1.0, 1.0))

        reward_delta = self.w_delta * delta_norm

        # --- 5) 合成 ---
        if feasible:
            reward = self.w_mass * reward_mass + reward_delta + self.feasible_bonus - penalty_slack
        else:
            # 不可行：给一个基础负分 + 违约惩罚（保留质量项，避免“摆烂重结构”）
            reward = self.w_mass * reward_mass + reward_delta + self.infeasible_base - penalty_viol

        # 更新缓存
        self.prev_mass = mass

        # 防止极端值影响 PPO 稳定性
        reward = float(np.clip(reward, -self.clip_reward, self.clip_reward))

        self.logger.info(
            f"Mass={mass:.0f}, sr={sr:.3f}, dr={dr:.3f}, "
            f"R_mass={reward_mass:.3f}, dM={delta_norm:.3f}, "
            f"feas={feasible}, Reward={reward:.3f}"
        )
        return reward

    def _calculate_volume_bounds(self):
        """预计算最大最小材料用量，用于归一化"""
        # 最大体积：所有主次梁都用最大截面
        params_max = StructuralParameters(
            length=self.current_params.length,
            width=self.current_params.width,
            height=self.current_params.height,
            main_beam_spacing=self.current_params.main_beam_spacing,
            num_secondary_width=self.current_params.num_secondary_width,
            num_secondary_height=self.current_params.num_secondary_height,
            add_support_beam=self.current_params.add_support_beam,
            frame_section_index=self.section_lib.section_counts[SectionType.BOX] - 1,
            main_beam_section_index=self.section_lib.section_counts[SectionType.H_MAIN] - 1,
            secondary_beam_section_index=self.section_lib.section_counts[SectionType.H_SEC] - 1,
            brace_section_index=self.section_lib.section_counts[SectionType.CHANNEL] - 1,
        )
        model_max = StructuralModel(params_max, self.section_lib)
        model_max.generate()
        self.max_volume = model_max.calculate_mass_kg() / 7850.0

        # 最小体积：所有主次梁都用最小截面
        params_min = StructuralParameters(
            length=self.current_params.length,
            width=self.current_params.width,
            height=self.current_params.height,
            main_beam_spacing=self.current_params.main_beam_spacing,
            num_secondary_width=self.current_params.num_secondary_width,
            num_secondary_height=self.current_params.num_secondary_height,
            add_support_beam=self.current_params.add_support_beam,
            frame_section_index=0,
            main_beam_section_index=0,
            secondary_beam_section_index=0,
            brace_section_index=0,
        )
        model_min = StructuralModel(params_min, self.section_lib)
        model_min.generate()
        self.min_volume = model_min.calculate_mass_kg() / 7850.0

    def _calculate_normalized_volume(self, model_volume: float) -> float:
        """计算归一化的材料用量"""
        return (model_volume - self.min_volume) / (self.max_volume - self.min_volume)

    def log_state(self):
        p = self.current_params
        print(f"{'='*20} Current Design Parameters {'='*20}")
        print(
            f"Length: {p.length:.2f} m, Width: {p.width:.2f} m, Height: {p.height:.2f} m, Pressure: {self.pressure:.1f} Pa"
        )
        print(
            # f"{p.main_beam_spacing=:.3f} m, "
            # f"{p.num_secondary_width=}, {p.num_secondary_height=}, "
            # f"{p.add_support_beam=}, "
            f"{p.frame_section_index}, {p.main_beam_section_index}, "
            f"{p.secondary_beam_section_index}, {p.brace_section_index}"
        )


if __name__ == "__main__":
    # 简单测试环境
    env = StructuralEnv(config={})
    obs, info = env.reset()
    print("Info after reset:\n", info)
