import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from src.analysis.opensees_analysis import AnalysisResults, OpenSeesSolver, ResultCombiner
from src.analysis.post_processor import ResultProcessor
from src.graph.structural_graph import BeamAsNodeGraph  # 假设这是上一轮重构后的 Beam-as-Node Graph
from src.structure.mesh_generator import MeshGenerator
from src.structure.section_library import SectionLibrary, SectionType

# 引入之前的模块
from src.structure.structural_model import BeamResponse  # 引入上面定义的类
from src.structure.structural_model import BeamType, StructuralModel, StructuralParameters


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
        self.section_lib = SectionLibrary("src/data/steel_sections.json")

        # 2. 定义动作空间 (归一化到 [-1, 1] 以便于 RL 训练)
        # 0: main_spacing (float)
        # 1: num_sec_width (int)
        # 2: num_sec_height (int)
        # 3: add_support (bool > 0)
        # 4-7: section indices (int)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

        # 3. 定义观察空间
        # 由于 PyG 图结构大小可变，标准 Gym Space 难以描述。
        # 这里为了兼容性定义为一个空的 Dict，实际上 step 会返回 Data 对象
        self.observation_space = spaces.Dict({})

        # 状态缓存
        self.current_design_model = None
        self.current_fem_model = None
        self.last_results = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 随机初始化或使用默认参数
        # 这里演示使用默认参数
        self.current_params = StructuralParameters(
            length=12.0,
            width=3.7,
            height=3.3,
            main_beam_spacing=1.5,
            num_secondary_width=3,
            num_secondary_height=3,
            add_support_beam=True,
            frame_section_index=0,
            main_beam_section_index=0,
            secondary_beam_section_index=0,
            brace_section_index=0,
        )

        # 执行完整的建模-分析-构图流程
        obs, _ = self._run_simulation()
        return obs, {}

    def step(self, action):
        # 1. Action -> Parameters
        self._apply_action_to_params(action)

        # 2. Run Simulation
        try:
            obs, info = self._run_simulation()

            # 3. Calculate Reward
            reward = self._calculate_reward(info)

            # 4. Check Termination
            terminated = False
            truncated = False

            # 可以在这里添加提前终止逻辑 (如结构极其不稳定导致分析失败)
            if info["analysis_failed"]:
                reward = -100.0  # 惩罚失败
                terminated = True

        except Exception as e:
            print(f"Simulation Error: {e}")
            obs = None
            reward = -100.0
            terminated = True
            truncated = False
            info = {}

        return obs, reward, terminated, truncated, info

    def _apply_action_to_params(self, action):
        """将 [-1, 1] 的动作映射回物理参数"""
        act = np.clip(action, -1.0, 1.0)

        # 辅助映射函数
        def map_val(val, min_v, max_v):
            return min_v + (val + 1.0) / 2.0 * (max_v - min_v)

        # 1. 几何参数
        self.current_params.main_beam_spacing = map_val(act[0], 0.75, 2.0)  # 0.75m ~ 2.0m
        self.current_params.num_secondary_width = int(map_val(act[1], 1.0, 6.0))
        self.current_params.num_secondary_height = int(map_val(act[2], 1.0, 6.0))
        self.current_params.add_support_beam = act[3] > 0

        # 2. 截面索引 (根据库的大小动态映射)
        # 假设库已经按照面积排序，index越大截面越大
        def map_idx(val, type_str):
            count = self.section_lib.section_counts[type_str]
            return int(map_val(val, 0, count - 0.99))

        self.current_params.frame_section_index = map_idx(act[4], SectionType.BOX)
        self.current_params.main_beam_section_index = map_idx(act[5], SectionType.H_MAIN)
        self.current_params.secondary_beam_section_index = map_idx(act[6], SectionType.H_SEC)
        self.current_params.brace_section_index = map_idx(act[7], SectionType.CHANNEL)

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

        # 运行工况 (DL, LL, PL)
        res_dl = solver.run_static_case("DL")
        if res_dl is None:
            return None, {"analysis_failed": True}
        res_ll = solver.run_static_case("LL")
        res_pl = solver.run_static_case("PL", pressure_pa=30000)

        # 组合结果
        combiner = ResultCombiner()
        res_strength = combiner.get_max_stress_combo(res_dl, res_ll, res_pl)  # 1.3DL + ...
        res_service = combiner.get_max_displacement_combo(res_dl, res_ll, res_pl)  # 1.0DL + ...

        # D. 规范校核 (获取 Ratio)
        # 我们需要一个稍作修改的 Processor，不仅返回 max，还能返回每个单元的 ratio
        # 这里为了简化，我们假设 Processor 可以辅助计算，或者我们手动计算映射

        # 关键步骤：将 FEM 结果映射回 Design Model
        self._map_results_to_design(mesh_gen, res_strength, res_service)

        # E. 生成图 (Observation)
        # StructuralGraph 会读取 beam.response 来构建特征
        graph_gen = BeamAsNodeGraph(self.current_design_model)
        obs_data = graph_gen.to_pyg_data()

        # F. 收集 Info (用于 Reward)
        total_mass = self.current_design_model.calculate_mass()  # 粗模型计算质量即可

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

        return obs_data, info

    def _map_results_to_design(
        self, mesh_gen: MeshGenerator, res_strength: AnalysisResults, res_service: AnalysisResults
    ):
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
            parent_id = getattr(fem_beam, "parent_id", None)

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
        奖励函数设计
        R = Weight_Reward + Penalty
        """
        # 1. 质量奖励 (归一化)
        # 假设基准质量 (全部最重截面) 约为 5000kg，最小约为 1000kg
        mass = info["mass"]
        ref_mass = 5000.0
        # 越轻越好，范围 [0, 1] 左右
        r_mass = (ref_mass - mass) / ref_mass

        # 2. 约束惩罚
        # 使用指数惩罚，稍微超限惩罚较小，严重超限惩罚巨大
        penalty = 0.0

        s_ratio = info["max_stress_ratio"]
        d_ratio = info["max_disp_ratio"]

        if s_ratio > 1.0:
            penalty += 10.0 * (s_ratio - 1.0) ** 2 + 2.0

        if d_ratio > 1.0:
            penalty += 10.0 * (d_ratio - 1.0) ** 2 + 2.0

        return r_mass - penalty
