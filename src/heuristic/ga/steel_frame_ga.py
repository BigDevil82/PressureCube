import copy
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from src.analysis.opensees_analysis import OpenSeesSolver
from src.analysis.post_processor import ResultProcessor
from src.structure.mesh_generator import MeshGenerator
from src.structure.section_library import SectionLibrary, SectionType
from src.structure.structural_model import StructuralModel, StructuralParameters
from src.utils.logger import Logger

# --- 配置常量 ---
POPULATION_SIZE = 30
MAX_GENERATIONS = 50
ELITE_SIZE = 2
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.8
REFERENCE_MASS = 10000.0

# 设计变量边界
SPACING_BOUNDS = {"main_min": 0.5, "main_max": 2.0, "brace_min": 0.5, "brace_max": 1.5}
# 假定截面库索引范围 (需根据实际库大小动态调整)
SEC_IDX_RANGES = [0, 0, 0, 0]  # [Frame, Main, Sec, Brace] 在 init 中更新


@dataclass
class Individual:
    """代表一个设计个体"""

    # 基因: [FrameIdx, MainIdx, SecIdx, BraceIdx, MainSpacing, BraceSpacing]
    genes: List[float]

    # 表现型 (评估结果)
    mass: float = float("inf")
    max_stress_ratio: float = 0.0
    max_deflection_ratio: float = 0.0
    fitness: float = 0.0
    is_feasible: bool = False

    def deep_copy(self):
        ind = Individual(copy.copy(self.genes))
        ind.mass = self.mass
        ind.max_stress_ratio = self.max_stress_ratio
        ind.max_deflection_ratio = self.max_deflection_ratio
        ind.fitness = self.fitness
        ind.is_feasible = self.is_feasible
        return ind


class SteelFrameGA:
    def __init__(self, base_params: StructuralParameters):
        self.base_params = base_params
        self.section_lib = SectionLibrary()
        self.best_history = []
        self.logger = Logger("SteelFrameGA")

        # 动态获取截面库大小
        counts = self.section_lib.section_counts
        # 对应顺序: Frame(BOX), Main(H_MAIN), Sec(H_SEC), Brace(C)
        global SEC_IDX_RANGES
        SEC_IDX_RANGES = [
            counts[SectionType.BOX] - 1,
            counts[SectionType.H_MAIN] - 1,
            counts[SectionType.H_SEC] - 1,
            counts[SectionType.CHANNEL] - 1,
        ]

    def _create_random_genes(self) -> List[float]:
        return [
            random.randint(0, SEC_IDX_RANGES[0]),  # Frame
            random.randint(0, SEC_IDX_RANGES[1]),  # Main
            random.randint(0, SEC_IDX_RANGES[2]),  # Sec
            random.randint(0, SEC_IDX_RANGES[3]),  # Brace
            # random.uniform(SPACING_BOUNDS["main_min"], SPACING_BOUNDS["main_max"]),
            # random.uniform(SPACING_BOUNDS["brace_min"], SPACING_BOUNDS["brace_max"]),
        ]

    def _create_heavy_genes(self) -> List[float]:
        """创建'最强'个体：最大截面，最小间距，确保有可行解"""
        return [
            SEC_IDX_RANGES[0],  # Max Frame
            SEC_IDX_RANGES[1],  # Max Main
            SEC_IDX_RANGES[2],  # Max Sec
            SEC_IDX_RANGES[3],  # Max Brace
            # SPACING_BOUNDS["main_min"],  # Min Spacing (Dense)
            # SPACING_BOUNDS["brace_min"],  # Min Spacing
        ]

    def _genes_to_params(self, genes: List[float]) -> StructuralParameters:
        """将基因解码为结构参数"""
        p = copy.deepcopy(self.base_params)
        p.frame_section_index = int(genes[0])
        p.main_beam_section_index = int(genes[1])
        p.secondary_beam_section_index = int(genes[2])
        p.brace_section_index = int(genes[3])

        # p.main_beam_spacing = genes[4]

        # # 将 brace_spacing 转换为数量 (int)
        # # width / spacing = segments
        # brace_spacing = genes[5]
        # p.num_secondary_width = max(1, int(round(p.width / brace_spacing)))
        # p.num_secondary_height = max(1, int(round(p.height / brace_spacing)))

        return p

    def evaluate(self, ind: Individual):
        """核心评估函数"""
        try:
            params = self._genes_to_params(ind.genes)
            model = StructuralModel(params, self.section_lib)
            model.generate()

            mesh_gen = MeshGenerator(model, max_mesh_size=0.6)
            meshed_model = mesh_gen.generate()

            solver = OpenSeesSolver(meshed_model)
            # 仅计算强度工况
            combo_strength = {"DL": 1.3, "LL": 1.5, "PL": 1.25}
            res = solver.run_load_combination(combo_strength, pressure_pa=30000)

            if res is None:
                raise Exception("Analysis Failed")

            processor = ResultProcessor(meshed_model)
            check_res = processor.run_gb50017_check(res)

            # --- 记录指标 ---
            ind.mass = meshed_model.calculate_mass_kg()
            ind.max_stress_ratio = check_res.max_stress_ratio
            ind.max_deflection_ratio = check_res.max_deflection_ratio

            # --- [关键修改] 分层适应度逻辑 ---
            # 逻辑：
            # 1. 如果可行：Fitness 由质量决定，范围 (1.0, +inf)
            # 2. 如果不可行：Fitness 由违约程度决定，范围 (0.0, 1.0)
            # 这样保证了 最差的可行解 > 最好的不可行解

            max_violation = max(ind.max_stress_ratio, ind.max_deflection_ratio)

            if max_violation <= 1.0:
                ind.is_feasible = True
                # 可行解区域：Mass 越小 Fitness 越大
                # 假设 Mass 范围 5000~50000，取基准 50000
                # Fitness = 50000 / Mass -> 范围 [1.0, 10.0]
                ind.fitness = 50000.0 / (ind.mass + 1.0) + 1.0
            else:
                ind.is_feasible = False
                # 不可行解区域：违约越小 Fitness 越大
                # Fitness = 1.0 / Violation^2 -> 范围 (0, 1.0]
                # 平方是为了让它快速远离大违约
                ind.fitness = 1.0 / (max_violation**2 + 1e-6)

            self.logger.info(
                f"Feasible: {ind.is_feasible}, Mass: {ind.mass:.2f} kg, Stress R: {ind.max_stress_ratio:.3f}, Disp R: {ind.max_deflection_ratio:.3f}, Fitness: {ind.fitness:.4f}"
            )

        except Exception as e:
            ind.fitness = 0.0
            ind.is_feasible = False
            ind.mass = float("inf")

    def evolve(self):
        """执行优化主循环"""
        # --- [关键修改] 初始化策略 ---
        # 80% 随机 + 20% 极强个体，确保种群里有“种子”能满足约束
        seed_count = int(POPULATION_SIZE * 0.2)
        pop = []

        # 插入"重型"种子
        print(f"Injecting {seed_count} heavy seeds to ensure feasibility...")
        for _ in range(seed_count):
            pop.append(Individual(self._create_heavy_genes()))

        # 补充随机个体
        for _ in range(POPULATION_SIZE - seed_count):
            pop.append(Individual(self._create_random_genes()))

        print(f"Initialized population of {POPULATION_SIZE}")

        for gen in range(MAX_GENERATIONS):
            # 1. 评估
            # (串行评估，若慢可改为 multiprocessing)
            for i, ind in enumerate(pop):
                if ind.fitness == 0.0:  # 避免重复评估
                    self.evaluate(ind)

            # 2. 统计与记录
            pop.sort(key=lambda x: x.fitness, reverse=True)
            best_ind = pop[0]
            self.best_history.append(best_ind)

            print(
                f"Gen {gen+1:02d} | Best Fit: {best_ind.fitness:.2e} | "
                f"Mass: {best_ind.mass:.0f}kg | "
                f"Stress R: {best_ind.max_stress_ratio:.2f} | "
                f"Disp R: {best_ind.max_deflection_ratio:.2f} | "
                f"Feasible: {best_ind.is_feasible}"
            )

            # 3. 终止条件检查 (可选)

            # 4. 生成下一代
            new_pop = []

            # 精英保留
            new_pop.extend([ind.deep_copy() for ind in pop[:ELITE_SIZE]])

            # 锦标赛选择 + 交叉 + 变异
            while len(new_pop) < POPULATION_SIZE:
                p1 = self._tournament_select(pop)
                p2 = self._tournament_select(pop)

                c1_genes, c2_genes = self._crossover(p1.genes, p2.genes)

                self._mutate(c1_genes)
                self._mutate(c2_genes)

                new_pop.append(Individual(c1_genes))
                if len(new_pop) < POPULATION_SIZE:
                    new_pop.append(Individual(c2_genes))

            pop = new_pop

        return pop[0]

    def _tournament_select(self, pop, k=3):
        candidates = random.sample(pop, k)
        return max(candidates, key=lambda x: x.fitness)

    def _crossover(self, g1, g2):
        if random.random() < CROSSOVER_RATE:
            # 简单单点交叉
            pt = random.randint(1, len(g1) - 1)
            new_g1 = g1[:pt] + g2[pt:]
            new_g2 = g2[:pt] + g1[pt:]
            return new_g1, new_g2
        return list(g1), list(g2)

    def _mutate(self, genes):
        for i in range(len(genes)):
            if random.random() < MUTATION_RATE:
                if i < 4:  # 离散变量 (截面索引)
                    genes[i] = random.randint(0, SEC_IDX_RANGES[i])
                # elif i == 4:  # 连续变量 (Main Spacing)
                #     genes[i] = random.uniform(SPACING_BOUNDS["main_min"], SPACING_BOUNDS["main_max"])
                # elif i == 5:  # 连续变量 (Brace Spacing)
                #     genes[i] = random.uniform(SPACING_BOUNDS["brace_min"], SPACING_BOUNDS["brace_max"])
