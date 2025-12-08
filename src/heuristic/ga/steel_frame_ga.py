"""
遗传算法优化钢框架结构
优化目标：在满足应力和位移约束的条件下，使材料用量最小
设计变量：
- 4个截面索引：框架梁(方钢)、主梁(H型钢-主梁)、次梁(H型钢-次梁)、支撑梁(C型钢)
- 2个间距参数：主梁间距[0.3, 1.8]、支撑梁间距[0.3, 1.0]，步长0.05
约束条件：
- 应力约束：≤355 MPa
- 位移约束：≤14.4 mm
"""

import copy
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import yaml

from src.analysis.opensees_analysis import OpenSeesAnalyzer
from src.structure.mesh_generator import MeshGenerator, MeshParameters
from src.structure.section_library import SectionLibrary
from src.structure.structural_model import StructuralModel
from src.utils.logger import Logger
from src.utils.model_utils import create_structural_model


@dataclass
class GAParameters:
    """遗传算法参数"""

    population_size: int = 50
    max_generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elite_ratio: float = 0.1
    tournament_size: int = 3
    seed: int = 42


@dataclass
class DesignConstraints:
    """设计约束"""

    max_stress: float = 355.0  # MPa
    max_displacement: float = 14.4  # mm


@dataclass
class SpacingLimits:
    """间距限制"""

    main_beam_min: float = 0.3
    main_beam_max: float = 1.8
    brace_min: float = 0.3
    brace_max: float = 1.0
    step: float = 0.05


class Chromosome:
    """染色体类"""

    def __init__(
        self,
        frame_idx: int = 0,
        main_idx: int = 0,
        secondary_idx: int = 0,
        brace_idx: int = 0,
        main_spacing: float = 0.55,
        brace_spacing: float = 1.0,
    ):
        # 截面索引
        self.frame_section_index = frame_idx
        self.main_beam_section_index = main_idx
        self.secondary_beam_section_index = secondary_idx
        self.brace_section_index = brace_idx

        # 间距参数
        self.main_beam_spacing = main_spacing
        self.brace_spacing = brace_spacing

        # 适应度相关
        self.fitness = 0.0
        self.material_volume = 0.0
        self.max_stress = 0.0
        self.max_displacement = 0.0
        self.is_feasible = False

    def copy(self):
        """复制染色体"""
        new_chromosome = Chromosome(
            self.frame_section_index,
            self.main_beam_section_index,
            self.secondary_beam_section_index,
            self.brace_section_index,
            self.main_beam_spacing,
            self.brace_spacing,
        )
        new_chromosome.fitness = self.fitness
        new_chromosome.material_volume = self.material_volume
        new_chromosome.max_stress = self.max_stress
        new_chromosome.max_displacement = self.max_displacement
        new_chromosome.is_feasible = self.is_feasible
        return new_chromosome

    def to_dict(self):
        """转换为字典格式用于创建结构模型"""
        return {
            "frame_section_index": self.frame_section_index,
            "main_beam_section_index": self.main_beam_section_index,
            "secondary_beam_section_index": self.secondary_beam_section_index,
            "brace_section_index": self.brace_section_index,
            "main_beam_spacing": self.main_beam_spacing,
            "brace_spacing": self.brace_spacing,
        }

    def __str__(self):
        return (
            f"Chromosome[Frame:{self.frame_section_index}, Main:{self.main_beam_section_index}, "
            f"Sec:{self.secondary_beam_section_index}, Brace:{self.brace_section_index}, "
            f"MainSpacing:{self.main_beam_spacing:.2f}, BraceSpacing:{self.brace_spacing:.2f}, "
            f"Fitness:{self.fitness:.4f}, Feasible:{self.is_feasible}]"
        )


class SteelFrameGA:
    """钢框架遗传算法优化器"""

    def __init__(
        self, config_path: str = "config.yaml", ga_params: GAParameters = None, logger: Logger = None
    ):
        self.config_path = config_path
        self.ga_params = ga_params or GAParameters()
        self.constraints = DesignConstraints()
        self.spacing_limits = SpacingLimits()

        self.logger = logger or Logger("GA_optimization.log")
        random.seed(self.ga_params.seed)

        # 加载配置文件
        with open(config_path, "r", encoding="utf-8") as f:
            self.base_config = yaml.safe_load(f)

        # 初始化截面库
        self.section_library = SectionLibrary()

        # 获取各类型截面数量
        self.section_counts = {
            "方钢": len(self.section_library.sections["方钢"]),
            "H型钢-主梁": len(self.section_library.sections["H型钢-主梁"]),
            "H型钢-次梁": len(self.section_library.sections["H型钢-次梁"]),
            "C型钢": len(self.section_library.sections["C型钢"]),
        }

        self.logger.info(f"截面库统计: {self.section_counts}")

        # 生成间距候选值
        self.main_spacing_candidates = self._generate_spacing_candidates(
            self.spacing_limits.main_beam_min, self.spacing_limits.main_beam_max, self.spacing_limits.step
        )
        self.brace_spacing_candidates = self._generate_spacing_candidates(
            self.spacing_limits.brace_min, self.spacing_limits.brace_max, self.spacing_limits.step
        )

        self.logger.info(f"主梁间距候选值数量: {len(self.main_spacing_candidates)}")
        self.logger.info(f"支撑梁间距候选值数量: {len(self.brace_spacing_candidates)}")

        # 初始化分析器
        self.opensees_analyzer = OpenSeesAnalyzer()

        # 统计信息
        self.generation = 0
        self.best_chromosome = None
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.material_history = []

    def _generate_spacing_candidates(self, min_val: float, max_val: float, step: float) -> List[float]:
        """生成间距候选值"""
        candidates = []
        current = min_val
        while current <= max_val + 1e-6:  # 添加小的容差
            candidates.append(round(current, 2))
            current += step
        return candidates

    def create_random_chromosome(self) -> Chromosome:
        """创建随机染色体"""
        return Chromosome(
            frame_idx=random.randint(0, self.section_counts["方钢"] - 1),
            main_idx=random.randint(0, self.section_counts["H型钢-主梁"] - 1),
            secondary_idx=random.randint(0, self.section_counts["H型钢-次梁"] - 1),
            brace_idx=random.randint(0, self.section_counts["C型钢"] - 1),
            main_spacing=random.choice(self.main_spacing_candidates),
            brace_spacing=random.choice(self.brace_spacing_candidates),
        )

    def initialize_population(self) -> List[Chromosome]:
        """初始化种群"""
        population = []
        for _ in range(self.ga_params.population_size):
            chromosome = self.create_random_chromosome()
            population.append(chromosome)
        return population

    def calculate_material_volume(self, model: StructuralModel) -> float:
        """计算材料用量（体积）"""
        return model.calculate_mass() / 7850.0

    def evaluate_chromosome(self, chromosome: Chromosome) -> None:
        """评估染色体适应度"""
        try:
            # 创建配置字典
            config: Dict[str, Dict] = copy.deepcopy(self.base_config)
            config["structural_params"].update(chromosome.to_dict())

            # 创建结构模型
            model = create_structural_model(config)

            # 生成网格
            mesh_params = MeshParameters(max_grid_size=0.4)
            mesh_generator = MeshGenerator(model, mesh_params)
            mesh_generator.generate_mesh()

            # OpenSees分析
            self.opensees_analyzer.setup_opensees_model(mesh_generator.meshed_model)
            self.opensees_analyzer.apply_gravity_and_pressure_loads(
                config["structural_params"]["pressure"] * 1e3
            )

            static_success = self.opensees_analyzer.run_static_analysis()

            if not static_success:
                # 分析失败，设置为不可行解
                chromosome.fitness = 0.0
                chromosome.is_feasible = False
                chromosome.max_stress = float("inf")
                chromosome.max_displacement = float("inf")
                chromosome.material_volume = float("inf")
                return

            # 获取最大指标
            max_indicators = self.opensees_analyzer.get_max_indicators()
            chromosome.max_stress = max_indicators["max_abs_stress"]
            chromosome.max_displacement = max_indicators["max_abs_displacement"]

            # 计算材料用量
            chromosome.material_volume = self.calculate_material_volume(model)

            # 检查约束
            stress_feasible = chromosome.max_stress <= self.constraints.max_stress
            displacement_feasible = chromosome.max_displacement <= self.constraints.max_displacement
            chromosome.is_feasible = stress_feasible and displacement_feasible

            # 计算适应度
            if chromosome.is_feasible:
                # 可行解：适应度 = 1 / 材料用量
                chromosome.fitness = 1.0 / (chromosome.material_volume + 1e-10)
            else:
                # 不可行解：使用惩罚函数
                stress_violation = max(0, chromosome.max_stress - self.constraints.max_stress)
                displacement_violation = max(
                    0, chromosome.max_displacement - self.constraints.max_displacement
                )
                penalty = 10 * (
                    stress_violation / self.constraints.max_stress
                    + displacement_violation / self.constraints.max_displacement
                )
                chromosome.fitness = 1.0 / (chromosome.material_volume + penalty + 1e-10)

        except Exception as e:
            self.logger.error(f"评估染色体时出错: {e}")
            chromosome.fitness = 0.0
            chromosome.is_feasible = False
            chromosome.max_stress = float("inf")
            chromosome.max_displacement = float("inf")
            chromosome.material_volume = float("inf")

    def roulette_wheel_selection(self, population: List[Chromosome]) -> Chromosome:
        """轮盘赌选择"""
        # 计算总适应度
        total_fitness = sum(individual.fitness for individual in population)

        if total_fitness <= 0:
            # 如果所有适应度都为0或负数，随机选择
            return random.choice(population)

        # 生成随机数
        r = random.uniform(0, total_fitness)

        # 轮盘赌选择
        current_sum = 0.0
        for individual in population:
            current_sum += individual.fitness
            if current_sum >= r:
                return individual

        # 如果由于浮点数精度问题没有选中，返回最后一个
        return population[-1]

    def tournament_selection(self, population: List[Chromosome]) -> Chromosome:
        """锦标赛选择（备用选择方法）"""
        tournament = random.sample(population, self.ga_params.tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """交叉操作"""
        child1 = parent1.copy()
        child2 = parent2.copy()

        if random.random() < self.ga_params.crossover_rate:
            # 对截面索引进行单点交叉
            if random.random() < 0.5:
                child1.frame_section_index, child2.frame_section_index = (
                    child2.frame_section_index,
                    child1.frame_section_index,
                )

            if random.random() < 0.5:
                child1.main_beam_section_index, child2.main_beam_section_index = (
                    child2.main_beam_section_index,
                    child1.main_beam_section_index,
                )

            if random.random() < 0.5:
                child1.secondary_beam_section_index, child2.secondary_beam_section_index = (
                    child2.secondary_beam_section_index,
                    child1.secondary_beam_section_index,
                )

            if random.random() < 0.5:
                child1.brace_section_index, child2.brace_section_index = (
                    child2.brace_section_index,
                    child1.brace_section_index,
                )

            # 对间距参数进行算术交叉
            if random.random() < 0.5:
                alpha = random.uniform(0.3, 0.7)
                new_main1 = alpha * parent1.main_beam_spacing + (1 - alpha) * parent2.main_beam_spacing
                new_main2 = alpha * parent2.main_beam_spacing + (1 - alpha) * parent1.main_beam_spacing

                # 找到最接近的候选值
                child1.main_beam_spacing = min(self.main_spacing_candidates, key=lambda x: abs(x - new_main1))
                child2.main_beam_spacing = min(self.main_spacing_candidates, key=lambda x: abs(x - new_main2))

            if random.random() < 0.5:
                alpha = random.uniform(0.3, 0.7)
                new_brace1 = alpha * parent1.brace_spacing + (1 - alpha) * parent2.brace_spacing
                new_brace2 = alpha * parent2.brace_spacing + (1 - alpha) * parent1.brace_spacing

                # 找到最接近的候选值
                child1.brace_spacing = min(self.brace_spacing_candidates, key=lambda x: abs(x - new_brace1))
                child2.brace_spacing = min(self.brace_spacing_candidates, key=lambda x: abs(x - new_brace2))

        return child1, child2

    def mutate(self, chromosome: Chromosome) -> None:
        """变异操作"""
        if random.random() < self.ga_params.mutation_rate:
            # 随机选择一个基因进行变异
            gene_to_mutate = random.randint(0, 5)

            if gene_to_mutate == 0:  # frame_section_index
                chromosome.frame_section_index = random.randint(0, self.section_counts["方钢"] - 1)
            elif gene_to_mutate == 1:  # main_beam_section_index
                chromosome.main_beam_section_index = random.randint(0, self.section_counts["H型钢-主梁"] - 1)
            elif gene_to_mutate == 2:  # secondary_beam_section_index
                chromosome.secondary_beam_section_index = random.randint(
                    0, self.section_counts["H型钢-次梁"] - 1
                )
            elif gene_to_mutate == 3:  # brace_section_index
                chromosome.brace_section_index = random.randint(0, self.section_counts["C型钢"] - 1)
            elif gene_to_mutate == 4:  # main_beam_spacing
                chromosome.main_beam_spacing = random.choice(self.main_spacing_candidates)
            elif gene_to_mutate == 5:  # brace_spacing
                chromosome.brace_spacing = random.choice(self.brace_spacing_candidates)

    def get_elite(self, population: List[Chromosome]) -> List[Chromosome]:
        """获取精英个体"""
        elite_count = max(1, int(self.ga_params.population_size * self.ga_params.elite_ratio))
        sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)
        return [individual.copy() for individual in sorted_population[:elite_count]]

    def run_optimization(self) -> Chromosome:
        """运行遗传算法优化"""
        self.logger.info("开始遗传算法优化...")

        # 初始化种群
        population = self.initialize_population()

        # 评估初始种群
        self.logger.info("评估初始种群...")
        for i, individual in enumerate(population):
            self.logger.info(f"评估个体 {i+1}/{len(population)}")
            self.evaluate_chromosome(individual)

        # 记录初始最优个体
        self.best_chromosome = max(population, key=lambda x: x.fitness).copy()

        # 主循环
        for generation in range(self.ga_params.max_generations):
            self.generation = generation
            self.logger.info(f"\n=== 第 {generation + 1} 代 ===")

            # 选择精英
            elite = self.get_elite(population)

            # 生成新种群
            new_population = elite.copy()

            while len(new_population) < self.ga_params.population_size:
                # 选择父代
                parent1 = self.roulette_wheel_selection(population)
                parent2 = self.roulette_wheel_selection(population)

                # 交叉
                child1, child2 = self.crossover(parent1, parent2)

                # 变异
                self.mutate(child1)
                self.mutate(child2)

                # 评估子代
                self.evaluate_chromosome(child1)
                if len(new_population) < self.ga_params.population_size:
                    new_population.append(child1)

                if len(new_population) < self.ga_params.population_size:
                    self.evaluate_chromosome(child2)
                    new_population.append(child2)

            # 更新种群
            population = new_population[: self.ga_params.population_size]

            # 统计信息
            current_best = max(population, key=lambda x: x.fitness)
            if current_best.fitness > self.best_chromosome.fitness:
                self.best_chromosome = current_best.copy()

            avg_fitness = sum(ind.fitness for ind in population) / len(population)
            feasible_count = sum(1 for ind in population if ind.is_feasible)

            self.best_fitness_history.append(self.best_chromosome.fitness)
            self.avg_fitness_history.append(avg_fitness)
            self.material_history.append(self.best_chromosome.material_volume)

            self.logger.info(f"最优适应度: {self.best_chromosome.fitness:.6f}")
            self.logger.info(f"平均适应度: {avg_fitness:.6f}")
            self.logger.info(f"可行解数量: {feasible_count}/{len(population)}")
            self.logger.info(f"最优个体: {self.best_chromosome}")

            if self.best_chromosome.is_feasible:
                self.logger.info(f"材料用量: {self.best_chromosome.material_volume:.6f} m³")
                self.logger.info(f"最大应力: {self.best_chromosome.max_stress:.2f} MPa")
                self.logger.info(f"最大位移: {self.best_chromosome.max_displacement:.2f} mm")

        self.logger.info(f"\n优化完成！最优解:")
        self.logger.info(f"适应度: {self.best_chromosome.fitness:.6f}")
        if self.best_chromosome.is_feasible:
            self.logger.info(f"材料用量: {self.best_chromosome.material_volume:.6f} m³")
            self.logger.info(f"最大应力: {self.best_chromosome.max_stress:.2f} MPa")
            self.logger.info(f"最大位移: {self.best_chromosome.max_displacement:.2f} mm")
        self.logger.info(f"设计参数: {self.best_chromosome}")

        return self.best_chromosome


def main():
    """主函数"""
    # 设置遗传算法参数
    ga_params = GAParameters(
        population_size=30, max_generations=50, crossover_rate=0.8, mutation_rate=0.1, elite_ratio=0.1
    )

    # 创建优化器
    optimizer = SteelFrameGA(config_path="config.yaml", ga_params=ga_params)

    # 运行优化
    best_solution = optimizer.run_optimization()

    return best_solution


if __name__ == "__main__":
    best_solution = main()
