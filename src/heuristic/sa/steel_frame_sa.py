"""
模拟退火算法优化钢框架结构
优化目标：在满足应力和位移约束的条件下，使材料用量最小
设计变量：
- 4个截面索引：框架梁(方钢)、主梁(H型钢-主梁)、次梁(H型钢-次梁)、支撑梁(C型钢)
- 2个间距参数：主梁间距[0.3, 1.8]、支撑梁间距[0.3, 1.0]，步长0.05
约束条件：
- 应力约束：≤355 MPa
- 位移约束：≤14.4 mm
"""

import copy
import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import yaml

from src.analysis.opensees_analysis import OpenSeesAnalyzer
from src.structure.mesh_generator import MeshGenerator, MeshParameters
from src.structure.section_library import SectionLibrary
from src.structure.structural_model import StructuralModel
from src.utils.logger import Logger
from src.utils.model_utils import create_structural_model


class CoolingSchedule(Enum):
    """冷却策略枚举"""

    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    ADAPTIVE = "adaptive"


@dataclass
class SAParameters:
    """模拟退火算法参数"""

    initial_temperature: float = 1000.0
    final_temperature: float = 0.1
    max_iterations: int = 5000
    max_iterations_per_temp: int = 100
    cooling_schedule: CoolingSchedule = CoolingSchedule.EXPONENTIAL
    cooling_rate: float = 0.95
    adaptive_factor: float = 1.1
    restart_threshold: int = 500


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


class Solution:
    """解表示类"""

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

        # 评估结果
        self.objective_value = float("inf")  # 材料用量
        self.fitness = 0.0  # 适应度（考虑约束）
        self.max_stress = 0.0
        self.max_displacement = 0.0
        self.is_feasible = False
        self.penalty = 0.0

    def copy(self):
        """复制解"""
        new_solution = Solution(
            self.frame_section_index,
            self.main_beam_section_index,
            self.secondary_beam_section_index,
            self.brace_section_index,
            self.main_beam_spacing,
            self.brace_spacing,
        )
        new_solution.objective_value = self.objective_value
        new_solution.fitness = self.fitness
        new_solution.max_stress = self.max_stress
        new_solution.max_displacement = self.max_displacement
        new_solution.is_feasible = self.is_feasible
        new_solution.penalty = self.penalty
        return new_solution

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
            f"Solution[Frame:{self.frame_section_index}, Main:{self.main_beam_section_index}, "
            f"Sec:{self.secondary_beam_section_index}, Brace:{self.brace_section_index}, "
            f"MainSpacing:{self.main_beam_spacing:.2f}, BraceSpacing:{self.brace_spacing:.2f}, "
            f"Objective:{self.objective_value:.6f}, Feasible:{self.is_feasible}]"
        )


class SteelFrameSA:
    """钢框架模拟退火算法优化器"""

    def __init__(
        self, config_path: str = "config.yaml", sa_params: SAParameters = None, logger: Logger = None
    ):
        self.config_path = config_path
        self.sa_params = sa_params or SAParameters()
        self.constraints = DesignConstraints()
        self.spacing_limits = SpacingLimits()
        self.logger = logger or Logger("sa_optimization.log")

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
        self.current_temperature = self.sa_params.initial_temperature
        self.iteration = 0
        self.accepted_moves = 0
        self.rejected_moves = 0
        self.best_solution = None
        self.current_solution = None
        self.temperature_history = []
        self.best_objective_history = []
        self.current_objective_history = []
        self.acceptance_rate_history = []

    def _generate_spacing_candidates(self, min_val: float, max_val: float, step: float) -> List[float]:
        """生成间距候选值"""
        candidates = []
        current = min_val
        while current <= max_val + 1e-6:  # 添加小的容差
            candidates.append(round(current, 2))
            current += step
        return candidates

    def create_random_solution(self) -> Solution:
        """创建随机解"""
        return Solution(
            frame_idx=random.randint(0, self.section_counts["方钢"] - 1),
            main_idx=random.randint(0, self.section_counts["H型钢-主梁"] - 1),
            secondary_idx=random.randint(0, self.section_counts["H型钢-次梁"] - 1),
            brace_idx=random.randint(0, self.section_counts["C型钢"] - 1),
            main_spacing=random.choice(self.main_spacing_candidates),
            brace_spacing=random.choice(self.brace_spacing_candidates),
        )

    def calculate_material_volume(self, model: StructuralModel) -> float:
        """计算材料用量（体积）"""
        return model.calculate_mass() / 7850.0

    def evaluate_solution(self, solution: Solution) -> None:
        """评估解的目标函数值和约束"""
        try:
            # 创建配置字典
            config = copy.deepcopy(self.base_config)
            config["structural_params"].update(solution.to_dict())

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
                print("分析失败")
                solution.objective_value = float("inf")
                solution.fitness = float("inf")
                solution.is_feasible = False
                solution.max_stress = float("inf")
                solution.max_displacement = float("inf")
                solution.penalty = float("inf")
                return

            # 获取最大指标
            max_indicators = self.opensees_analyzer.get_max_indicators()
            solution.max_stress = max_indicators["max_abs_stress"]
            solution.max_displacement = max_indicators["max_abs_displacement"]

            # 计算材料用量
            solution.objective_value = self.calculate_material_volume(model)

            # 检查约束
            stress_feasible = solution.max_stress <= self.constraints.max_stress
            displacement_feasible = solution.max_displacement <= self.constraints.max_displacement
            solution.is_feasible = stress_feasible and displacement_feasible

            # 计算惩罚项
            if solution.is_feasible:
                solution.penalty = 0.0
                solution.fitness = solution.objective_value  # 可行解直接使用目标值
            else:
                # 不可行解：计算惩罚项
                stress_violation = max(0, solution.max_stress - self.constraints.max_stress)
                displacement_violation = max(0, solution.max_displacement - self.constraints.max_displacement)
                solution.penalty = 100 * (
                    stress_violation / self.constraints.max_stress
                    + displacement_violation / self.constraints.max_displacement
                )
                solution.fitness = solution.objective_value + solution.penalty

        except Exception as e:
            self.logger.error(f"评估解时出错: {e}")
            solution.objective_value = float("inf")
            solution.fitness = float("inf")
            solution.is_feasible = False
            solution.max_stress = float("inf")
            solution.max_displacement = float("inf")
            solution.penalty = float("inf")

    def generate_neighbor(self, solution: Solution) -> Solution:
        """生成邻域解"""
        neighbor = solution.copy()

        # 随机选择一个变量进行扰动
        variable_to_change = random.randint(0, 5)

        if variable_to_change == 0:  # frame_section_index
            # 在当前索引附近随机选择
            current_idx = neighbor.frame_section_index
            max_idx = self.section_counts["方钢"] - 1
            # 选择相邻的索引或随机索引
            if random.random() < 0.7:  # 70%概率选择相邻索引
                delta = random.choice([-1, 1])
                neighbor.frame_section_index = max(0, min(max_idx, current_idx + delta))
            else:  # 30%概率随机选择
                neighbor.frame_section_index = random.randint(0, max_idx)

        elif variable_to_change == 1:  # main_beam_section_index
            current_idx = neighbor.main_beam_section_index
            max_idx = self.section_counts["H型钢-主梁"] - 1
            if random.random() < 0.7:
                delta = random.choice([-1, 1])
                neighbor.main_beam_section_index = max(0, min(max_idx, current_idx + delta))
            else:
                neighbor.main_beam_section_index = random.randint(0, max_idx)

        elif variable_to_change == 2:  # secondary_beam_section_index
            current_idx = neighbor.secondary_beam_section_index
            max_idx = self.section_counts["H型钢-次梁"] - 1
            if random.random() < 0.7:
                delta = random.choice([-1, 1])
                neighbor.secondary_beam_section_index = max(0, min(max_idx, current_idx + delta))
            else:
                neighbor.secondary_beam_section_index = random.randint(0, max_idx)

        elif variable_to_change == 3:  # brace_section_index
            current_idx = neighbor.brace_section_index
            max_idx = self.section_counts["C型钢"] - 1
            if random.random() < 0.7:
                delta = random.choice([-1, 1])
                neighbor.brace_section_index = max(0, min(max_idx, current_idx + delta))
            else:
                neighbor.brace_section_index = random.randint(0, max_idx)

        elif variable_to_change == 4:  # main_beam_spacing
            current_spacing = neighbor.main_beam_spacing
            current_idx = self.main_spacing_candidates.index(current_spacing)
            if random.random() < 0.7:
                delta = random.choice([-1, 1])
                new_idx = max(0, min(len(self.main_spacing_candidates) - 1, current_idx + delta))
                neighbor.main_beam_spacing = self.main_spacing_candidates[new_idx]
            else:
                neighbor.main_beam_spacing = random.choice(self.main_spacing_candidates)

        elif variable_to_change == 5:  # brace_spacing
            current_spacing = neighbor.brace_spacing
            current_idx = self.brace_spacing_candidates.index(current_spacing)
            if random.random() < 0.7:
                delta = random.choice([-1, 1])
                new_idx = max(0, min(len(self.brace_spacing_candidates) - 1, current_idx + delta))
                neighbor.brace_spacing = self.brace_spacing_candidates[new_idx]
            else:
                neighbor.brace_spacing = random.choice(self.brace_spacing_candidates)

        return neighbor

    def update_temperature(self, iteration: int) -> float:
        """更新温度"""
        if self.sa_params.cooling_schedule == CoolingSchedule.LINEAR:
            # 线性冷却
            ratio = iteration / self.sa_params.max_iterations
            temperature = self.sa_params.initial_temperature * (1 - ratio)
            return max(temperature, self.sa_params.final_temperature)

        elif self.sa_params.cooling_schedule == CoolingSchedule.EXPONENTIAL:
            # 指数冷却
            temperature = self.sa_params.initial_temperature * (self.sa_params.cooling_rate**iteration)
            return max(temperature, self.sa_params.final_temperature)

        elif self.sa_params.cooling_schedule == CoolingSchedule.LOGARITHMIC:
            # 对数冷却
            temperature = self.sa_params.initial_temperature / (1 + math.log(1 + iteration))
            return max(temperature, self.sa_params.final_temperature)

        elif self.sa_params.cooling_schedule == CoolingSchedule.ADAPTIVE:
            # 自适应冷却
            if hasattr(self, "_last_acceptance_rate"):
                if self._last_acceptance_rate > 0.5:
                    # 接受率太高，降温更快
                    temperature = self.current_temperature * (self.sa_params.cooling_rate * 0.9)
                elif self._last_acceptance_rate < 0.1:
                    # 接受率太低，降温更慢
                    temperature = self.current_temperature * (self.sa_params.cooling_rate * 1.1)
                else:
                    temperature = self.current_temperature * self.sa_params.cooling_rate
            else:
                temperature = self.current_temperature * self.sa_params.cooling_rate
            return max(temperature, self.sa_params.final_temperature)

        return self.sa_params.final_temperature

    def accept_solution(self, current_fitness: float, neighbor_fitness: float, temperature: float) -> bool:
        """接受准则"""
        if neighbor_fitness < current_fitness:  # 更好的解（目标是最小化）
            return True

        if temperature <= 0:
            return False

        # Metropolis准则
        delta = neighbor_fitness - current_fitness
        probability = math.exp(-delta / temperature)
        return random.random() < probability

    def run_optimization(self) -> Solution:
        """运行模拟退火优化"""
        self.logger.info("开始模拟退火优化...")
        self.logger.info(f"初始温度: {self.sa_params.initial_temperature}")
        self.logger.info(f"最终温度: {self.sa_params.final_temperature}")
        self.logger.info(f"最大迭代次数: {self.sa_params.max_iterations}")
        self.logger.info(f"冷却策略: {self.sa_params.cooling_schedule.value}")

        # 初始化解
        self.current_solution = self.create_random_solution()
        self.evaluate_solution(self.current_solution)
        self.best_solution = self.current_solution.copy()

        self.logger.info(f"初始解: {self.current_solution}")

        # 初始化统计
        self.current_temperature = self.sa_params.initial_temperature
        self.iteration = 0
        self.accepted_moves = 0
        self.rejected_moves = 0
        iterations_without_improvement = 0

        # 主循环
        while (
            self.iteration < self.sa_params.max_iterations
            and self.current_temperature > self.sa_params.final_temperature
        ):

            temp_accepted = 0
            temp_total = 0

            # 在当前温度下进行多次尝试
            for _ in range(self.sa_params.max_iterations_per_temp):
                # 生成邻域解
                neighbor = self.generate_neighbor(self.current_solution)
                self.evaluate_solution(neighbor)

                # 判断是否接受
                accept = self.accept_solution(
                    self.current_solution.fitness, neighbor.fitness, self.current_temperature
                )

                if accept:
                    self.current_solution = neighbor
                    self.accepted_moves += 1
                    temp_accepted += 1

                    # 更新最优解
                    if neighbor.fitness < self.best_solution.fitness or (
                        neighbor.is_feasible and not self.best_solution.is_feasible
                    ):
                        self.best_solution = neighbor.copy()
                        iterations_without_improvement = 0
                        self.logger.info(f"发现更好解 - 迭代{self.iteration}: {self.best_solution}")
                    else:
                        iterations_without_improvement += 1
                else:
                    self.rejected_moves += 1

                temp_total += 1
                self.iteration += 1

                # 检查是否需要重启
                if iterations_without_improvement >= self.sa_params.restart_threshold:
                    self.logger.info(f"长时间无改进，重启搜索...")
                    self.current_solution = self.create_random_solution()
                    self.evaluate_solution(self.current_solution)
                    iterations_without_improvement = 0

                if self.iteration >= self.sa_params.max_iterations:
                    break

            # 计算当前温度下的接受率
            acceptance_rate = temp_accepted / temp_total if temp_total > 0 else 0
            self._last_acceptance_rate = acceptance_rate

            # 记录历史
            self.temperature_history.append(self.current_temperature)
            self.best_objective_history.append(self.best_solution.objective_value)
            self.current_objective_history.append(self.current_solution.objective_value)
            self.acceptance_rate_history.append(acceptance_rate)

            # 更新温度
            self.current_temperature = self.update_temperature(self.iteration)

            # 输出进度信息
            if self.iteration % 100 == 0 or self.current_temperature <= self.sa_params.final_temperature:
                self.logger.info(
                    f"迭代 {self.iteration}: T={self.current_temperature:.4f}, "
                    f"当前目标值={self.current_solution.objective_value:.6f}, "
                    f"最优目标值={self.best_solution.objective_value:.6f}, "
                    f"接受率={acceptance_rate:.3f}, "
                    f"可行性={self.best_solution.is_feasible}"
                )

        # 输出最终结果
        total_moves = self.accepted_moves + self.rejected_moves
        overall_acceptance_rate = self.accepted_moves / total_moves if total_moves > 0 else 0

        self.logger.info(f"\n模拟退火优化完成！")
        self.logger.info(f"总迭代次数: {self.iteration}")
        self.logger.info(f"接受的移动: {self.accepted_moves}")
        self.logger.info(f"拒绝的移动: {self.rejected_moves}")
        self.logger.info(f"总体接受率: {overall_acceptance_rate:.3f}")
        self.logger.info(f"最终温度: {self.current_temperature:.6f}")

        self.logger.info(f"\n最优解:")
        self.logger.info(f"目标函数值: {self.best_solution.objective_value:.6f} m³")
        self.logger.info(f"可行性: {self.best_solution.is_feasible}")
        if self.best_solution.is_feasible:
            self.logger.info(f"最大应力: {self.best_solution.max_stress:.2f} MPa")
            self.logger.info(f"最大位移: {self.best_solution.max_displacement:.2f} mm")
        self.logger.info(f"设计参数: {self.best_solution}")

        return self.best_solution

    def get_optimization_statistics(self) -> Dict:
        """获取优化统计信息"""
        total_moves = self.accepted_moves + self.rejected_moves
        return {
            "total_iterations": self.iteration,
            "accepted_moves": self.accepted_moves,
            "rejected_moves": self.rejected_moves,
            "overall_acceptance_rate": self.accepted_moves / total_moves if total_moves > 0 else 0,
            "final_temperature": self.current_temperature,
            "best_objective_value": (
                self.best_solution.objective_value if self.best_solution else float("inf")
            ),
            "best_is_feasible": self.best_solution.is_feasible if self.best_solution else False,
        }


def main():
    """主函数"""
    # 设置模拟退火参数
    sa_params = SAParameters(
        initial_temperature=1000.0,
        final_temperature=0.1,
        max_iterations=3000,
        max_iterations_per_temp=50,
        cooling_schedule=CoolingSchedule.EXPONENTIAL,
        cooling_rate=0.95,
        restart_threshold=300,
    )

    # 创建优化器
    optimizer = SteelFrameSA(config_path="config.yaml", sa_params=sa_params)

    # 运行优化
    best_solution = optimizer.run_optimization()

    return best_solution


if __name__ == "__main__":
    best_solution = main()
