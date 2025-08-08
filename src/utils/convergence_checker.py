import itertools
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.reinforcement.structural_env import StructuralEnv
from src.structure.structural_model import SectionLibrary
from src.utils.logger import Logger


@dataclass
class OptimalSolution:
    """最优解的数据结构"""

    section_indices: Dict[str, int]
    volume: float
    max_displacement: float
    max_stress: float
    is_feasible: bool


@dataclass
class ConvergenceReport:
    """收敛性报告"""

    total_combinations: int
    evaluated_combinations: int
    feasible_count: int
    optimal_volume: float
    optimal_solution: OptimalSolution
    feasible_solutions: List[OptimalSolution]
    evaluation_time: float


class ConvergenceChecker:
    """
    收敛性检查器，用于评估强化学习模型的最优性
    通过穷举搜索找到真正的最优解，并计算optimality gap
    """

    def __init__(self, env_config_path: str, logger: Optional[Logger] = None):
        self.env_config_path = env_config_path
        self.logger = logger or Logger("convergence_check.log")
        self.section_library = SectionLibrary()

        # 获取各类型截面的数量 (4, 11, 10, 17)
        self.section_counts = {
            "frame": self.section_library.section_num_by_frame_type("frame"),
            "main": self.section_library.section_num_by_frame_type("main"),
            "secondary": self.section_library.section_num_by_frame_type("secondary"),
            "brace": self.section_library.section_num_by_frame_type("brace"),
        }

    def evaluate_single_combination(
        self, indices: Tuple[int, int, int, int], base_params: dict
    ) -> Optional[OptimalSolution]:
        """
        评估单个截面组合

        Args:
            indices: (frame_idx, main_idx, secondary_idx, brace_idx)
            base_params: 基础结构参数

        Returns:
            OptimalSolution对象，如果分析失败则返回None
        """
        try:
            # 创建临时环境和参数
            temp_params = base_params.copy()
            temp_params["frame_section_index"] = indices[0]
            temp_params["main_beam_section_index"] = indices[1]
            temp_params["secondary_beam_section_index"] = indices[2]
            temp_params["brace_section_index"] = indices[3]

            # 创建临时配置
            temp_config = {"structural_params": temp_params}

            # 创建环境并评估
            env = StructuralEnv(temp_config, self.logger)
            state, info = env.reset()

            # 获取分析结果
            volume = info["model_volume_m3"]
            max_disp = info["max_displacement_mm"]
            max_stress = info["max_stress"]

            # 判断是否可行 (满足约束)
            is_feasible = max_disp <= env.max_disp and max_stress <= env.max_stress

            return OptimalSolution(
                section_indices={
                    "frame_section_index": indices[0],
                    "main_beam_section_index": indices[1],
                    "secondary_beam_section_index": indices[2],
                    "brace_section_index": indices[3],
                },
                volume=volume,
                max_displacement=max_disp,
                max_stress=max_stress,
                is_feasible=is_feasible,
            )

        except Exception as e:
            self.logger.info(f"评估组合 {indices} 失败: {e}")
            return None

    def exhaustive_search(
        self, pressure: float = 30.0, max_workers: int = 4, save_results: bool = True
    ) -> ConvergenceReport:
        """
        穷举搜索所有可能的截面组合，找到最优解

        Args:
            pressure: 内压值 (kPa)
            max_workers: 并行工作进程数
            save_results: 是否保存结果到文件

        Returns:
            ConvergenceReport对象
        """
        start_time = time.time()

        # 生成所有可能的组合
        combinations = list(
            itertools.product(
                range(self.section_counts["frame"]),
                range(self.section_counts["main"]),
                range(self.section_counts["secondary"]),
                range(self.section_counts["brace"]),
            )
        )

        total_combinations = len(combinations)
        self.logger.info(f"开始穷举搜索，总计 {total_combinations} 个组合")

        # 准备基础参数
        from src.utils.model_utils import load_config

        config = load_config(self.env_config_path)
        base_params = config["structural_params"].copy()
        base_params["pressure"] = pressure

        # 并行评估所有组合
        feasible_solutions = []
        evaluated_count = 0

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_combination = {
                executor.submit(self.evaluate_single_combination, combo, base_params): combo
                for combo in combinations
            }

            # 收集结果
            for future in as_completed(future_to_combination):
                combination = future_to_combination[future]
                try:
                    result = future.result()
                    evaluated_count += 1

                    if result is not None and result.is_feasible:
                        feasible_solutions.append(result)

                    if evaluated_count % 100 == 0:
                        self.logger.info(f"已评估 {evaluated_count}/{total_combinations} 个组合")

                except Exception as e:
                    self.logger.info(f"组合 {combination} 处理失败: {e}")

        # 找到最优解 (体积最小的可行解)
        if feasible_solutions:
            optimal_solution = min(feasible_solutions, key=lambda x: x.volume)
            optimal_volume = optimal_solution.volume
        else:
            self.logger.info("警告: 没有找到可行解!")
            optimal_solution = None
            optimal_volume = float("inf")

        evaluation_time = time.time() - start_time

        # 生成报告
        report = ConvergenceReport(
            total_combinations=total_combinations,
            evaluated_combinations=evaluated_count,
            feasible_count=len(feasible_solutions),
            optimal_volume=optimal_volume,
            optimal_solution=optimal_solution,
            feasible_solutions=feasible_solutions,
            evaluation_time=evaluation_time,
        )

        self.logger.info(f"穷举搜索完成！")
        self.logger.info(f"总组合数: {total_combinations}")
        self.logger.info(f"成功评估: {evaluated_count}")
        self.logger.info(f"可行解数量: {len(feasible_solutions)}")
        self.logger.info(f"最优体积: {optimal_volume:.4f} m³")
        self.logger.info(f"评估时间: {evaluation_time:.2f} 秒")

        if optimal_solution:
            self.logger.info(f"最优解截面配置: {optimal_solution.section_indices}")
            self.logger.info(f"最优解位移: {optimal_solution.max_displacement:.2f} mm")
            self.logger.info(f"最优解应力: {optimal_solution.max_stress:.2f} MPa")

        # 保存结果
        if save_results:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"exhaustive_search_results_{timestamp}.pkl"
            with open(filename, "wb") as f:
                pickle.dump(report, f)
            self.logger.info(f"结果已保存到: {filename}")

        return report

    def calculate_optimality_gap(self, agent_solution: Dict[str, int], ground_truth_volume: float) -> float:
        """
        计算智能体解与最优解的optimality gap

        Args:
            agent_solution: 智能体找到的解 (截面索引字典)
            ground_truth_volume: 真实最优体积

        Returns:
            optimality gap (百分比)
        """
        # 评估智能体解
        from src.utils.model_utils import load_config

        config = load_config(self.env_config_path)
        base_params = config["structural_params"].copy()

        # 更新截面索引
        for key, value in agent_solution.items():
            base_params[key] = value

        temp_config = {"structural_params": base_params}
        env = StructuralEnv(temp_config, self.logger)
        state, info = env.reset()

        agent_volume = info["model_volume_m3"]
        max_disp = info["max_displacement_mm"]
        max_stress = info["max_stress"]

        # 检查是否可行
        is_feasible = max_disp <= env.max_disp and max_stress <= env.max_stress

        if not is_feasible:
            self.logger.info("警告: 智能体解不满足约束条件!")
            return float("inf")

        # 计算gap
        gap = ((agent_volume - ground_truth_volume) / ground_truth_volume) * 100

        self.logger.info(f"智能体解体积: {agent_volume:.4f} m³")
        self.logger.info(f"最优解体积: {ground_truth_volume:.4f} m³")
        self.logger.info(f"Optimality Gap: {gap:.2f}%")

        return gap


def quick_convergence_test(config_path: str, pressure_values: List[float] = None):
    """
    快速收敛性测试函数

    Args:
        config_path: 配置文件路径
        pressure_values: 要测试的压力值列表
    """
    if pressure_values is None:
        pressure_values = [28.0, 30.0, 32.0, 35.0, 38.0]

    logger = Logger("convergence_test.log")
    checker = ConvergenceChecker(config_path, logger)

    results = {}
    for pressure in pressure_values:
        logger.info(f"\n{'='*50}")
        logger.info(f"测试压力值: {pressure} kPa")
        logger.info(f"{'='*50}")

        report = checker.exhaustive_search(pressure=pressure, max_workers=4)
        results[pressure] = report

        logger.info(f"压力 {pressure} kPa 的结果:")
        logger.info(f"  可行解比例: {report.feasible_count/report.total_combinations*100:.1f}%")
        logger.info(f"  最优体积: {report.optimal_volume:.4f} m³")

    return results


if __name__ == "__main__":
    # 示例用法
    results = quick_convergence_test("config.yaml")
