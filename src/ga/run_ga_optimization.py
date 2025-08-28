"""
钢框架结构优化主程序
使用遗传算法优化钢框架的截面尺寸和构件间距
"""

import json
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from src.ga.steel_frame_ga import GAParameters, SteelFrameGA
from src.utils.logger import Logger

chinese_font = "SimSun"  # 宋体
english_font = "Times New Roman"
plt.rcParams["font.sans-serif"] = [chinese_font, english_font]
plt.rcParams["font.serif"] = [english_font, chinese_font]

logger = Logger("GA_optimization.log")


def convert_to_serializable(obj):
    """转换numpy类型为Python原生类型以便JSON序列化"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def save_section_details(optimizer: SteelFrameGA):
    """保存详细的截面信息"""
    best_solution = optimizer.best_chromosome
    section_library = optimizer.section_library

    section_details = {
        "框架梁(方钢)": {
            "index": best_solution.frame_section_index,
            "section": (
                section_library.get_section("方钢", best_solution.frame_section_index).name
                if section_library.get_section("方钢", best_solution.frame_section_index)
                else None
            ),
        },
        "主梁(H型钢-主梁)": {
            "index": best_solution.main_beam_section_index,
            "section": (
                section_library.get_section("H型钢-主梁", best_solution.main_beam_section_index).name
                if section_library.get_section("H型钢-主梁", best_solution.main_beam_section_index)
                else None
            ),
        },
        "次梁(H型钢-次梁)": {
            "index": best_solution.secondary_beam_section_index,
            "section": (
                section_library.get_section("H型钢-次梁", best_solution.secondary_beam_section_index).name
                if section_library.get_section("H型钢-次梁", best_solution.secondary_beam_section_index)
                else None
            ),
        },
        "支撑梁(C型钢)": {
            "index": best_solution.brace_section_index,
            "section": (
                section_library.get_section("C型钢", best_solution.brace_section_index).name
                if section_library.get_section("C型钢", best_solution.brace_section_index)
                else None
            ),
        },
        "间距参数": {"主梁间距": best_solution.main_beam_spacing, "支撑梁间距": best_solution.brace_spacing},
    }

    serializable_details = convert_to_serializable(section_details)

    return serializable_details


def save_optimization_results(optimizer: SteelFrameGA, output_dir: str):
    """保存优化结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存最优解
    best_solution = optimizer.best_chromosome
    result_data = {
        "timestamp": datetime.now().isoformat(),
        "ga_parameters": {
            "population_size": optimizer.ga_params.population_size,
            "max_generations": optimizer.ga_params.max_generations,
            "crossover_rate": optimizer.ga_params.crossover_rate,
            "mutation_rate": optimizer.ga_params.mutation_rate,
            "elite_ratio": optimizer.ga_params.elite_ratio,
            "seed": optimizer.ga_params.seed,
        },
        "constraints": {
            "max_stress": optimizer.constraints.max_stress,
            "max_displacement": optimizer.constraints.max_displacement,
        },
        "best_solution": {
            "frame_section_index": best_solution.frame_section_index,
            "main_beam_section_index": best_solution.main_beam_section_index,
            "secondary_beam_section_index": best_solution.secondary_beam_section_index,
            "brace_section_index": best_solution.brace_section_index,
            "main_beam_spacing": best_solution.main_beam_spacing,
            "brace_spacing": best_solution.brace_spacing,
            "fitness": best_solution.fitness,
            "material_volume": best_solution.material_volume,
            "max_stress": best_solution.max_stress,
            "max_displacement": best_solution.max_displacement,
            "is_feasible": best_solution.is_feasible,
        },
        "section_count": optimizer.section_counts,
        "section_details": save_section_details(optimizer),
        "optimization_history": {
            "best_fitness": optimizer.best_fitness_history,
            "avg_fitness": optimizer.avg_fitness_history,
        },
    }

    # 保存JSON结果
    result_file = os.path.join(output_dir, "optimization_results.json")
    serializable_data = convert_to_serializable(result_data)

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)

    logger.info(f"优化结果已保存到: {result_file}")

    # 绘制收敛曲线
    plot_convergence_curve(optimizer, output_dir)


def plot_convergence_curve(optimizer: SteelFrameGA, output_dir: str):
    """绘制收敛曲线"""
    plt.figure(figsize=(12, 8))

    generations = range(1, len(optimizer.best_fitness_history) + 1)

    plt.subplot(2, 1, 1)
    plt.plot(generations, optimizer.best_fitness_history, "b-", linewidth=2, label="最优适应度")
    plt.plot(generations, optimizer.avg_fitness_history, "r--", linewidth=1, label="平均适应度")
    plt.xlabel("代数")
    plt.ylabel("适应度")
    plt.title("遗传算法收敛曲线")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    # 计算材料用量历史（从适应度反推）
    material_volume_history = optimizer.material_history

    plt.plot(generations, material_volume_history, "g-", linewidth=2, label="最优材料用量")
    plt.xlabel("代数")
    plt.ylabel("材料用量 (m³)")
    plt.title("材料用量优化过程")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_file = os.path.join(output_dir, "convergence_curve.png")
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"收敛曲线已保存到: {plot_file}")


def run_optimization_with_different_parameters():
    """使用不同参数运行多次优化"""

    # 定义不同的参数组合
    parameter_sets = [
        {
            "name": "小规模快速测试",
            "params": GAParameters(
                population_size=20, max_generations=10, crossover_rate=0.8, mutation_rate=0.1, elite_ratio=0.1
            ),
        },
        {
            "name": "中等规模优化",
            "params": GAParameters(
                population_size=50, max_generations=30, crossover_rate=0.8, mutation_rate=0.1, elite_ratio=0.1
            ),
        },
        {
            "name": "大规模深度优化",
            "params": GAParameters(
                population_size=100,
                max_generations=100,
                crossover_rate=0.8,
                mutation_rate=0.05,
                elite_ratio=0.05,
            ),
        },
    ]

    # 创建结果目录
    base_output_dir = "results/GA_optimization"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, param_set in enumerate(parameter_sets):
        logger.info(f"\n{'='*50}")
        logger.info(f"开始运行: {param_set['name']}")
        logger.info(f"{'='*50}")

        # 创建输出目录
        output_dir = os.path.join(base_output_dir, f"{timestamp}_{i+1}_{param_set['name']}")

        # 创建优化器
        optimizer = SteelFrameGA(config_path="config.yaml", ga_params=param_set["params"], logger=logger)

        # 记录开始时间
        start_time = time.time()

        try:
            # 运行优化
            best_solution = optimizer.run_optimization()

            # 记录结束时间
            end_time = time.time()
            optimization_time = end_time - start_time

            logger.info(f"优化完成！耗时: {optimization_time:.2f} 秒")

            # 保存结果
            save_optimization_results(optimizer, output_dir)

            # 输出结果摘要
            logger.info(f"\n--- {param_set['name']} 结果摘要 ---")
            logger.info(f"最优适应度: {best_solution.fitness:.6f}")
            logger.info(f"可行性: {best_solution.is_feasible}")
            if best_solution.is_feasible:
                logger.info(f"材料用量: {best_solution.material_volume:.6f} m³")
                logger.info(f"最大应力: {best_solution.max_stress:.2f} MPa")
                logger.info(f"最大位移: {best_solution.max_displacement:.2f} mm")
            logger.info(f"优化时间: {optimization_time:.2f} 秒")

        except Exception as e:
            logger.error(f"优化过程中出现错误: {e}")
            import traceback

            traceback.print_exc()


def run_single_optimization():
    """运行单次优化"""
    logger.info("开始钢框架结构优化...")

    # 设置遗传算法参数
    ga_params = GAParameters(
        population_size=40, max_generations=100, crossover_rate=0.8, mutation_rate=0.1, elite_ratio=0.1
    )

    # 创建优化器
    optimizer = SteelFrameGA(config_path="config.yaml", ga_params=ga_params, logger=logger)

    # 显示优化设置
    logger.info(f"种群大小: {ga_params.population_size}")
    logger.info(f"最大代数: {ga_params.max_generations}")
    logger.info(f"交叉率: {ga_params.crossover_rate}")
    logger.info(f"变异率: {ga_params.mutation_rate}")
    logger.info(f"精英比例: {ga_params.elite_ratio}")
    logger.info(f"应力约束: ≤{optimizer.constraints.max_stress} MPa")
    logger.info(f"位移约束: ≤{optimizer.constraints.max_displacement} mm")

    # 记录开始时间
    start_time = time.time()

    try:
        # 运行优化
        best_solution = optimizer.run_optimization()

        # 记录结束时间
        end_time = time.time()
        optimization_time = end_time - start_time

        logger.info(f"优化完成！总耗时: {optimization_time:.2f} 秒")

        # 创建输出目录并保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/GA_optimization/{timestamp}_single_run"
        save_optimization_results(optimizer, output_dir)

        # 输出最终结果
        print("\n" + "=" * 60)
        print("钢框架结构优化结果")
        print("=" * 60)
        print(f"优化时间: {optimization_time:.2f} 秒")
        print(f"最优适应度: {best_solution.fitness:.6f}")
        print(f"解的可行性: {'可行' if best_solution.is_feasible else '不可行'}")

        if best_solution.is_feasible:
            print(f"材料用量: {best_solution.material_volume:.6f} m³")
            print(
                f"最大应力: {best_solution.max_stress:.2f} MPa (约束: ≤{optimizer.constraints.max_stress} MPa)"
            )
            print(
                f"最大位移: {best_solution.max_displacement:.2f} mm (约束: ≤{optimizer.constraints.max_displacement} mm)"
            )

        print("\n设计参数:")
        print(f"框架梁截面索引: {best_solution.frame_section_index}")
        print(f"主梁截面索引: {best_solution.main_beam_section_index}")
        print(f"次梁截面索引: {best_solution.secondary_beam_section_index}")
        print(f"支撑梁截面索引: {best_solution.brace_section_index}")
        print(f"主梁间距: {best_solution.main_beam_spacing:.2f} m")
        print(f"支撑梁间距: {best_solution.brace_spacing:.2f} m")
        print(f"\n结果已保存到: {output_dir}")
        print("=" * 60)

        return best_solution

    except Exception as e:
        logger.error(f"优化过程中出现错误: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """主函数"""
    print("钢框架结构遗传算法优化程序")
    print("=" * 50)
    run_single_optimization()

    # print("1. 运行单次优化")
    # print("2. 运行多参数对比优化")
    # print("3. 退出")

    # choice = input("\n请选择运行模式 (1-3): ").strip()

    # if choice == "1":
    #     run_single_optimization()
    # elif choice == "2":
    #     run_optimization_with_different_parameters()
    # elif choice == "3":
    #     print("程序退出")
    # else:
    #     print("无效选择，请输入1-3")


if __name__ == "__main__":
    main()
