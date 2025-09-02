"""
钢框架结构模拟退火优化主程序
使用模拟退火算法优化钢框架的截面尺寸和构件间距
"""

import copy
import json
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt

from src.sa.steel_frame_sa import CoolingSchedule, SAParameters, SteelFrameSA
from src.utils.logger import Logger
from src.utils.util import convert_to_serializable

chinese_font = "SimSun"  # 宋体
english_font = "Times New Roman"
plt.rcParams["font.sans-serif"] = [chinese_font, english_font]
plt.rcParams["font.serif"] = [english_font, chinese_font]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

logger = Logger("sa_optimization.log")


def save_optimization_results(optimizer: SteelFrameSA, output_dir: str):
    """保存优化结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存最优解
    best_solution = optimizer.best_solution
    stats = optimizer.get_optimization_statistics()

    result_data = {
        "timestamp": datetime.now().isoformat(),
        "algorithm": "Simulated Annealing",
        "sa_parameters": {
            "initial_temperature": optimizer.sa_params.initial_temperature,
            "final_temperature": optimizer.sa_params.final_temperature,
            "max_iterations": optimizer.sa_params.max_iterations,
            "max_iterations_per_temp": optimizer.sa_params.max_iterations_per_temp,
            "cooling_schedule": optimizer.sa_params.cooling_schedule.value,
            "cooling_rate": optimizer.sa_params.cooling_rate,
            "restart_threshold": optimizer.sa_params.restart_threshold,
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
            "objective_value": best_solution.objective_value,
            "fitness": best_solution.fitness,
            "max_stress": best_solution.max_stress,
            "max_displacement": best_solution.max_displacement,
            "is_feasible": best_solution.is_feasible,
            "penalty": best_solution.penalty,
        },
        "optimization_statistics": stats,
        "section_library_info": optimizer.section_counts,
        "section_details": save_section_details(optimizer),
        "optimization_history": {
            "temperature": optimizer.temperature_history,
            "best_objective": optimizer.best_objective_history,
            "current_objective": optimizer.current_objective_history,
            "acceptance_rate": optimizer.acceptance_rate_history,
        },
    }

    # 保存JSON结果
    result_file = os.path.join(output_dir, "sa_optimization_results.json")
    serializable_data = convert_to_serializable(result_data)

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)

    logger.info(f"优化结果已保存到: {result_file}")

    # 绘制优化过程图
    plot_optimization_process(optimizer, output_dir)


def plot_optimization_process(optimizer: SteelFrameSA, output_dir: str):
    """绘制优化过程图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    iterations = range(len(optimizer.temperature_history))

    # 温度变化曲线
    axes[0, 0].plot(iterations, optimizer.temperature_history, "r-", linewidth=2)
    axes[0, 0].set_xlabel("迭代次数")
    axes[0, 0].set_ylabel("温度")
    axes[0, 0].set_title("温度冷却曲线")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale("log")

    # 目标函数值变化
    axes[0, 1].plot(iterations, optimizer.best_objective_history, "b-", linewidth=2, label="最优目标值")
    axes[0, 1].plot(
        iterations, optimizer.current_objective_history, "g--", linewidth=1, alpha=0.7, label="当前目标值"
    )
    axes[0, 1].set_xlabel("迭代次数")
    axes[0, 1].set_ylabel("材料用量 (m³)")
    axes[0, 1].set_title("目标函数值变化")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 接受率变化
    axes[1, 0].plot(iterations, optimizer.acceptance_rate_history, "purple", linewidth=2)
    axes[1, 0].set_xlabel("迭代次数")
    axes[1, 0].set_ylabel("接受率")
    axes[1, 0].set_title("接受率变化")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)

    # 温度vs目标值散点图
    valid_indices = [i for i, obj in enumerate(optimizer.best_objective_history) if obj != float("inf")]
    if valid_indices:
        temp_vals = [optimizer.temperature_history[i] for i in valid_indices]
        obj_vals = [optimizer.best_objective_history[i] for i in valid_indices]
        axes[1, 1].scatter(temp_vals, obj_vals, c=valid_indices, cmap="viridis", alpha=0.6)
        axes[1, 1].set_xlabel("温度")
        axes[1, 1].set_ylabel("最优目标值 (m³)")
        axes[1, 1].set_title("温度与目标值关系")
        axes[1, 1].set_xscale("log")
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    plot_file = os.path.join(output_dir, "sa_optimization_process.png")
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"优化过程图已保存到: {plot_file}")


def save_section_details(optimizer: SteelFrameSA):
    """保存详细的截面信息"""
    best_solution = optimizer.best_solution
    section_library = optimizer.section_library

    section_details = {
        "框架梁(方钢)": {
            "index": best_solution.frame_section_index,
            "section": (section_library.get_section("方钢", best_solution.frame_section_index).name),
        },
        "主梁(H型钢-主梁)": {
            "index": best_solution.main_beam_section_index,
            "section": (
                section_library.get_section("H型钢-主梁", best_solution.main_beam_section_index).name
            ),
        },
        "次梁(H型钢-次梁)": {
            "index": best_solution.secondary_beam_section_index,
            "section": (
                section_library.get_section("H型钢-次梁", best_solution.secondary_beam_section_index).name
            ),
        },
        "支撑梁(C型钢)": {
            "index": best_solution.brace_section_index,
            "section": (section_library.get_section("C型钢", best_solution.brace_section_index).name),
        },
        "间距参数": {"主梁间距": best_solution.main_beam_spacing, "支撑梁间距": best_solution.brace_spacing},
    }

    serializable_details = convert_to_serializable(section_details)

    return serializable_details


def run_optimization_with_different_cooling_schedules():
    """使用不同冷却策略运行多次优化"""

    # 定义不同的冷却策略
    cooling_strategies = [
        {
            "name": "指数冷却",
            "params": SAParameters(
                initial_temperature=1000.0,
                final_temperature=0.1,
                max_iterations=2000,
                max_iterations_per_temp=50,
                cooling_schedule=CoolingSchedule.EXPONENTIAL,
                cooling_rate=0.95,
            ),
        },
        {
            "name": "线性冷却",
            "params": SAParameters(
                initial_temperature=1000.0,
                final_temperature=0.1,
                max_iterations=2000,
                max_iterations_per_temp=50,
                cooling_schedule=CoolingSchedule.LINEAR,
            ),
        },
        {
            "name": "对数冷却",
            "params": SAParameters(
                initial_temperature=1000.0,
                final_temperature=0.1,
                max_iterations=2000,
                max_iterations_per_temp=50,
                cooling_schedule=CoolingSchedule.LOGARITHMIC,
            ),
        },
        {
            "name": "自适应冷却",
            "params": SAParameters(
                initial_temperature=1000.0,
                final_temperature=0.1,
                max_iterations=2000,
                max_iterations_per_temp=50,
                cooling_schedule=CoolingSchedule.ADAPTIVE,
                cooling_rate=0.95,
            ),
        },
    ]

    # 创建结果目录
    base_output_dir = "results/sa_optimization"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    comparison_results = []

    for i, strategy in enumerate(cooling_strategies):
        logger.info(f"\n{'='*50}")
        logger.info(f"开始运行: {strategy['name']}")
        logger.info(f"{'='*50}")

        # 创建输出目录
        output_dir = os.path.join(base_output_dir, f"{timestamp}_{i+1}_{strategy['name']}")

        # 创建优化器
        optimizer = SteelFrameSA(config_path="config.yaml", sa_params=strategy["params"])

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

            # 记录对比结果
            stats = optimizer.get_optimization_statistics()
            comparison_results.append(
                {
                    "strategy": strategy["name"],
                    "optimization_time": optimization_time,
                    "best_objective": best_solution.objective_value,
                    "is_feasible": best_solution.is_feasible,
                    "max_stress": best_solution.max_stress,
                    "max_displacement": best_solution.max_displacement,
                    "total_iterations": stats["total_iterations"],
                    "acceptance_rate": stats["overall_acceptance_rate"],
                    "final_temperature": stats["final_temperature"],
                }
            )

            # 输出结果摘要
            logger.info(f"\n--- {strategy['name']} 结果摘要 ---")
            logger.info(f"目标函数值: {best_solution.objective_value:.6f} m³")
            logger.info(f"可行性: {best_solution.is_feasible}")
            if best_solution.is_feasible:
                logger.info(f"最大应力: {best_solution.max_stress:.2f} MPa")
                logger.info(f"最大位移: {best_solution.max_displacement:.2f} mm")
            logger.info(f"优化时间: {optimization_time:.2f} 秒")
            logger.info(f"总迭代次数: {stats['total_iterations']}")
            logger.info(f"接受率: {stats['overall_acceptance_rate']:.3f}")

        except Exception as e:
            logger.error(f"优化过程中出现错误: {e}")
            import traceback

            traceback.print_exc()

    # 保存对比结果
    if comparison_results:
        comparison_file = os.path.join(base_output_dir, f"{timestamp}_comparison_results.json")
        with open(comparison_file, "w", encoding="utf-8") as f:
            json.dump(convert_to_serializable(comparison_results), f, indent=2, ensure_ascii=False)
        logger.info(f"对比结果已保存到: {comparison_file}")


def run_single_optimization():
    """运行单次优化"""
    logger.info("开始钢框架结构模拟退火优化...")

    # 设置模拟退火参数
    sa_params = SAParameters(
        initial_temperature=1000.0,
        final_temperature=0.1,
        max_iterations=5000,
        max_iterations_per_temp=50,
        cooling_schedule=CoolingSchedule.EXPONENTIAL,
        cooling_rate=0.98,
        restart_threshold=100,
    )

    # 创建优化器
    optimizer = SteelFrameSA(config_path="config.yaml", sa_params=sa_params)

    # 显示优化设置
    logger.info(f"初始温度: {sa_params.initial_temperature}")
    logger.info(f"最终温度: {sa_params.final_temperature}")
    logger.info(f"最大迭代次数: {sa_params.max_iterations}")
    logger.info(f"每温度最大迭代次数: {sa_params.max_iterations_per_temp}")
    logger.info(f"冷却策略: {sa_params.cooling_schedule.value}")
    logger.info(f"冷却率: {sa_params.cooling_rate}")
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
        output_dir = f"results/sa_optimization/{timestamp}_single_run"
        save_optimization_results(optimizer, output_dir)

        # 获取统计信息
        stats = optimizer.get_optimization_statistics()

        # 输出最终结果
        print("\n" + "=" * 60)
        print("钢框架结构模拟退火优化结果")
        print("=" * 60)
        print(f"算法: 模拟退火 ({optimizer.sa_params.cooling_schedule.value}冷却)")
        print(f"优化时间: {optimization_time:.2f} 秒")
        print(f"总迭代次数: {stats['total_iterations']}")
        print(f"接受的移动: {stats['accepted_moves']}")
        print(f"拒绝的移动: {stats['rejected_moves']}")
        print(f"总体接受率: {stats['overall_acceptance_rate']:.3f}")
        print(f"最终温度: {stats['final_temperature']:.6f}")

        print(f"\n目标函数值: {best_solution.objective_value:.6f} m³")
        print(f"解的可行性: {'可行' if best_solution.is_feasible else '不可行'}")

        if best_solution.is_feasible:
            print(
                f"最大应力: {best_solution.max_stress:.2f} MPa (约束: ≤{optimizer.constraints.max_stress} MPa)"
            )
            print(
                f"最大位移: {best_solution.max_displacement:.2f} mm (约束: ≤{optimizer.constraints.max_displacement} mm)"
            )
        else:
            print(f"惩罚项: {best_solution.penalty:.2f}")

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


def run_parameter_sensitivity_analysis():
    """运行参数敏感性分析"""
    logger.info("开始模拟退火参数敏感性分析...")

    # 基础参数
    base_params = SAParameters(
        initial_temperature=1000.0,
        final_temperature=0.1,
        max_iterations=1500,
        max_iterations_per_temp=30,
        cooling_schedule=CoolingSchedule.EXPONENTIAL,
        cooling_rate=0.95,
    )

    # 参数变化范围
    parameter_variations = [
        {
            "name": "初始温度对比",
            "variations": [
                {"initial_temperature": 500.0},
                {"initial_temperature": 1000.0},
                {"initial_temperature": 2000.0},
                {"initial_temperature": 5000.0},
            ],
        },
        {
            "name": "冷却率对比",
            "variations": [
                {"cooling_rate": 0.90},
                {"cooling_rate": 0.95},
                {"cooling_rate": 0.98},
                {"cooling_rate": 0.99},
            ],
        },
        {
            "name": "每温度迭代次数对比",
            "variations": [
                {"max_iterations_per_temp": 20},
                {"max_iterations_per_temp": 50},
                {"max_iterations_per_temp": 100},
                {"max_iterations_per_temp": 200},
            ],
        },
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"results/sa_sensitivity/{timestamp}"

    for param_group in parameter_variations:
        logger.info(f"\n{'='*50}")
        logger.info(f"开始 {param_group['name']} 分析")
        logger.info(f"{'='*50}")

        group_results = []

        for j, variation in enumerate(param_group["variations"]):
            # 创建修改后的参数
            current_params = copy.deepcopy(base_params)
            for key, value in variation.items():
                setattr(current_params, key, value)

            logger.info(f"运行变化 {j+1}: {variation}")

            # 创建优化器
            optimizer = SteelFrameSA(config_path="config.yaml", sa_params=current_params)

            start_time = time.time()
            try:
                best_solution = optimizer.run_optimization()
                end_time = time.time()

                stats = optimizer.get_optimization_statistics()

                result = {
                    "variation": variation,
                    "optimization_time": end_time - start_time,
                    "best_objective": best_solution.objective_value,
                    "is_feasible": best_solution.is_feasible,
                    "max_stress": best_solution.max_stress,
                    "max_displacement": best_solution.max_displacement,
                    "statistics": stats,
                }

                group_results.append(result)

                logger.info(
                    f"结果: 目标值={best_solution.objective_value:.6f}, "
                    f"可行={best_solution.is_feasible}, "
                    f"时间={end_time - start_time:.2f}s"
                )

            except Exception as e:
                logger.error(f"参数组合 {variation} 优化失败: {e}")

        # 保存该参数组的结果
        group_output_dir = os.path.join(base_output_dir, param_group["name"].replace(" ", "_"))
        os.makedirs(group_output_dir, exist_ok=True)

        group_file = os.path.join(group_output_dir, "sensitivity_results.json")
        with open(group_file, "w", encoding="utf-8") as f:
            json.dump(convert_to_serializable(group_results), f, indent=2, ensure_ascii=False)

        logger.info(f"{param_group['name']} 结果已保存到: {group_file}")


def main():
    """主函数"""
    print("钢框架结构模拟退火优化程序")
    print("=" * 50)
    print("1. 运行单次优化")
    print("2. 运行多冷却策略对比")
    print("3. 运行参数敏感性分析")
    print("4. 退出")

    while True:
        choice = input("\n请选择运行模式 (1-4): ").strip()

        if choice == "1":
            run_single_optimization()
            break
        elif choice == "2":
            run_optimization_with_different_cooling_schedules()
            break
        elif choice == "3":
            run_parameter_sensitivity_analysis()
            break
        elif choice == "4":
            print("程序退出")
            break
        else:
            print("无效选择，请输入1-4")


if __name__ == "__main__":
    main()
