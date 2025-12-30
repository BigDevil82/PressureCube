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

from src.heuristic.ga.steel_frame_ga import SteelFrameGA
from src.structure.structural_model import StructuralParameters
from src.utils.logger import Logger

chinese_font = "SimSun"  # 宋体
english_font = "Times New Roman"
plt.rcParams["font.sans-serif"] = [chinese_font, english_font]
plt.rcParams["font.serif"] = [english_font, chinese_font]


def main():
    # 1. 定义基础几何参数 (优化过程中不变量)
    base_params = StructuralParameters(
        length=7.2,
        width=3.7,
        height=3.3,
        # 下面这些将被 GA 覆盖，给个初值即可
        main_beam_spacing=0.7,
        num_secondary_width=4,
        num_secondary_height=4,
        add_support_beam=True,
        plate_thickness=0.005,  # 假定板厚不参与优化，或在 GA 中添加此基因
    )

    print("=== Starting Structural Optimization ===")
    start_time = time.time()

    # 2. 初始化优化器
    optimizer = SteelFrameGA(base_params)

    # 3. 运行遗传算法
    best_solution = optimizer.evolve()

    total_time = time.time() - start_time
    print(f"\nOptimization Finished in {total_time:.1f}s")

    # 4. 打印最终结果
    print("\n====== Best Solution Found ======")
    genes = best_solution.genes
    print(f"Feasible: {best_solution.is_feasible}")
    print(f"Mass: {best_solution.mass:.2f} kg")
    print(f"Max Stress Ratio: {best_solution.max_stress_ratio:.3f}")
    print(f"Max Deflection Ratio: {best_solution.max_deflection_ratio:.3f}")

    print("-" * 30)
    print("Design Variables:")
    # 简单的索引解码
    sec_types = ["Frame (Box)", "Main (H)", "Sec (H)", "Brace (C)"]
    for i in range(4):
        idx = int(genes[i])
        # 这里为了演示简单，直接打印索引，实际可调用 optimizer.section_lib.get_section 获取名字
        print(f"  {sec_types[i]} Index: {idx}")

    # print(f"  Main Beam Spacing: {genes[4]:.3f} m")
    # print(f"  Brace Spacing: {genes[5]:.3f} m (Approx)")

    # 5. 简单绘图 (收敛曲线)
    history = optimizer.best_history
    fitness_vals = [ind.fitness for ind in history]
    mass_vals = [ind.mass for ind in history]

    fig, ax1 = plt.subplots()

    color = "tab:red"
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color=color)
    ax1.plot(fitness_vals, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Mass (kg)", color=color)
    ax2.plot(mass_vals, color=color, linestyle="--")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Optimization Convergence")
    plt.tight_layout()
    # plt.show()
    save_dir = "results/GA_optimization"
    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    save_dir = os.path.join(save_dir, time_stamp)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "convergence_curve.png"))

    # save results to a json file
    results = {
        "feasible": best_solution.is_feasible,
        "mass_kg": best_solution.mass,
        "max_stress_ratio": best_solution.max_stress_ratio,
        "max_deflection_ratio": best_solution.max_deflection_ratio,
        "design_variables": {
            "frame_section_index": int(genes[0]),
            "main_beam_section_index": int(genes[1]),
            "secondary_beam_section_index": int(genes[2]),
            "brace_section_index": int(genes[3]),
            # "main_beam_spacing": genes[4],
            # "brace_spacing_approx": genes[5],
        },
        "base_parameters": {
            "length_m": base_params.length,
            "width_m": base_params.width,
            "height_m": base_params.height,
            "main_beam_spacing_m": base_params.main_beam_spacing,
            "num_secondary_width": base_params.num_secondary_width,
            "num_secondary_height": base_params.num_secondary_height,
            "add_support_beam": base_params.add_support_beam,
            "plate_thickness_m": base_params.plate_thickness,
        },
    }

    with open(os.path.join(save_dir, "best_solution.json"), "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
