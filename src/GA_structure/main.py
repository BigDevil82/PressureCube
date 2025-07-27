import numpy as np
import pandas as pd
import time
import json
import matplotlib.pyplot as plt
import matplotlib
import platform

from .StructuralOptimization.Crossover_structure import CrossoverStructure
from .StructuralOptimization.Fitness_structure import FitnessStructure
from .StructuralOptimization.Mutator_structure import MutatorStructure
from .Common_components.Configuration import Configuration
from .Common_components.Evolution import Genetic_search
from .Common_components.Population import Population
from .Common_components.parents_selector.Parents_selector import Parents_selector
from .StructuralOptimization.Individual_factory_structure import Individual_factory_Structure
from ..utils.GA_utils import setup_logger

# 设置中文字体（防止中文乱码）
if platform.system() == 'Windows':
    matplotlib.rcParams['font.family'] = 'SimHei'
elif platform.system() == 'Darwin':
    matplotlib.rcParams['font.family'] = 'Heiti TC'
else:
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# 初始化日志器
logger = setup_logger()


if __name__ == '__main__':
    mutation_rate_list = np.arange(0.0, 0.5, 0.1).tolist()
    iterations = 30
    population_sizes = 30
    sample_num_per_setting = 1

    column_names = ['mutation_rate', 'iterations_number', 'population_size', 'average_fitness']
    df = pd.DataFrame(columns=column_names)

    best_run_history = []
    tracking = {
        "best_individual": None,
        "best_fitness": float("inf")
    }

    total_tasks = len(mutation_rate_list) * iterations * population_sizes * sample_num_per_setting
    task_counter = 0
    start_time = time.time()

    logger.info(f"开始结构优化实验，总任务数：{total_tasks}")

    for mutation_rate in mutation_rate_list:
        results = []

        for sample_i in range(sample_num_per_setting):
            task_counter += 1
            logger.info(f"当前任务 [{task_counter}/{total_tasks}] - μ={mutation_rate:.2f}, "
                        f"iter={iterations}, pop={population_sizes}, run={sample_i + 1}")

            t0 = time.time()

            crossover = CrossoverStructure()
            fitness_obj = FitnessStructure()

            # 控制是否优化间距（True表示固定截面，False表示固定间距）
            factory = Individual_factory_Structure("config.yaml", optimize_spacing=False)

            mutator = MutatorStructure(rate=mutation_rate)
            parents_selector = Parents_selector()
            population = Population()
            population.create_first_population(size_pop=population_sizes, ind_factory=factory)

            config_object = Configuration(population, crossover, mutator, fitness_obj, factory, parents_selector)
            evolution = Genetic_search(config_object, population_sizes, population)

            fitness_history = []

            def hook_callback(best_individual):
                fitness_history.append(best_individual.fitness_score)
                if best_individual.fitness_score < tracking["best_fitness"]:
                    tracking["best_fitness"] = best_individual.fitness_score
                    tracking["best_individual"] = best_individual
                    best_run_history.clear()
                    best_run_history.extend(fitness_history)

            final_fitness = evolution.optimize(iterations, callback=hook_callback)
            results.append(final_fitness)

            t1 = time.time()
            elapsed = t1 - t0
            estimated_remaining = (t1 - start_time) / task_counter * (total_tasks - task_counter)
            logger.info(f"当前任务完成，用时 {elapsed:.2f} 秒，估计剩余 {estimated_remaining / 60:.1f} 分钟")

        tmp_df = pd.DataFrame(columns=column_names, data=[[mutation_rate, iterations, population_sizes, np.mean(results)]])
        df = pd.concat([df, tmp_df], ignore_index=True)

        logger.info(f"mutation rate = {mutation_rate:.2f} 的全部组合完成")

    total_elapsed = time.time() - start_time
    logger.info(f" 所有任务完成，总耗时：{total_elapsed:.2f} 秒")

    # 保存结果
    df.to_csv('structure_optimization_results.csv', index=False)
    logger.info("所有组合结果已保存为 structure_optimization_results.csv")

    # 保存最优个体
    if tracking["best_individual"] is not None:
        with open("best_individual.json", "w", encoding="utf-8") as f:
            json.dump(tracking["best_individual"].phenotype.params, f, indent=2, ensure_ascii=False)
        logger.info(f"最优个体参数已保存至 best_individual.json")
        logger.info(f"最优适应度（结构总质量）为：{tracking['best_fitness']:.2f} kg")
        print(f"最优适应度为：{tracking['best_fitness']:.2f} kg")

    # 绘图
    if best_run_history:
        plt.figure(figsize=(10, 6))
        plt.plot(best_run_history, marker='o', color='blue')
        plt.xlabel("迭代次数")
        plt.ylabel("最优适应度（结构总质量）")
        plt.title("最优运行过程中的适应度演化")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("fitness_evolution.png", dpi=300)
        plt.show()
