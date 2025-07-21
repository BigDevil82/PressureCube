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

# 设置中文字体以防止乱码（自动判断操作系统）
if platform.system() == 'Windows':
    matplotlib.rcParams['font.family'] = 'SimHei'  # 黑体（Windows 常用中文字体）
elif platform.system() == 'Darwin':
    matplotlib.rcParams['font.family'] = 'Heiti TC'  # macOS 中文字体
else:
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # Linux（支持部分中文）
    
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号


if __name__ == '__main__':
    mutation_rate_list = np.arange(0.0, 1.0, 0.1).tolist()
    iterations_list = [10,20, 50]
    population_sizes = [10, 20, 50]
    sample_num_per_setting = 5

    column_names = ['mutation_rate', 'iterations_number', 'population_size', 'average_fitness']
    df = pd.DataFrame(columns=column_names)

    best_run_history = []  # 最优个体所在演化过程的适应度历史
    tracking = {
        "best_individual": None,
        "best_fitness": float("inf")
    }

    start = time.time()

    for mutation_rate in mutation_rate_list:
        for num_iterations in iterations_list:
            for pop_size in population_sizes:
                results = []

                for sample_i in range(sample_num_per_setting):
                    crossover = CrossoverStructure()
                    fitness_obj = FitnessStructure()
                    factory = Individual_factory_Structure("config.yaml")
                    mutator = MutatorStructure(rate=mutation_rate)
                    parents_selector = Parents_selector()

                    population = Population()
                    population.create_first_population(size_pop=pop_size, ind_factory=factory)

                    config_object = Configuration(population, crossover, mutator, fitness_obj, factory, parents_selector)
                    evolution = Genetic_search(config_object, pop_size, population)

                    fitness_history = []

                    def hook_callback(best_individual):
                        fitness_history.append(best_individual.fitness_score)
                        if best_individual.fitness_score < tracking["best_fitness"]:
                            tracking["best_fitness"] = best_individual.fitness_score
                            tracking["best_individual"] = best_individual
                            # 如果发现新的最优个体，就记录这个过程的历史
                            best_run_history.clear()
                            best_run_history.extend(fitness_history)

                    final_fitness = evolution.optimize(num_iterations, callback=hook_callback)
                    results.append(final_fitness)

                tmp_df = pd.DataFrame(columns=column_names, data=[[mutation_rate, num_iterations, pop_size, np.mean(results)]])
                df = pd.concat([df, tmp_df], ignore_index=True)

            print(f'已完成 mutation rate = {mutation_rate:.2f}')

    end = time.time()
    print('总耗时（秒）:', round(end - start, 2))

    # 保存所有实验组合结果
    df.to_csv('structure_optimization_results.csv', index=False)
    print('所有组合结果已保存为 structure_optimization_results.csv')

    # 保存最优个体参数并输出适应度
    if tracking["best_individual"] is not None:
        with open("best_individual.json", "w", encoding="utf-8") as f:
            json.dump(tracking["best_individual"].phenotype.params, f, indent=2, ensure_ascii=False)
        print(f"最优个体参数已保存至 best_individual.json")
        print(f"最优适应度（结构总质量）为：{tracking['best_fitness']:.2f} kg")

    # 绘制最佳运行过程的适应度变化图
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

