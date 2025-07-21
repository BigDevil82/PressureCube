# test_individual_fitness.py
from .StructuralOptimization.Individual_factory_structure import Individual_factory_Structure
from .StructuralOptimization.Fitness_structure import FitnessStructure
from .Common_components.Population import Population

if __name__ == "__main__":
    # 1. 创建个体工厂，传入配置文件
    factory = Individual_factory_Structure(config_path="config.yaml")

    # 2. 创建一个个体（测试用）
    individual = factory.create_random_individual()

    # 3. 输出参数
    print("\n 个体参数如下：")
    for k, v in individual.phenotype.params.items():
        print(f"  - {k}: {v}")

    # 4. 创建适应度评估器
    fitness_obj = FitnessStructure()

    # 5. 评估适应度
    score = fitness_obj.calculate_fitness_individual(individual)
    print(f"\n📏 结构总质量（适应度）：{score:.2f} kg")

    # 6. 检查是否满足约束
    feasible = fitness_obj.test_solution_is_feasible(individual)
    print(f"✅ 是否满足约束条件：{'是' if feasible else '否'}")
