from abc import ABC
from ..Common_components.Fitness_obj import Fitness_calculator


class FitnessStructure(Fitness_calculator, ABC):
    """
    钢结构优化问题的适应度评估器
    - 目标：最小化结构总质量（越小越好）
    - 约束：最大应力、最大位移、最大层间位移角
    """

    def calculate_fitness_individual(self, individual):
        """
        为个体计算适应度（结构质量），并赋值给 individual.fitness_score
        :param individual: 个体对象，包含 phenotype
        :return: 适应度（质量）
        """
        fitness = individual.phenotype.get_objective()
        individual.fitness_score = fitness
        return fitness

    def test_solution_is_feasible(self, individual):
        """
        检查个体是否满足所有结构约束条件
        :param individual: 个体对象
        :return: True 表示可行，False 表示违反约束
        """
        return individual.phenotype.is_feasible()
