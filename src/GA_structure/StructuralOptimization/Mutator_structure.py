import random
from ..Common_components.Individual import Individual
from .Phenotype_structure import PhenotypeStructure


class MutatorStructure:
    def __init__(self, rate):
        self.rate = rate

        # 定义可变异参数及其范围
        self.param_ranges = {
            "main_beam_spacing": (0.5, 1.5, "float"),
            "secondary_beam_spacing": (0.5, 1.5, "float"),
            "brace_spacing": (0.1, 0.7, "float"),
            "frame_section_index": (0, 3, "int"),
            "main_beam_section_index": (0, 10, "int"),
            "secondary_beam_section_index": (0, 9, "int"),
            "brace_section_index": (0, 19, "int"),
        }

    def mutate(self, individual):
        """
        对个体执行一次变异：随机选择一个参数进行扰动（在其定义范围内）。
        """
        if random.random() >= self.rate:
            # 没触发变异概率，直接返回原始个体的副本
            return Individual(PhenotypeStructure(individual.phenotype.params.copy()))

        # 获取参数字典副本
        mutated_params = individual.phenotype.params.copy()

        # 从所有可优化参数中随机选一个进行变异
        param_to_mutate = random.choice(list(self.param_ranges.keys()))
        min_val, max_val, param_type = self.param_ranges[param_to_mutate]

        # 根据类型决定变异方式
        if param_type == "int":
            mutated_params[param_to_mutate] = random.randint(min_val, max_val)
        elif param_type == "float":
            mutated_params[param_to_mutate] = round(random.uniform(min_val, max_val), 2)

        # 创建新个体
        new_phenotype = PhenotypeStructure(mutated_params)
        return Individual(new_phenotype)
