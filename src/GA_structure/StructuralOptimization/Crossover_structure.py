import random
from ..Common_components.Crossover import Crossover
from ..Common_components.Individual import Individual
from .Phenotype_structure import PhenotypeStructure


class CrossoverStructure(Crossover):
    def crossover(self, father, mother):
        """
        钢结构个体的交叉函数，返回一个新子代个体
        """
        father_params = father.phenotype.params
        mother_params = mother.phenotype.params

        child_params = {}

        for key in father_params:
            val_f = father_params[key]
            val_m = mother_params[key]

            # 浮点型参数：采用均值 + 扰动（BLX-α思想）
            if isinstance(val_f, float) or isinstance(val_m, float):
                alpha = 0.2
                cmin = min(val_f, val_m)
                cmax = max(val_f, val_m)
                range_expand = alpha * (cmax - cmin)
                new_val = random.uniform(cmin - range_expand, cmax + range_expand)
                child_params[key] = round(new_val, 3)  # 保留三位小数，可调

            # 整数型参数（如截面编号）：随机从父或母选择
            elif isinstance(val_f, int) and isinstance(val_m, int):
                child_params[key] = random.choice([val_f, val_m])

            else:
                # 默认：保留父代值
                child_params[key] = val_f

        # 构造表现型并返回个体
        phenotype = PhenotypeStructure(child_params)
        return Individual(phenotype)
