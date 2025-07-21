import random
import yaml

# 假设你有一个专用于结构优化问题的表现型类
from .Phenotype_structure import PhenotypeStructure

from ..Common_components.Individual import Individual
from ..Common_components.Individual_factory import Individual_factory




class Individual_factory_Structure(Individual_factory):
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        # with open(config_path, 'r') as file:
        #     config = yaml.safe_load(file)
        self.structural_params = config['structural_params']

        # 固定参数（不优化）
        self.fixed_keys = [
            'length', 'width', 'height',
            'win_bottom', 'win_height',
            'plate_thickness', 'pressure', 'windows'
        ]

        # 待优化的参数键
        self.optimized_keys = [
            k for k in self.structural_params.keys() if k not in self.fixed_keys
        ]

    def create_random_individual(self, n=0):
        # 构造优化参数（随机初始化）
        optimized_params = {
            k: self._random_value_for_param(k)
            for k in self.optimized_keys
        }

        # 加入固定参数
        all_params = {**optimized_params}
        for k in self.fixed_keys:
            all_params[k] = self.structural_params[k]

        # 构建表现型，并封装为个体
        phenotype = PhenotypeStructure(all_params)
        return Individual(phenotype)

        
    def _random_value_for_param(self, param_key):
        """
        可以根据具体结构优化参数的物理意义，定义不同的取值范围。
        这里我们使用简单的范围作为示例。
        """

        if param_key == 'main_beam_spacing':
            # 主梁间距：0.5 ~ 1.5，步长0.01
            return round(random.randint(50, 150) / 100.0, 2)

        elif param_key == 'secondary_beam_spacing':
            # 次梁间距：0.5 ~ 1.5，步长0.01
            return round(random.randint(50, 150) / 100.0, 2)

        elif param_key == 'brace_spacing':
            # 支撑间距：0.0 ~ 0.7，步长0.01
            return round(random.randint(30, 70) / 100.0, 2)

        elif param_key == 'frame_section_index':
            # 框架截面编号：1 ~ 4
            return random.choice([0, 1, 2, 3])

        elif param_key == 'main_beam_section_index':
            # 主梁截面编号：35 ~ 45
            return random.randint(0, 10)

        elif param_key == 'secondary_beam_section_index':
            # 次梁截面编号：25 ~ 34
            return random.randint(0, 9)

        elif param_key == 'brace_section_index':
            # 支撑截面编号：5 ~ 24
            return random.randint(0, 19)

    

