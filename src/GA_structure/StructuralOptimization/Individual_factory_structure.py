import random
import yaml

from .Phenotype_structure import PhenotypeStructure
from ..Common_components.Individual import Individual
from ..Common_components.Individual_factory import Individual_factory


class Individual_factory_Structure(Individual_factory):
    def __init__(self, config_path, optimize_spacing=True):
        # 读取配置文件
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        self.structural_params = config['structural_params']

        # 固定参数（不参与优化）
        self.fixed_keys = [
            'length', 'width', 'height',
            'win_bottom', 'win_height',
            'plate_thickness', 'pressure', 'windows'
        ]

        # 控制优化间距还是截面（人为选择）
        self.optimize_spacing = optimize_spacing

        # 分类优化参数
        self.spacing_keys = ['main_beam_spacing', 'secondary_beam_spacing', 'brace_spacing']
        self.section_keys = [
            'frame_section_index', 'main_beam_section_index',
            'secondary_beam_section_index', 'brace_section_index'
        ]

        # 决定优化哪些键
        if self.optimize_spacing:
            self.optimized_keys = self.spacing_keys
            self.fixed_keys += self.section_keys
        else:
            self.optimized_keys = self.section_keys
            self.fixed_keys += self.spacing_keys

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

        # 构建表现型并封装为个体
        phenotype = PhenotypeStructure(all_params)
        return Individual(phenotype)

    def _random_value_for_param(self, param_key):
        """
        根据参数的不同，返回对应范围内的随机值
        """
        if param_key == 'main_beam_spacing':
            return round(random.randint(50, 150) / 100.0, 2)

        elif param_key == 'secondary_beam_spacing':
            return round(random.randint(50, 150) / 100.0, 2)

        elif param_key == 'brace_spacing':
            return round(random.randint(10, 70) / 100.0, 2)

        elif param_key == 'frame_section_index':
            return random.choice([0, 1, 2, 3])

        elif param_key == 'main_beam_section_index':
            return random.randint(0, 10)

        elif param_key == 'secondary_beam_section_index':
            return random.randint(0, 9)

        elif param_key == 'brace_section_index':
            return random.randint(0, 19)

        raise ValueError(f"未知的参数键: {param_key}")
