import math
from dataclasses import field
from typing import Dict, List, Union

import numpy as np

from src.structure.section_library import CSection, HSection, RectangularTubeSection, SectionLibrary
from src.structure.structural_model import StructuralParameters, Window
from src.utils.model_utils import ViewType


class RandomStructure:
    """随机生成舱体结构参数的类"""

    def __init__(self, section_library: SectionLibrary):
        """
        初始化RandomStructure。

        Args:
            section_library (SectionLibrary): 截面库实例。
        """
        self.section_library = section_library
        self.section_counts = self._get_section_counts()

        # 定义参数的随机取值范围
        self._length_options = np.arange(7.2, 12.01, 0.3)
        self._main_beam_spacing_options = np.arange(0.7, 1.51, 0.05)

        # 固定参数
        self.fixed_width = 3.7
        self.fixed_height = 3.3
        self.fixed_win_height = 1.5
        self.fixed_win_bottom = 0.9
        self.fixed_plate_thickness = 0.005
        self._min_brace_spacing = 0.3
        self._min_grid_size = 0.2

    def _get_section_counts(self) -> Dict[str, int]:
        """获取每种截面类型的数量"""
        return {
            "方钢": len(self.section_library.sections["方钢"]),
            "H型钢-主梁": len(self.section_library.sections["H型钢-主梁"]),
            "H型钢-次梁": len(self.section_library.sections["H型钢-次梁"]),
            "C型钢": len(self.section_library.sections["C型钢"]),
        }

    def _generate_random_windows(self, length: float, main_beam_spacing: float) -> List[Window]:
        """根据规则生成随机窗户"""
        windows = []

        # 左面窗户
        pos_x_options = [
            self.fixed_width / 2 - main_beam_spacing / 2,
            self.fixed_width / 2 + main_beam_spacing / 2,
        ]
        for y_pos_on_left_face in pos_x_options:
            if np.random.random() < 0.5:  # 50%概率在左侧开窗
                windows.append(
                    Window(
                        y_position=y_pos_on_left_face.round(3),
                        width=main_beam_spacing,
                        height=self.fixed_win_height,
                        bottom_height=self.fixed_win_bottom,
                        on_face="left",
                    )
                )

        # 前、后面窗户
        num_bays = 1
        y = main_beam_spacing
        while length - y > self._min_grid_size:
            y += main_beam_spacing
            num_bays += 1

        possible_bays = list(range(1, num_bays - 1))  # 第2跨到倒数第二跨

        if len(possible_bays) > 0:
            for face in ["front", "back"]:
                num_windows = np.random.randint(1, min(4, len(possible_bays) + 1))  # 最多3个窗户
                selected_bays = np.random.choice(possible_bays, num_windows, replace=False)

                for bay_index in selected_bays:
                    # 窗户中线在所选跨的中间
                    y_pos = bay_index * main_beam_spacing + main_beam_spacing / 2
                    windows.append(
                        Window(
                            y_position=y_pos.round(3),
                            width=main_beam_spacing,
                            height=self.fixed_win_height,
                            bottom_height=self.fixed_win_bottom,
                            on_face=face,
                        )
                    )
        return windows

    def generate_random_parameters(self, generate_windows: bool = True) -> StructuralParameters:
        """
        生成一组随机的结构参数。

        Args:
            generate_windows (bool): 是否生成窗户。

        Returns:
            StructuralParameters: 随机生成的结构参数实例。
        """
        length = np.random.choice(self._length_options).round(3)
        main_beam_spacing = np.random.choice(self._main_beam_spacing_options).round(3)

        # 支撑梁间距
        num_spans = np.random.randint(1, 4)
        if num_spans > 1:
            brace_spacing = (main_beam_spacing / num_spans).round(3)
        else:
            brace_spacing = (1.3 * (main_beam_spacing / 2)).round(3)
        if brace_spacing < self._min_brace_spacing:
            brace_spacing = self._min_brace_spacing

        # 次梁间距，与主梁间距相同
        secondary_beam_spacing = main_beam_spacing

        # 随机选择截面索引
        frame_section_index = np.random.randint(0, self.section_counts["方钢"])
        main_beam_section_index = np.random.randint(0, self.section_counts["H型钢-主梁"])
        secondary_beam_section_index = np.random.randint(0, self.section_counts["H型钢-次梁"])
        brace_section_index = np.random.randint(0, self.section_counts["C型钢"])

        # 生成随机窗户
        windows = []
        if generate_windows:
            windows = self._generate_random_windows(length, main_beam_spacing)

        # 随机选择内压
        pressure = np.random.randint(28, 38)

        return StructuralParameters(
            length=length,
            width=self.fixed_width,
            height=self.fixed_height,
            main_beam_spacing=main_beam_spacing,
            secondary_beam_spacing=secondary_beam_spacing,
            brace_spacing=brace_spacing,
            windows=windows,
            frame_section_index=frame_section_index,
            main_beam_section_index=main_beam_section_index,
            secondary_beam_section_index=secondary_beam_section_index,
            brace_section_index=brace_section_index,
            win_bottom=self.fixed_win_bottom,
            win_height=self.fixed_win_height,
            plate_thickness=self.fixed_plate_thickness,
            pressure=pressure,
        )


if __name__ == "__main__":
    from traceback import print_exc

    from src.analysis.opensees_analysis import OpenSeesAnalyzer
    from src.structure.mesh_generator import MeshGenerator, MeshParameters
    from src.structure.structural_model import StructuralModel
    from src.utils.model_utils import StructureVisualizer
    from src.utils.util import Timer

    np.random.seed(10)

    section_library = SectionLibrary()
    random_structure = RandomStructure(section_library)

    for i in range(100):
        print(f"Generating random model {i}")
        try:
            random_parameters = random_structure.generate_random_parameters()
            random_model = StructuralModel(random_parameters, section_library)
            random_model.generate_grid_mesh()

            mesh_params = MeshParameters(max_grid_size=0.2)
            mesh_generator = MeshGenerator(random_model, mesh_params)
            mesh_generator.generate_mesh()
            model = mesh_generator.meshed_model
            print("结构总质量: ", model.calculate_mass() / 1000, "t")
            # visualizer = StructureVisualizer(model)
            # # visualizer.show_all_views()
            # visualizer.show_face(ViewType.FRONT, False)

            analyzer = OpenSeesAnalyzer()

            with Timer("setup opensees model cost"):
                analyzer.setup_opensees_model(model)
                analyzer.apply_pressure_loads(internal_pressure_pa=30000.0)
            with Timer("static analysis cost"):
                analyzer.run_static_analysis()
            max_indicators = analyzer.get_max_indicators()
            print(
                "max stress: ",
                max_indicators["max_abs_stress"],
                "max_disp: ",
                max_indicators["max_abs_displacement"],
            )
            print("-" * 100 + "\n")
        except Exception as e:
            print_exc()
            print(f"Error generating random model {i}: {e}")
            print(f"Random parameters: {random_parameters}")
            print("-" * 100 + "\n" * 2)
            visualizer = StructureVisualizer(random_model)
            visualizer.show_all_views()
