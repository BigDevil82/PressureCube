import os
import sys
from enum import Enum
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# 添加src目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.structure.structural_model import BeamType, Point, StructuralModel, StructuralParameters, Window


class ViewType(Enum):
    """视图类型枚举"""

    LEFT = "left"  # 左侧面
    RIGHT = "right"  # 右侧面
    FRONT = "front"  # 前面
    BACK = "back"  # 后面
    TOP = "top"  # 顶面
    BOTTOM = "bottom"  # 底面
    FULL_3D = "3d"  # 3D视图


class StructureVisualizer:
    """结构可视化工具类"""

    def __init__(self, model: StructuralModel):
        self.model = model
        self.colors = {
            BeamType.FRAME: "black",
            BeamType.MAIN: "blue",
            BeamType.SECONDARY: "green",
            BeamType.BRACE: "red",
        }
        self.view_params = {
            ViewType.LEFT: {"elev": 0, "azim": 90, "title": "Left Face View"},
            ViewType.RIGHT: {"elev": 0, "azim": 270, "title": "Right Face View"},
            ViewType.FRONT: {"elev": 0, "azim": 0, "title": "Front Face View"},
            ViewType.BACK: {"elev": 0, "azim": 180, "title": "Back Face View"},
            ViewType.TOP: {"elev": 90, "azim": 0, "title": "Top Face View"},
            ViewType.BOTTOM: {"elev": -90, "azim": 0, "title": "Bottom Face View"},
            ViewType.FULL_3D: {"elev": 20, "azim": 45, "title": "Pressure Vessel Structure (3D)"},
        }

    def _is_beam_in_face(self, beam: Tuple[Point, Point], face_type: ViewType) -> bool:
        """判断梁是否在指定面上"""
        start_node, end_node = beam
        if face_type == ViewType.LEFT:
            return abs(start_node.y) < 0.01 and abs(end_node.y) < 0.01
        elif face_type == ViewType.RIGHT:
            return (
                abs(start_node.y - self.model.params.length) < 0.01
                and abs(end_node.y - self.model.params.length) < 0.01
            )
        elif face_type == ViewType.FRONT:
            return abs(start_node.x) < 0.01 and abs(end_node.x) < 0.01
        elif face_type == ViewType.BACK:
            return (
                abs(start_node.x - self.model.params.width) < 0.01
                and abs(end_node.x - self.model.params.width) < 0.01
            )
        elif face_type == ViewType.TOP:
            return (
                abs(start_node.z - self.model.params.height) < 0.01
                and abs(end_node.z - self.model.params.height) < 0.01
            )
        elif face_type == ViewType.BOTTOM:
            return abs(start_node.z) < 0.01 and abs(end_node.z) < 0.01
        return True  # 3D视图显示所有梁

    def _create_plot(self, face_type: ViewType) -> Tuple[plt.Figure, plt.Axes]:
        """创建绘图对象"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # 设置坐标轴标签
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        # 设置坐标轴范围
        ax.set_xlim([0, self.model.params.width])
        ax.set_ylim([0, self.model.params.length])
        ax.set_zlim([0, self.model.params.height])

        # 设置视角
        view_params = self.view_params[face_type]
        ax.view_init(elev=view_params["elev"], azim=view_params["azim"])

        # 设置标题
        ax.set_title(view_params["title"])

        return fig, ax

    def _plot_beams(self, ax: plt.Axes, face_type: ViewType):
        """绘制梁构件"""
        for beam_id, beam in self.model.beams.items():
            start_node = self.model.nodes[beam.start_node]
            end_node = self.model.nodes[beam.end_node]

            if self._is_beam_in_face((start_node, end_node), face_type):
                x = [start_node.x, end_node.x]
                y = [start_node.y, end_node.y]
                z = [start_node.z, end_node.z]

                ax.plot(x, y, z, color=self.colors[beam.beam_type], linewidth=1.5)

    def _add_legend(self, ax: plt.Axes):
        """添加图例"""
        for beam_type, color in self.colors.items():
            ax.plot([], [], [], color=color, label=beam_type.name)
        ax.legend()

    def show_face(self, face_type: ViewType):
        """显示指定面的视图"""
        fig, ax = self._create_plot(face_type)
        self._plot_beams(ax, face_type)
        self._add_legend(ax)
        plt.axis("equal")
        plt.show()

    def show_all_views(self):
        """显示所有视图"""
        # 显示六个面的视图
        for face_type in [
            ViewType.LEFT,
            ViewType.RIGHT,
            ViewType.FRONT,
            ViewType.BACK,
            ViewType.TOP,
            ViewType.BOTTOM,
        ]:
            self.show_face(face_type)

        # 显示3D视图
        self.show_face(ViewType.FULL_3D)


def test_side_faces():
    """测试左右侧面生成"""
    # 创建结构参数，包含窗户和门
    params = StructuralParameters(
        length=9.5,  # 舱体长度 (y方向)
        width=3.6,  # 舱体宽度 (x方向)
        height=3.3,  # 舱体高度 (z方向)
        main_beam_spacing=1.2,  # 环向主梁间距
        secondary_beam_spacing=1.2,  # 环向次梁间距
        brace_spacing=0.6,  # 支撑梁间距
        windows=[
            # 左侧面窗户
            Window(
                y_position=1.2,  # 窗户中心在x轴的位置(对于侧面)
                width=1.2,  # 窗户宽度
                height=1.0,  # 窗户高度
                bottom_height=1.0,  # 窗户底部离地高度
                on_face="left",  # 窗户所在面
            ),
            # 右侧面门
            Window(
                y_position=1.2,
                width=1.2,
                height=2.0,
                bottom_height=0,  # 门从地面开始
                on_face="right",
                is_door=True,  # 这是一个门
            ),
            # 保留前后面的窗户，确保完整性
            Window(
                y_position=1.8,
                width=1.2,
                height=1.0,
                bottom_height=1.0,
                on_face="front",
            ),
            Window(
                y_position=3.0,
                width=1.2,
                height=1.0,
                bottom_height=1.0,
                on_face="back",
            ),
            Window(
                y_position=4.2,
                width=1.2,
                height=1.0,
                bottom_height=1.0,
                on_face="back",
            ),
        ],
    )

    # 创建结构模型
    model = StructuralModel(params)

    # 生成结构网格
    model.generate_grid_mesh()

    # 输出结构信息
    print("\n左右侧面测试:")
    print(f"生成的节点数量: {len(model.nodes)}")
    print(f"生成的梁构件数量: {len(model.beams)}")
    print(f"结构总质量: {model.calculate_mass():.2f} kg")

    # 分析各类型梁的数量和总长度
    beam_type_counts = {beam_type: 0 for beam_type in BeamType}
    beam_type_lengths = {beam_type: 0.0 for beam_type in BeamType}

    for beam in model.beams.values():
        beam_type_counts[beam.beam_type] += 1
        beam_type_lengths[beam.beam_type] += beam.length(model.nodes)

    for beam_type in BeamType:
        print(
            f"{beam_type.name} 梁数量: {beam_type_counts[beam_type]}, "
            f"总长度: {beam_type_lengths[beam_type]:.2f} m"
        )

    # 返回模型供后续可能的可视化
    return model


if __name__ == "__main__":
    # 测试左右侧面
    model = test_side_faces()

    # 创建可视化器并显示所有视图
    visualizer = StructureVisualizer(model)
    visualizer.show_all_views()
