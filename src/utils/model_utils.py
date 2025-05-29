import os
from enum import Enum
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import yaml

from src.structure.section_library import SectionLibrary
from src.structure.structural_model import BeamType, Point, StructuralModel, StructuralParameters, Window


def load_config(config_path):
    """加载配置文件"""
    if not os.path.exists(config_path):
        print(f"配置文件 {config_path} 不存在，使用默认配置")
        return {
            "structural_params": {
                "length": 7.2,
                "width": 3.6,
                "height": 3.3,
                "main_beam_spacing": 1.2,
                "secondary_beam_spacing": 1.2,
                "brace_spacing": 0.6,
                "windows": [
                    {"y_position": 3.6, "width": 1.2, "height": 1.5, "bottom_height": 1.0, "on_face": "front"}
                ],
            },
            "rl_params": {
                "actor_lr": 1e-4,
                "critic_lr": 1e-3,
                "gamma": 0.99,
                "tau": 0.005,
                "batch_size": 64,
                "buffer_size": 10000,
                "update_interval": 5,
            },
            "gnn_params": {"in_channels": 10, "hidden_channels": 64, "out_channels": 32},
        }

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def create_structural_model(config):
    """创建结构模型"""
    params_dict = config["structural_params"]

    # 处理窗户参数
    windows = []
    if "windows" in params_dict:
        for window_dict in params_dict["windows"]:
            window_dict = {
                "y_position": window_dict["y_position"],
                "width": params_dict["main_beam_spacing"],
                "height": params_dict["win_height"],
                "bottom_height": 0.0 if window_dict.get("is_door", False) else params_dict["win_bottom"],
                "on_face": window_dict["on_face"],
            }
            windows.append(Window(**window_dict))

    # 创建结构参数对象
    section_library = SectionLibrary()
    params = StructuralParameters(
        length=params_dict["length"],
        width=params_dict["width"],
        height=params_dict["height"],
        main_beam_spacing=params_dict["main_beam_spacing"],
        secondary_beam_spacing=params_dict["main_beam_spacing"],  # 次梁间距与主梁间距相同
        brace_spacing=params_dict["brace_spacing"],
        frame_section_index=params_dict["frame_section_index"],
        main_beam_section_index=params_dict["main_beam_section_index"],
        secondary_beam_section_index=params_dict["secondary_beam_section_index"],
        brace_section_index=params_dict["brace_section_index"],
        windows=windows,
    )

    # 创建结构模型并生成网格
    model = StructuralModel(params, section_library)

    # 使用网格化建模方法
    model.generate_grid_mesh()

    return model


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
            start_node = self.model.points[beam.start_node]
            end_node = self.model.points[beam.end_node]

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
        return fig, ax

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
