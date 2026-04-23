import os
from enum import Enum
from typing import Tuple

import matplotlib.pyplot as plt
import yaml
from mpl_toolkits.mplot3d import Axes3D

from src.structure.enums import BeamType
from src.structure.layout.params import StructuralParameters
from src.structure.section_library import SectionLibrary
from src.structure.structural_model import Beam, Point, StructuralModel


def load_config(config_path):
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在")

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
                "is_door": window_dict.get("is_door", False),
            }

    # 创建结构参数对象
    section_library = SectionLibrary()
    params = StructuralParameters(
        length=params_dict["length"],
        width=params_dict["width"],
        height=params_dict["height"],
        main_beam_spacing=params_dict["main_beam_spacing"],
        num_secondary_width=params_dict["num_secondardy_witdth"],
        num_secondary_height=params_dict["num_secondardy_height"],
        add_support_beam=params_dict["add_support_beam"],
        frame_section_index=params_dict["frame_section_index"],
        main_beam_section_index=params_dict["main_beam_section_index"],
        secondary_beam_section_index=params_dict["secondary_beam_section_index"],
        brace_section_index=params_dict["brace_section_index"],
        plate_thickness=params_dict["plate_thickness"],
        windows=windows,
    )

    # 创建结构模型并生成网格
    model = StructuralModel(params, section_library)

    # 使用网格化建模方法
    model.generate()

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
            BeamType.FRAME: "brown",
            BeamType.MAIN: "blue",
            BeamType.SECONDARY: "orange",
            BeamType.BRACE: "green",
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

    def _is_beam_in_face(self, beam: Beam, face_type: ViewType) -> bool:
        """判断梁是否在指定面上"""
        if face_type == ViewType.FULL_3D:
            return True

        if face_type.value in beam.on_face:
            return True
        else:
            return False

    def _create_plot(self, face_type: ViewType) -> Tuple[plt.Figure, plt.Axes]:
        """创建绘图对象"""
        fig = plt.figure(figsize=(10, 8))
        ax: Axes3D = fig.add_subplot(111, projection="3d")

        # 设置坐标轴标签
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        # 设置坐标轴范围
        # 获取模型的最大尺寸
        L = self.model.params.width
        W = self.model.params.length
        H = self.model.params.height

        ax.set_xlim([0, L])
        ax.set_ylim([0, W])
        ax.set_zlim([0, H])
        ax.set_box_aspect((L, W, H))

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

            if self._is_beam_in_face(beam, face_type):
                x = [start_node.x, end_node.x]
                y = [start_node.y, end_node.y]
                z = [start_node.z, end_node.z]
                alpha = 0.2 if beam.is_virtual else 1.0
                ax.plot(x, y, z, color=self.colors[beam.beam_type], linewidth=1.5, zorder=10, alpha=alpha)

    def _plot_plates(self, ax: plt.Axes, face_type: ViewType):
        """绘制钢板"""
        for plate_id, plate in self.model.plates.items():
            # 获取板的四个角点坐标
            corner_points = [self.model.points[node_id] for node_id in plate.nodes]

            # 添加第一个点以闭合多边形
            corner_points.append(corner_points[0])

            # 提取x, y, z坐标
            x = [point.x for point in corner_points]
            y = [point.y for point in corner_points]
            z = [point.z for point in corner_points]

            # 根据板所在面判断是否需要显示
            if face_type == ViewType.FULL_3D or plate.on_face.lower() == face_type.value:
                # 绘制板的边界
                ax.plot(x, y, z, color="gray", linestyle="--", linewidth=0.7, alpha=0.8, zorder=5)

                # 填充板面
                # verts = [list(zip(x[:-1], y[:-1], z[:-1]))]
                # poly = Poly3DCollection(verts, alpha=0.6, facecolor="lightgray", edgecolor=None, zorder=5)
                # ax.add_collection3d(poly)

    def _plot_nodes(self, ax: plt.Axes, face_type: ViewType):
        """绘制节点"""
        # 收集所有节点的坐标
        x, y, z = [], [], []
        for node_id, point in self.model.points.items():
            # 对于2D视图，只显示在对应面上的节点
            if face_type == ViewType.FULL_3D:
                x.append(point.x)
                y.append(point.y)
                z.append(point.z)
            elif (
                (face_type == ViewType.LEFT and abs(point.y) < 0.01)
                or (face_type == ViewType.RIGHT and abs(point.y - self.model.params.length) < 0.01)
                or (face_type == ViewType.FRONT and abs(point.x) < 0.01)
                or (face_type == ViewType.BACK and abs(point.x - self.model.params.width) < 0.01)
                or (face_type == ViewType.TOP and abs(point.z - self.model.params.height) < 0.01)
                or (face_type == ViewType.BOTTOM and abs(point.z) < 0.01)
            ):
                x.append(point.x)
                y.append(point.y)
                z.append(point.z)

        # 绘制节点
        if x:  # 只在有节点时绘制
            ax.scatter(x, y, z, color="gray", s=5, marker="o", alpha=0.6, zorder=1, label="Node")

    def _add_legend(self, ax: plt.Axes):
        """添加图例"""
        for beam_type, color in self.colors.items():
            ax.plot([], [], [], color=color, label=beam_type.name)
        # 添加板的图例
        ax.plot([], [], [], color="gray", linestyle="--", label="Plate Edge")
        # 添加板填充的图例
        # poly = plt.Rectangle((0, 0), 1, 1, fc="lightgray", alpha=0.2, label="Plate Area")
        # ax.add_artist(plt.legend([poly], ["Plate Area"], loc="upper right"))
        # 添加节点的图例（只添加一次）
        if not any(item.get_label() == "Node" for item in ax.get_legend_handles_labels()[0]):
            ax.scatter([], [], [], color="gray", s=5, marker="o", alpha=0.6, label="Node")
        ax.legend()

    def show_face(self, face_type: ViewType, grid: bool = True):
        """显示指定面的视图"""
        fig, ax = self._create_plot(face_type)
        self._plot_plates(ax, face_type)  # 先绘制板（在底层）
        self._plot_beams(ax, face_type)  # 再绘制梁
        self._plot_nodes(ax, face_type)  # 最后绘制节点（在顶层）
        self._add_legend(ax)
        ax.grid(grid)
        plt.tight_layout(pad=1.2)
        plt.show()
        return fig, ax

    def show_all_views(self, grid: bool = True):
        """显示所有视图"""
        # 显示六个面的视图
        for face_type in [
            ViewType.LEFT,
            ViewType.RIGHT,
            ViewType.FRONT,
            ViewType.BACK,
            ViewType.TOP,
            ViewType.BOTTOM,
            ViewType.FULL_3D,
        ]:
            self.show_face(face_type, grid)
