import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


class BeamType(Enum):
    """梁构件类型枚举"""

    FRAME = 0  # 框架梁
    MAIN = 1  # 环向主梁
    SECONDARY = 2  # 环向次梁
    BRACE = 3  # 支撑梁


@dataclass
class Section:
    """截面属性类"""

    name: str
    area: float  # 截面积 (m^2)
    Ix: float  # 惯性矩 (m^4)
    Iy: float
    Iz: float
    Zx: float  # 截面模量 (m^3)
    Zy: float
    Zz: float
    density: float = 7850.0  # 材料密度 (kg/m^3)


@dataclass
class Node:
    """节点类"""

    id: int
    x: float
    y: float
    z: float

    def coordinates(self) -> np.ndarray:
        """返回节点坐标"""
        return np.array([self.x, self.y, self.z])

    def distance_to(self, other: "Node") -> float:
        """计算到另一个节点的距离"""
        return np.linalg.norm(self.coordinates() - other.coordinates())


@dataclass
class Beam:
    """梁构件类"""

    id: int
    start_node: int  # 起始节点ID
    end_node: int  # 终止节点ID
    beam_type: BeamType  # 梁类型
    section: str  # 截面类型

    def length(self, nodes: Dict[int, Node]) -> float:
        """计算梁长度"""
        if self.start_node not in nodes or self.end_node not in nodes:
            raise ValueError(f"Node {self.start_node} or {self.end_node} not found")
        return nodes[self.start_node].distance_to(nodes[self.end_node])

    def direction(self, nodes: Dict[int, Node]) -> np.ndarray:
        """计算梁的方向向量"""
        start = nodes[self.start_node].coordinates()
        end = nodes[self.end_node].coordinates()
        direction = end - start
        return direction / np.linalg.norm(direction)


@dataclass
class Window:
    """窗户类"""

    y_position: float  # 窗户中心在y轴的位置
    width: float  # 窗户宽度
    height: float  # 窗户高度
    bottom_height: float  # 窗户底部离地高度
    on_face: str  # 窗户所在面 ('front', 'back', 'right')


@dataclass
class StructuralParameters:
    """结构参数类"""

    length: float  # 舱体长度 (y方向)
    width: float  # 舱体宽度 (x方向)
    height: float  # 舱体高度 (z方向)
    main_beam_spacing: float  # 环向主梁间距
    secondary_beam_spacing: float  # 环向次梁间距
    brace_spacing: float  # 支撑梁间距
    windows: List[Window] = field(default_factory=list)  # 窗户参数列表


class StructuralModel:
    """舱体结构模型类"""

    def __init__(self, params: StructuralParameters):
        self.params = params
        self.nodes: Dict[int, Node] = {}  # 节点字典 {节点ID: 节点}
        self.beams: Dict[int, Beam] = {}  # 梁构件字典 {梁ID: 梁}
        self.sections: Dict[str, Section] = {}  # 截面属性字典 {截面名: 截面属性}
        self.node_counter = 0  # 节点ID计数器
        self.beam_counter = 0  # 梁ID计数器
        self._initialize_sections()  # 初始化截面属性

    def _initialize_sections(self):
        """初始化默认截面属性"""
        # 方钢管 □150×6
        self.sections["FRAME"] = Section(
            name="□150×6",
            area=3.384e-3,  # m^2
            Ix=1.164e-5,  # m^4
            Iy=1.164e-5,
            Iz=2.328e-5,
            Zx=1.55e-4,  # m^3
            Zy=1.55e-4,
            Zz=1.55e-4,
            density=7850.0,  # kg/m^3
        )

        # H型钢 HW150×150
        self.sections["MAIN"] = Section(
            name="HW150×150",
            area=3.982e-3,  # m^2
            Ix=1.383e-5,  # m^4
            Iy=4.763e-5,
            Iz=1.383e-5,
            Zx=1.842e-4,  # m^3
            Zy=6.35e-4,
            Zz=1.842e-4,
            density=7850.0,  # kg/m^3
        )

        # H型钢 HN100×50
        self.sections["SECONDARY"] = Section(
            name="HN100×50",
            area=1.184e-3,  # m^2
            Ix=1.32e-6,  # m^4
            Iy=1.97e-6,
            Iz=1.32e-6,
            Zx=2.63e-5,  # m^3
            Zy=3.94e-5,
            Zz=2.63e-5,
            density=7850.0,  # kg/m^3
        )

        # 槽钢 C10
        self.sections["BRACE"] = Section(
            name="C10",
            area=1.31e-3,  # m^2
            Ix=1.5e-6,  # m^4
            Iy=2.36e-5,
            Iz=1.5e-6,
            Zx=3.0e-5,  # m^3
            Zy=4.72e-4,
            Zz=3.0e-5,
            density=7850.0,  # kg/m^3
        )

    def add_node(self, x: float, y: float, z: float) -> int:
        """添加节点并返回节点ID"""
        # 首先检查是否已经存在相同坐标的节点（考虑一定的精度）
        tolerance = 1e-6
        for node in self.nodes.values():
            if abs(node.x - x) < tolerance and abs(node.y - y) < tolerance and abs(node.z - z) < tolerance:
                return node.id  # 返回已存在节点的ID

        # 如果不存在，创建新节点
        node_id = self.node_counter
        self.nodes[node_id] = Node(id=node_id, x=x, y=y, z=z)
        self.node_counter += 1
        return node_id

    def add_beam(self, start_node: int, end_node: int, beam_type: BeamType, section: str) -> int:
        """添加梁构件并返回梁ID"""
        # 检查节点是否存在
        if start_node not in self.nodes or end_node not in self.nodes:
            raise ValueError(f"Node {start_node} or {end_node} not found")

        # 检查是否已经存在相同的梁（同样的起止节点和类型）
        for beam in self.beams.values():
            if (
                (beam.start_node == start_node and beam.end_node == end_node)
                or (beam.start_node == end_node and beam.end_node == start_node)
            ) and beam.beam_type == beam_type:
                return beam.id  # 返回已存在梁的ID

        # 创建新梁
        beam_id = self.beam_counter
        self.beams[beam_id] = Beam(
            id=beam_id, start_node=start_node, end_node=end_node, beam_type=beam_type, section=section
        )
        self.beam_counter += 1
        return beam_id

    def generate_mesh(self):
        """生成结构网格"""
        # 清空现有节点和构件
        self.nodes = {}
        self.beams = {}
        self.node_counter = 0
        self.beam_counter = 0

        # 按顺序生成各类型梁
        self._generate_frame_beams()
        self._generate_main_beams()
        self._generate_secondary_beams()
        self._generate_brace_beams()

    def _generate_frame_beams(self):
        """生成框架梁"""
        # 获取舱体参数
        length = self.params.length
        width = self.params.width
        height = self.params.height

        # 生成框架的8个顶点
        corners = list(itertools.product([0, width], [0, length], [0, height]))
        corner_nodes = []

        for i, (x, y, z) in enumerate(corners):
            corner_nodes.append(self.add_node(x, y, z))

        # 连接框架梁（12条边）
        # 底部矩形（z=0）
        self.add_beam(corner_nodes[0], corner_nodes[1], BeamType.FRAME, "FRAME")  # 底部前边
        self.add_beam(corner_nodes[1], corner_nodes[3], BeamType.FRAME, "FRAME")  # 底部右边
        self.add_beam(corner_nodes[3], corner_nodes[2], BeamType.FRAME, "FRAME")  # 底部后边
        self.add_beam(corner_nodes[2], corner_nodes[0], BeamType.FRAME, "FRAME")  # 底部左边

        # 顶部矩形（z=height）
        self.add_beam(corner_nodes[4], corner_nodes[5], BeamType.FRAME, "FRAME")  # 顶部前边
        self.add_beam(corner_nodes[5], corner_nodes[7], BeamType.FRAME, "FRAME")  # 顶部右边
        self.add_beam(corner_nodes[7], corner_nodes[6], BeamType.FRAME, "FRAME")  # 顶部后边
        self.add_beam(corner_nodes[6], corner_nodes[4], BeamType.FRAME, "FRAME")  # 顶部左边

        # 垂直边（连接顶部和底部）
        self.add_beam(corner_nodes[0], corner_nodes[4], BeamType.FRAME, "FRAME")  # 左前柱
        self.add_beam(corner_nodes[1], corner_nodes[5], BeamType.FRAME, "FRAME")  # 右前柱
        self.add_beam(corner_nodes[2], corner_nodes[6], BeamType.FRAME, "FRAME")  # 左后柱
        self.add_beam(corner_nodes[3], corner_nodes[7], BeamType.FRAME, "FRAME")  # 右后柱

    def _generate_main_beams(self):
        """生成环向主梁"""
        # 获取舱体参数
        length = self.params.length
        width = self.params.width
        height = self.params.height
        spacing = self.params.main_beam_spacing

        # 计算沿y轴的主梁分布位置
        y_positions = np.arange(spacing, length, spacing)
        if abs(y_positions[-1] - length) < spacing * 0.5:  # 如果最后一个位置太接近尾部
            y_positions = y_positions[:-1]  # 移除最后一个位置

        # 为每个y坐标位置创建主梁（在xz平面上）
        for y in y_positions:
            # 创建主梁所需的4个节点（形成一个垂直于y轴的矩形）
            bottom_left = self.add_node(0, y, 0)
            bottom_right = self.add_node(width, y, 0)
            top_left = self.add_node(0, y, height)
            top_right = self.add_node(width, y, height)

            # 添加主梁构件
            self.add_beam(bottom_left, bottom_right, BeamType.MAIN, "MAIN")  # 底部横梁
            self.add_beam(top_left, top_right, BeamType.MAIN, "MAIN")  # 顶部横梁
            self.add_beam(bottom_left, top_left, BeamType.MAIN, "MAIN")  # 左侧立柱
            self.add_beam(bottom_right, top_right, BeamType.MAIN, "MAIN")  # 右侧立柱

    def _generate_secondary_beams(self):
        """生成环向次梁"""
        # 获取舱体参数
        length = self.params.length
        width = self.params.width
        height = self.params.height
        spacing = self.params.secondary_beam_spacing

        # 处理窗户位置
        window_heights = set()  # 存储窗户上下边沿的z坐标

        for window in self.params.windows:
            window_heights.add(window.bottom_height)
            window_heights.add(window.bottom_height + window.height)

        # 计算z方向的次梁分布位置
        z_positions = list(np.arange(spacing, height, spacing))

        # 如果有窗户，确保窗户边沿有次梁
        for z in window_heights:
            if 0 < z < height:  # 确保z在合理范围内
                # 查找最接近的现有z位置
                closest_idx = np.abs(np.array(z_positions) - z).argmin()
                closest_z = z_positions[closest_idx]

                # 如果不够接近，添加窗户边沿位置
                if abs(closest_z - z) > 0.1:  # 允许10cm的误差
                    z_positions.append(z)

        # 排序并去重
        z_positions = sorted(list(set(z_positions)))

        # 添加次梁（平行于y轴）
        for z in z_positions:
            # 创建前后面的次梁
            front_left = self.add_node(0, 0, z)
            front_right = self.add_node(width, 0, z)
            back_left = self.add_node(0, length, z)
            back_right = self.add_node(width, length, z)

            self.add_beam(front_left, front_right, BeamType.SECONDARY, "SECONDARY")  # 前面横梁
            self.add_beam(back_left, back_right, BeamType.SECONDARY, "SECONDARY")  # 后面横梁

            # 检查是否需要避开窗户位置
            skip_positions = []
            for window in self.params.windows:
                if (
                    abs(z - window.bottom_height) < 0.01
                    or abs(z - (window.bottom_height + window.height)) < 0.01
                ):
                    # 这是窗户边沿，需要计算跳过的区间
                    if window.on_face == "front" or window.on_face == "back":
                        y_min = window.y_position - window.width / 2
                        y_max = window.y_position + window.width / 2
                        skip_positions.append((y_min, y_max, window.on_face))
                    elif window.on_face == "right":
                        y_min = window.y_position - window.width / 2
                        y_max = window.y_position + window.width / 2
                        skip_positions.append((y_min, y_max, "right"))

            # 在左右面添加次梁（考虑窗户位置）
            y_positions = (
                [0]
                + list(np.arange(self.params.main_beam_spacing, length, self.params.main_beam_spacing))
                + [length]
            )

            for i in range(len(y_positions) - 1):
                y1, y2 = y_positions[i], y_positions[i + 1]
                y_mid = (y1 + y2) / 2

                # 检查是否在窗户位置
                skip_left = False
                skip_right = False

                for y_min, y_max, face in skip_positions:
                    if y_min <= y_mid <= y_max:
                        if face == "right":
                            skip_right = True
                        elif face == "left":
                            skip_left = True

                # 添加左侧次梁
                if not skip_left:
                    left_front = self.add_node(0, y1, z)
                    left_back = self.add_node(0, y2, z)
                    self.add_beam(left_front, left_back, BeamType.SECONDARY, "SECONDARY")

                # 添加右侧次梁
                if not skip_right:
                    right_front = self.add_node(width, y1, z)
                    right_back = self.add_node(width, y2, z)
                    self.add_beam(right_front, right_back, BeamType.SECONDARY, "SECONDARY")

    def _generate_brace_beams(self):
        """生成支撑梁"""
        # 获取舱体参数
        length = self.params.length
        width = self.params.width
        height = self.params.height

        # 计算支撑梁位置（在连续两排次梁之间添加）
        if self.params.secondary_beam_spacing > self.params.brace_spacing and self.params.brace_spacing > 0:
            # 计算每个次梁间距内应该放置的支撑梁数量
            n_braces = max(1, int(self.params.secondary_beam_spacing / self.params.brace_spacing) - 1)

            # 获取所有次梁的z坐标位置
            secondary_beams_z = sorted(
                list(
                    set(
                        [
                            self.nodes[beam.start_node].z
                            for beam in self.beams.values()
                            if beam.beam_type == BeamType.SECONDARY
                        ]
                    )
                )
            )

            # 在每对相邻次梁之间添加支撑梁
            for i in range(len(secondary_beams_z) - 1):
                z1, z2 = secondary_beams_z[i], secondary_beams_z[i + 1]

                # 平均分配支撑梁
                z_positions = [z1 + (z2 - z1) * (j + 1) / (n_braces + 1) for j in range(n_braces)]

                for z in z_positions:
                    # 添加前后面的支撑梁
                    front_left = self.add_node(0, 0, z)
                    front_right = self.add_node(width, 0, z)
                    back_left = self.add_node(0, length, z)
                    back_right = self.add_node(width, length, z)

                    self.add_beam(front_left, front_right, BeamType.BRACE, "BRACE")  # 前面支撑
                    self.add_beam(back_left, back_right, BeamType.BRACE, "BRACE")  # 后面支撑

                    # 添加左右面的支撑梁
                    # 同样按照主梁间距划分
                    y_positions = (
                        [0]
                        + list(
                            np.arange(self.params.main_beam_spacing, length, self.params.main_beam_spacing)
                        )
                        + [length]
                    )

                    for j in range(len(y_positions) - 1):
                        y1, y2 = y_positions[j], y_positions[j + 1]

                        # 添加左侧支撑梁
                        left_front = self.add_node(0, y1, z)
                        left_back = self.add_node(0, y2, z)
                        self.add_beam(left_front, left_back, BeamType.BRACE, "BRACE")

                        # 添加右侧支撑梁
                        right_front = self.add_node(width, y1, z)
                        right_back = self.add_node(width, y2, z)
                        self.add_beam(right_front, right_back, BeamType.BRACE, "BRACE")

    def get_structure_graph(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取结构图表示

        Returns:
            Tuple[np.ndarray, np.ndarray]: (节点特征矩阵, 邻接矩阵)
        """
        # 构建节点特征矩阵
        n_nodes = len(self.nodes)
        node_features = np.zeros((n_nodes, 10))  # 节点特征:x,y,z坐标,是否为各类梁的端点等

        # 节点的基本特征：坐标
        for node_id, node in self.nodes.items():
            node_features[node_id, 0] = node.x
            node_features[node_id, 1] = node.y
            node_features[node_id, 2] = node.z

        # 节点的结构特征：连接的梁类型
        for beam_id, beam in self.beams.items():
            beam_type_index = 3 + beam.beam_type.value  # 索引3-6对应四种梁类型
            node_features[beam.start_node, beam_type_index] = 1
            node_features[beam.end_node, beam_type_index] = 1

            # 标记负载和边界条件（这里简化处理，下部节点固定）
            if self.nodes[beam.start_node].z < 0.01:  # 接近地面的节点
                node_features[beam.start_node, 7] = 1  # 固定约束
            if self.nodes[beam.end_node].z < 0.01:  # 接近地面的节点
                node_features[beam.end_node, 7] = 1  # 固定约束

            # 上表面受内压力
            if self.nodes[beam.start_node].z > self.params.height - 0.01:
                node_features[beam.start_node, 8] = 1  # 受压
            if self.nodes[beam.end_node].z > self.params.height - 0.01:
                node_features[beam.end_node, 8] = 1  # 受压

        # 构建邻接矩阵
        adjacency_matrix = np.zeros((n_nodes, n_nodes))
        for beam_id, beam in self.beams.items():
            adjacency_matrix[beam.start_node, beam.end_node] = 1
            adjacency_matrix[beam.end_node, beam.start_node] = 1  # 无向图

        return node_features, adjacency_matrix

    def update_sections(self, section_params: Dict):
        """更新构件截面"""
        for section_name, params in section_params.items():
            if section_name in self.sections:
                for key, value in params.items():
                    setattr(self.sections[section_name], key, value)

    def calculate_mass(self) -> float:
        """计算结构总质量"""
        total_mass = 0.0

        for beam_id, beam in self.beams.items():
            # 获取梁的长度
            length = beam.length(self.nodes)

            # 获取截面属性
            section = self.sections.get(beam.section)
            if section:
                # 计算单个构件质量: 面积 * 长度 * 密度
                mass = section.area * length * section.density
                total_mass += mass

        return total_mass
