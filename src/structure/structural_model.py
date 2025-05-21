import itertools
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.structure.section_library import SectionLibrary, SteelSection
from src.utils.util import is_close


class BeamType(Enum):
    """梁构件类型枚举"""

    FRAME = 0  # 框架梁
    MAIN = 1  # 环向主梁
    SECONDARY = 2  # 环向次梁
    BRACE = 3  # 支撑梁


@dataclass
class Point:
    """节点类"""

    id: int
    x: float
    y: float
    z: float

    def coordinates(self) -> np.ndarray:
        """返回节点坐标"""
        return np.array([self.x, self.y, self.z])

    def distance_to(self, other: "Point") -> float:
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

    def length(self, nodes: Dict[int, Point]) -> float:
        """计算梁长度"""
        if self.start_node not in nodes or self.end_node not in nodes:
            raise ValueError(f"Node {self.start_node} or {self.end_node} not found")
        return nodes[self.start_node].distance_to(nodes[self.end_node])

    def direction(self, nodes: Dict[int, Point]) -> np.ndarray:
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
    on_face: str  # 窗户所在面 ('front', 'back', 'left', 'right')
    is_door: bool = False  # 是否为门


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
    frame_section_index: int = 0  # 方钢索引
    main_beam_section_index: int = 0  # H型钢索引 (用于主梁)
    secondary_beam_section_index: int = 0  # H型钢索引 (用于次梁)
    brace_section_index: int = 0  # 槽钢索引


class NodeMap:
    """节点坐标到节点ID的映射"""

    def __init__(self):
        self._node_map: Dict[Tuple[float, float, float], int] = {}

    def add_node(self, x: float, y: float, z: float, node_id: int):
        key = (round(x, 6), round(y, 6), round(z, 6))
        if key in self._node_map:
            return self._node_map[key]
        self._node_map[key] = node_id
        return node_id

    def get_node_id(self, x: float, y: float, z: float) -> int:
        key = (round(x, 6), round(y, 6), round(z, 6))
        return self._node_map.get(key, None)


class StructuralModel:
    """舱体结构模型类"""

    def __init__(self, params: StructuralParameters, section_library: SectionLibrary):
        self.params = params
        self.section_library = section_library
        self.points: Dict[int, Point] = {}  # 节点字典 {节点ID: 节点}
        self.node_map = NodeMap()  # 节点坐标到节点ID的映射
        self.beams: Dict[int, Beam] = {}  # 梁构件字典 {梁ID: 梁}
        self.sections: Dict[str, SteelSection] = {}  # 截面属性字典 {截面名: 截面属性}
        self.node_counter = 0  # 节点ID计数器
        self.beam_counter = 0  # 梁ID计数器
        self._initialize_sections()  # 初始化截面属性

    def _initialize_sections(self):
        """初始化默认截面属性"""
        # FRAME section
        frame_sec = self.section_library.get_section("方钢", self.params.frame_section_index)
        if frame_sec:
            self.sections["FRAME"] = frame_sec
        else:
            # Fallback or error handling if section not found
            print(
                f"Warning: Frame section (方钢, index {self.params.frame_section_index}) not found in library. Using default or placeholders."
            )
            # Add a default placeholder if necessary, or raise an error
            self.sections["FRAME"] = SteelSection(
                name="Default-Frame", area=1e-3, Ix=1e-5, Iy=1e-5, Iz=1e-5, Zx=1e-4, Zy=1e-4, Zz=1e-4
            )

        # MAIN beam section
        main_beam_sec = self.section_library.get_section("H型钢", self.params.main_beam_section_index)
        if main_beam_sec:
            self.sections["MAIN"] = main_beam_sec
        else:
            print(
                f"Warning: Main beam section (H型钢, index {self.params.main_beam_section_index}) not found. Using default."
            )
            self.sections["MAIN"] = SteelSection(
                name="Default-Main", area=1e-3, Ix=1e-5, Iy=1e-5, Iz=1e-5, Zx=1e-4, Zy=1e-4, Zz=1e-4
            )

        # SECONDARY beam section
        secondary_beam_sec = self.section_library.get_section(
            "H型钢", self.params.secondary_beam_section_index
        )
        if secondary_beam_sec:
            self.sections["SECONDARY"] = secondary_beam_sec
        else:
            print(
                f"Warning: Secondary beam section (H型钢, index {self.params.secondary_beam_section_index}) not found. Using default."
            )
            self.sections["SECONDARY"] = SteelSection(
                name="Default-Secondary", area=1e-3, Ix=1e-5, Iy=1e-5, Iz=1e-5, Zx=1e-4, Zy=1e-4, Zz=1e-4
            )

        # BRACE section
        brace_sec = self.section_library.get_section("C型钢", self.params.brace_section_index)
        if brace_sec:
            self.sections["BRACE"] = brace_sec
        else:
            print(
                f"Warning: Brace section (C型钢, index {self.params.brace_section_index}) not found. Using default."
            )
            self.sections["BRACE"] = SteelSection(
                name="Default-Brace", area=1e-3, Ix=1e-5, Iy=1e-5, Iz=1e-5, Zx=1e-4, Zy=1e-4, Zz=1e-4
            )

    def add_node(self, x: float, y: float, z: float) -> int:
        """添加节点并返回节点ID"""
        # 首先检查是否已经存在相同坐标的节点（考虑一定的精度）
        node_id = self.node_map.get_node_id(x, y, z)
        if node_id is not None:
            return node_id

        # 如果不存在，创建新节点
        node_id = self.node_counter
        self.points[node_id] = Point(id=node_id, x=x, y=y, z=z)
        self.node_map.add_node(x, y, z, node_id)
        self.node_counter += 1
        return node_id

    def add_beam(self, start_node: int, end_node: int, beam_type: BeamType, section: str) -> int:
        """添加梁构件并返回梁ID"""
        # 检查节点是否存在
        if start_node not in self.points or end_node not in self.points:
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

    def get_structure_graph(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取结构图表示

        Returns:
            Tuple[np.ndarray, np.ndarray]: (节点特征矩阵, 邻接矩阵)
        """
        # 构建节点特征矩阵
        n_nodes = len(self.points)
        node_features = np.zeros((n_nodes, 10))  # 节点特征:x,y,z坐标,是否为各类梁的端点等

        # 节点的基本特征：坐标
        for node_id, node in self.points.items():
            node_features[node_id, 0] = node.x
            node_features[node_id, 1] = node.y
            node_features[node_id, 2] = node.z

        # 节点的结构特征：连接的梁类型
        for beam_id, beam in self.beams.items():
            beam_type_index = 3 + beam.beam_type.value  # 索引3-6对应四种梁类型
            node_features[beam.start_node, beam_type_index] = 1
            node_features[beam.end_node, beam_type_index] = 1

            # 标记负载和边界条件（这里简化处理，下部节点固定）
            if self.points[beam.start_node].z < 0.01:  # 接近地面的节点
                node_features[beam.start_node, 7] = 1  # 固定约束
            if self.points[beam.end_node].z < 0.01:  # 接近地面的节点
                node_features[beam.end_node, 7] = 1  # 固定约束

            # 上表面受内压力
            if self.points[beam.start_node].z > self.params.height - 0.01:
                node_features[beam.start_node, 8] = 1  # 受压
            if self.points[beam.end_node].z > self.params.height - 0.01:
                node_features[beam.end_node, 8] = 1  # 受压

        # 构建邻接矩阵
        adjacency_matrix = np.zeros((n_nodes, n_nodes))
        for beam_id, beam in self.beams.items():
            adjacency_matrix[beam.start_node, beam.end_node] = 1
            adjacency_matrix[beam.end_node, beam.start_node] = 1  # 无向图

        return node_features, adjacency_matrix

    def calculate_mass(self) -> float:
        """计算结构总质量"""
        total_mass = 0.0

        for beam_id, beam in self.beams.items():
            # 获取梁的长度
            length = beam.length(self.points)

            # 获取截面属性
            section = self.sections.get(beam.section)
            if section:
                # 计算单个构件质量: 面积 * 长度 * 密度
                mass = section.area * length * section.density
                total_mass += mass

        return total_mass

    def generate_grid_mesh(self):
        """使用网格法生成结构，将长直梁拆分成多段小梁段"""
        # 清空现有节点和构件
        self.points = {}
        self.beams = {}
        self.node_counter = 0
        self.beam_counter = 0

        # 生成水平面的网格
        self._generate_horizontal_grids()

        # 生成前后立面的网格
        self._generate_vertical_grids()

        # 生成左右侧面的网格
        self._generate_side_grids()

    def _generate_horizontal_grids(self):
        """生成上下平面的网格布置"""
        length = self.params.length
        width = self.params.width
        height = self.params.height
        main_spacing = self.params.main_beam_spacing
        brace_spacing = self.params.brace_spacing

        # 计算所有垂直坐标线（x轴方向）
        all_x = set([0, width])  # 框架边界

        # 计算次梁的x坐标（从中间向两边对称扩展）
        x_mid = width / 2
        secondary_x = {x_mid}  # 中间的次梁

        # 向左右两侧扩展次梁
        x = x_mid + main_spacing
        while x < width:
            secondary_x.add(x)
            x += main_spacing

        x = x_mid - main_spacing
        while x > 0:
            secondary_x.add(x)
            x -= main_spacing

        # 计算支撑梁的x坐标（在次梁之间添加支撑梁）
        support_x = set()
        sorted_secondary = sorted(secondary_x)
        for i in range(len(sorted_secondary) - 1):
            x_start = sorted_secondary[i]
            x_end = sorted_secondary[i + 1]

            n = math.ceil((x_end - x_start) / brace_spacing)
            _new_brace_spacing = (x_end - x_start) / n

            x_current = x_start + _new_brace_spacing
            while x_current < x_end:
                support_x.add(x_current)
                x_current += _new_brace_spacing

        # 计算所有水平坐标线（y轴方向）
        all_y = set([0, length])  # 框架边界

        # 计算主梁的y坐标
        primary_y = set()
        y = main_spacing
        while y < length:
            primary_y.add(y)
            y += main_spacing

        # 合并所有坐标线，形成完整的网格
        all_x = all_x.union(secondary_x).union(support_x)
        all_y = all_y.union(primary_y)

        # 排序坐标线，便于后续处理
        all_x = sorted(all_x)
        all_y = sorted(all_y)

        # 生成底部面的网格
        self._generate_plane_grid(all_x, all_y, 0, secondary_x, support_x, primary_y)

        # 生成顶部面的网格
        self._generate_plane_grid(all_x, all_y, height, secondary_x, support_x, primary_y)

    def _generate_vertical_grids(self):
        """生成前后面的网格布置"""
        width = self.params.width

        # 获取所有与前后面相关的窗户
        front_windows = [window for window in self.params.windows if window.on_face == "front"]
        back_windows = [window for window in self.params.windows if window.on_face == "back"]

        # 生成前面的网格
        self._generate_vertical_face_grid(0, front_windows)

        # 生成后面的网格
        self._generate_vertical_face_grid(width, back_windows)

    def _generate_side_grids(self):
        """生成左右侧面的网格布置"""
        length = self.params.length

        # 获取所有与左右侧面相关的窗户和门
        left_windows = [window for window in self.params.windows if window.on_face == "left"]
        right_windows = [window for window in self.params.windows if window.on_face == "right"]

        windows = [window for window in left_windows + right_windows if not window.is_door]
        if windows:
            win_bottom = windows[0].bottom_height
            win_top = win_bottom + windows[0].height
        else:
            win_bottom = -1
            win_top = -1

        # 生成左侧面的网格
        self._generate_side_face_grid(0, left_windows, win_bottom, win_top)

        # 生成右侧面的网格
        self._generate_side_face_grid(length, right_windows, win_bottom, win_top, has_door=True)

    def _generate_vertical_face_grid(self, x_coord: float, windows: List[Window]):
        """在指定x坐标位置生成前/后立面的网格

        Args:
            x_coord: 面所在的x坐标
            windows: 该面上的窗户列表
        """
        length = self.params.length
        height = self.params.height
        main_spacing = self.params.main_beam_spacing
        brace_spacing = self.params.brace_spacing

        # 计算所有z坐标点
        # 首先添加底部和顶部框架点
        all_z = {0, height}

        # 为每个窗户添加窗户底部和顶部高度作为次梁位置
        window = windows[0]  # 每个窗户高度和离地高度相同
        window_bottom = window.bottom_height
        window_top = window.bottom_height + window.height
        all_z.add(window_bottom)
        all_z.add(window_top)

        # 添加支撑梁的z坐标
        # 窗户以下的支撑梁
        z = window_bottom - brace_spacing
        while z > 0:
            all_z.add(z)
            z -= brace_spacing

        # 窗户以上的支撑梁
        z = window_top + brace_spacing
        while z < height - 1e-4:
            all_z.add(z)
            z += brace_spacing

        # 窗户之间的支撑梁
        n = math.ceil(window.height / brace_spacing)
        _new_brace_spacing = window.height / n
        for i in range(n - 1):
            z = window_bottom + (i + 1) * _new_brace_spacing
            all_z.add(z)

        # 计算所有y坐标点
        all_y = {0, length}  # 框架边界

        # 计算主梁的y坐标
        primary_y = set()
        y = main_spacing
        while y < length:
            primary_y.add(y)
            y += main_spacing

        all_y = all_y.union(primary_y)

        # 排序所有坐标点
        all_y = sorted(all_y)
        all_z = sorted(all_z)

        # 生成节点
        for y in all_y:
            for z in all_z:
                self.add_node(x_coord, y, z)

        # 生成水平方向的梁段（沿y方向）
        for z in all_z:
            for i in range(len(all_y) - 1):
                y1, y2 = all_y[i], all_y[i + 1]
                start_node = self.node_map.get_node_id(x_coord, y1, z)
                end_node = self.node_map.get_node_id(x_coord, y2, z)

                # 确定梁类型
                if z == 0 or z == height:  # 底部或顶部的梁
                    beam_type = BeamType.FRAME
                    section = "FRAME"
                else:
                    # 检查是否在窗户区域内
                    in_window = False
                    for window in windows:
                        window_bottom = window.bottom_height
                        window_top = window.bottom_height + window.height
                        window_left = window.y_position - window.width / 2
                        window_right = window.y_position + window.width / 2

                        if (window_bottom < z < window_top) and y1 < window.y_position < y2:
                            # 这个区域在窗户范围内，不添加梁
                            in_window = True
                            break

                    if not in_window:
                        # 判断是否为次梁或支撑梁
                        is_secondary = False
                        for window in windows:
                            window_bottom = window.bottom_height
                            window_top = window.bottom_height + window.height
                            if z == window_bottom or z == window_top:
                                is_secondary = True
                                break

                        if is_secondary:
                            beam_type = BeamType.SECONDARY
                            section = "SECONDARY"
                        else:
                            beam_type = BeamType.BRACE
                            section = "BRACE"

                        self.add_beam(start_node, end_node, beam_type, section)

        # 生成竖直方向的梁段（沿z方向）
        for y in all_y:
            for i in range(len(all_z) - 1):
                z1, z2 = all_z[i], all_z[i + 1]
                start_node = self.node_map.get_node_id(x_coord, y, z1)
                end_node = self.node_map.get_node_id(x_coord, y, z2)

                # 确定梁类型
                if y == 0 or y == length:  # 左右边界上的梁
                    beam_type = BeamType.FRAME
                    section = "FRAME"
                    self.add_beam(start_node, end_node, beam_type, section)
                else:
                    # 检查是否在窗户区域内
                    in_window = False
                    for window in windows:
                        window_left = window.y_position - window.width / 2
                        window_right = window.y_position + window.width / 2

                        if is_close(y, window_left) or is_close(y, window_right):
                            # 这个区域在窗户范围内，不添加梁
                            in_window = True
                            beam_type = BeamType.FRAME
                            section = "FRAME"
                            break

                    if not in_window:
                        # 判断是否为主梁
                        if y in primary_y:
                            beam_type = BeamType.MAIN
                            section = "MAIN"
                        else:
                            raise ValueError(f"Invalid y coordinate: {y}")

                    self.add_beam(start_node, end_node, beam_type, section)

    def _generate_side_face_grid(
        self, y_coord: float, windows: List[Window], win_bottom, win_top, has_door: bool = False
    ):
        """在指定y坐标位置生成左/右侧面的网格

        Args:
            y_coord: 面所在的y坐标
            windows: 该面上的窗户或门列表
        """
        width = self.params.width
        height = self.params.height
        main_spacing = self.params.main_beam_spacing
        brace_spacing = self.params.brace_spacing

        # 计算所有z坐标点
        # 首先添加底部和顶部框架点
        all_z = {0, height}

        # 检查是否有窗户或门
        if windows:
            for window in windows:
                # 为每个窗户或门添加底部和顶部高度作为次梁位置
                window_bottom = window.bottom_height
                window_top = window.bottom_height + window.height
                all_z.add(window_bottom)
                all_z.add(window_top)

                # 添加支撑梁的z坐标
                # 窗户/门以下的支撑梁
                z = window_bottom - brace_spacing
                while z > 0:
                    all_z.add(z)
                    z -= brace_spacing

                # 窗户/门以上的支撑梁
                z = window_top + brace_spacing
                while z < height - 1e-4:
                    all_z.add(z)
                    z += brace_spacing

                # 窗户/门之间的支撑梁
                n = math.ceil(window.height / brace_spacing)
                _new_brace_spacing = window.height / n
                for i in range(n - 1):
                    z = window_bottom + (i + 1) * _new_brace_spacing
                    all_z.add(z)
        else:
            # 如果没有窗户或门，按照支撑梁间距布置支撑梁
            z = brace_spacing
            while z < height - 1e-4:
                all_z.add(z)
                z += brace_spacing

        # 计算所有x坐标点
        all_x = {0, width}  # 框架边界

        # 计算主梁的x坐标（先添加跨中主梁，然后向两侧扩展）
        primary_x = set()
        x_mid = width / 2
        primary_x.add(x_mid)  # 跨中主梁

        # 向左右两侧扩展主梁
        x = x_mid + main_spacing
        while x < width:
            primary_x.add(x)
            x += main_spacing

        x = x_mid - main_spacing
        while x > 0:
            primary_x.add(x)
            x -= main_spacing

        all_x = all_x.union(primary_x)

        # 排序所有坐标点
        all_x = sorted(all_x)
        all_z = sorted(all_z)

        # 生成节点
        for x in all_x:
            for z in all_z:
                self.add_node(x, y_coord, z)

        # 生成水平方向的梁段（沿x方向）
        for z in all_z:
            for i in range(len(all_x) - 1):
                x1, x2 = all_x[i], all_x[i + 1]
                start_node = self.node_map.get_node_id(x1, y_coord, z)
                end_node = self.node_map.get_node_id(x2, y_coord, z)

                # 确定梁类型
                if z == 0 or z == height:  # 底部或顶部的梁
                    beam_type = BeamType.FRAME
                    section = "FRAME"
                else:
                    # 检查是否在窗户/门区域内
                    in_window = False
                    for window in windows:
                        _bt = 0 if has_door else win_bottom
                        # 对于门或窗，检查是否在其区域内（对于侧面，窗户中心是x坐标）
                        if (_bt < z < win_top) and x1 < window.y_position < x2:
                            # 这个区域在窗户/门范围内，不添加梁
                            in_window = True
                            break

                    if not in_window:
                        # 判断是否为次梁或支撑梁
                        is_secondary = False
                        if z == win_bottom or z == win_top:
                            is_secondary = True

                        if is_secondary:
                            beam_type = BeamType.SECONDARY
                            section = "SECONDARY"
                        else:
                            beam_type = BeamType.BRACE
                            section = "BRACE"

                        self.add_beam(start_node, end_node, beam_type, section)

        # 生成竖直方向的梁段（沿z方向）
        for x in all_x:
            for i in range(len(all_z) - 1):
                z1, z2 = all_z[i], all_z[i + 1]
                start_node = self.node_map.get_node_id(x, y_coord, z1)
                end_node = self.node_map.get_node_id(x, y_coord, z2)

                # 确定梁类型
                if x == 0 or x == width:  # 左右边界上的梁
                    beam_type = BeamType.FRAME
                    section = "FRAME"
                    self.add_beam(start_node, end_node, beam_type, section)
                else:
                    # 检查是否在窗户/门区域内
                    in_window = False
                    for window in windows:
                        # 在左右侧面，窗户/门的宽度方向是x轴
                        window_left = window.y_position - window.width / 2
                        window_right = window.y_position + window.width / 2

                        if is_close(x, window_left) or is_close(x, window_right):
                            # 这是窗户/门的边缘，添加框架梁
                            in_window = True
                            beam_type = BeamType.FRAME
                            section = "FRAME"
                            break

                    if not in_window:
                        # 判断是否为主梁
                        if x in primary_x:
                            beam_type = BeamType.MAIN
                            section = "MAIN"
                        else:
                            raise ValueError(f"Invalid x coordinate: {x}")

                    self.add_beam(start_node, end_node, beam_type, section)

    def _generate_plane_grid(self, all_x, all_y, z, secondary_x, support_x, primary_y):
        """在指定z平面生成网格

        Args:
            all_x: 所有x坐标点
            all_y: 所有y坐标点
            z: 平面的z坐标
            secondary_x: 次梁的x坐标集合
            support_x: 支撑梁的x坐标集合
            primary_y: 主梁的y坐标集合
        """

        for x in all_x:
            for y in all_y:
                node_id = self.add_node(x, y, z)
                # print(f"add node: {node_id}, x: {x}, y: {y}, z: {z}")

        # 生成水平方向的梁段（y方向，框架梁、次梁、支撑梁）
        for x in all_x:
            for i in range(len(all_y) - 1):
                y1, y2 = all_y[i], all_y[i + 1]
                start_node = self.node_map.get_node_id(x, y1, z)
                end_node = self.node_map.get_node_id(x, y2, z)

                # 确定梁的类型
                if x == 0 or x == all_x[-1]:  # 边界上的梁
                    beam_type = BeamType.FRAME
                    section = "FRAME"
                elif x in secondary_x:  # 次梁
                    beam_type = BeamType.SECONDARY
                    section = "SECONDARY"
                elif x in support_x:  # 支撑梁
                    beam_type = BeamType.BRACE
                    section = "BRACE"
                else:
                    raise ValueError(f"Invalid x coordinate: {x}")

                self.add_beam(start_node, end_node, beam_type, section)

        # 生成垂直方向的梁段（x方向，框架梁、主梁）
        for y in all_y:
            for i in range(len(all_x) - 1):
                x1, x2 = all_x[i], all_x[i + 1]
                start_node = self.node_map.get_node_id(x1, y, z)
                end_node = self.node_map.get_node_id(x2, y, z)

                # 确定梁的类型
                if y == 0 or y == all_y[-1]:  # 边界上的梁
                    beam_type = BeamType.FRAME
                    section = "FRAME"
                elif y in primary_y:  # 主梁
                    beam_type = BeamType.MAIN
                    section = "MAIN"
                else:  # 这种情况不应该发生，为了安全起见，当作次梁处理
                    raise ValueError(f"Invalid y coordinate: {y}")

                self.add_beam(start_node, end_node, beam_type, section)
