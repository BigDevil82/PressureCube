import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from src.structure.enums import BeamType
from src.structure.section_library import SectionLibrary, SectionProperties, SectionType


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

    main_beam_spacing: float  # 主梁间距
    num_secondary_width: int  # 宽度方向(X)次梁分段数
    num_secondary_height: int  # 高度方向(Z)次梁分段数
    add_support_beam: bool = True  # 是否添加支撑梁

    frame_section_index: int = 0  # 方钢索引
    main_beam_section_index: int = 0  # H型钢索引 (用于主梁)
    secondary_beam_section_index: int = 0  # H型钢索引 (用于次梁)
    brace_section_index: int = 0  # 槽钢索引

    windows: List[Window] = field(default_factory=list)  # 窗户参数列表
    win_bottom: float = 1.0  # 窗户底部离地高度
    win_height: float = 1.5  # 窗户高度
    plate_thickness: float = 0.01  # 钢板厚度
    pressure: float = 30  # 内压 (kPa)


@dataclass
class Point:
    id: int
    x: float
    y: float
    z: float

    def coordinates(self) -> np.ndarray:
        """返回节点坐标"""
        return np.array([self.x, self.y, self.z])

    def distance_to(self, other) -> float:
        """计算到另一个节点的距离"""
        if not isinstance(other, Point):
            raise TypeError("Parameter to calculate distance must be Point instance")
        return np.linalg.norm(self.coordinates() - other.coordinates())


@dataclass
class BeamResponse:
    """存储单根梁的聚合分析结果"""

    max_stress_ratio: float = 0.0  # 应力比 (Actual / Limit)
    max_disp_ratio: float = 0.0  # 挠度比 (Actual / Limit)
    is_violated: bool = False  # 是否超限


@dataclass
class Beam:
    id: int
    start_node: int
    end_node: int
    beam_type: BeamType
    section: SectionProperties
    is_virtual: bool  # 为了后续网格划分方便，参数化建模时把不存在的梁也生成，但是FEA建模时不生成
    on_face: str  # 记录属于哪个面，方便调试
    parent_id: int = None  # 用于记录该梁是由哪个原始梁细分而来的
    response: BeamResponse = None  # 存储分析结果

    def direction(self, nodes: Dict[int, Point]) -> np.ndarray:
        """计算梁的方向向量"""
        start = nodes[self.start_node].coordinates()
        end = nodes[self.end_node].coordinates()
        direction = end - start
        return direction / np.linalg.norm(direction)


@dataclass
class Plate:
    id: int
    nodes: List[int]  # [n1, n2, n3, n4]
    on_face: str
    area: float


class NodeManager:
    """管理节点，处理浮点精度去重"""

    def __init__(self):
        self._node_map: Dict[Tuple[int, int, int], int] = {}
        self.points: Dict[int, Point] = {}
        self.counter = 0
        self.precision = 1000  # 精度保留3位小数

    def _get_key(self, x, y, z):
        return (
            int(round(x * self.precision)),
            int(round(y * self.precision)),
            int(round(z * self.precision)),
        )

    def get_or_create(self, x: float, y: float, z: float) -> int:
        key = self._get_key(x, y, z)
        if key in self._node_map:
            return self._node_map[key]

        node_id = self.counter
        self.points[node_id] = Point(node_id, x, y, z)
        self._node_map[key] = node_id
        self.counter += 1
        return node_id


class BeamManager:
    """管理梁，处理重复梁并保留高优先级类型"""

    def __init__(self, node_manager: NodeManager):
        self.node_manager = node_manager
        self.beams: Dict[int, Beam] = {}
        self._beam_map: Dict[Tuple[int, int], int] = {}  # (min_id, max_id) -> beam_id
        self.counter = 0

        # 定义梁优先级：FRAME > MAIN > SECONDARY > BRACE
        # 数值越小优先级越高 (基于Enum定义)
        self.priority = {BeamType.FRAME: 0, BeamType.MAIN: 1, BeamType.SECONDARY: 2, BeamType.BRACE: 3}

    def add_or_update_beam(
        self,
        n1: int,
        n2: int,
        b_type: BeamType,
        section_prop: SectionProperties,
        is_virtual: bool,
        face: str,
        parent_id=None,
    ):
        if n1 == n2:
            return  # 忽略零长梁

        key = tuple(sorted((n1, n2)))

        if key in self._beam_map:
            # 梁已存在，检查是否需要升级类型
            existing_id = self._beam_map[key]
            existing_beam = self.beams[existing_id]

            if self.priority[b_type] <= self.priority[existing_beam.beam_type]:
                # 新梁优先级更高，更新类型
                existing_beam.beam_type = b_type
                # existing_beam.on_face += f"/{face}"  # 标记该梁是多面共用
            return existing_id

        # 创建新梁
        beam_id = self.counter
        self.beams[beam_id] = Beam(beam_id, n1, n2, b_type, section_prop, is_virtual, face, parent_id)
        self._beam_map[key] = beam_id
        self.counter += 1
        return beam_id


@dataclass
class GridLine:
    pos: float  # 在轴上的位置
    # 如果有梁沿着此线放置，其类型是什么（用于V向线定义U向梁的端点，反之亦然，需仔细辨析）
    line_type: BeamType


# 映射关系： 参数化模型的梁类型 -> 截面库的类型
BEAM_TYPE_MAP = {
    BeamType.FRAME: SectionType.BOX,
    BeamType.MAIN: SectionType.H_MAIN,
    BeamType.SECONDARY: SectionType.H_SEC,
    BeamType.BRACE: SectionType.CHANNEL,
}


class StructuralModel:
    """
    参数化结构模型类，用于生成节点、梁和板。
    通过 StructuralParameters 定义参数，通过 SectionLibrary 获取截面属性。
    长度单位：米 (m)。
    """

    def __init__(self, params: StructuralParameters, section_lib: SectionLibrary):
        self.params = params
        self.section_lib = section_lib
        self.node_manager = NodeManager()
        self.beam_manager = BeamManager(self.node_manager)

        self.plate_counter = 10000  # 板ID从10000开始，避免opensees建模时与梁ID冲突
        self.plates: Dict[int, Plate] = {}
        self.points = self.node_manager.points
        self.beams = self.beam_manager.beams

    def generate(self):
        L, W, H = self.params.length, self.params.width, self.params.height

        # 1. 定义6个面的生成逻辑
        # 每个面定义: 原点, U轴向量, V轴向量, U方向长度, V方向长度, 面名称, 策略

        self._build_face(
            origin=(0, 0, H),
            u_vec=(1, 0, 0),
            v_vec=(0, 1, 0),
            u_len=W,
            v_len=L,
            name="top",
            u_beam_logic="MAIN",
            v_beam_logic="SECONDARY",
        )

        # Bottom (Z=0): Same as Top.
        self._build_face(
            origin=(0, 0, 0),
            u_vec=(1, 0, 0),
            v_vec=(0, 1, 0),
            u_len=W,
            v_len=L,
            name="bottom",
            u_beam_logic="MAIN",
            v_beam_logic="SECONDARY",
        )

        # Front (X=W): U=Y(L), V=Z(H). (长侧面)
        self._build_face(
            origin=(W, 0, 0),
            u_vec=(0, 1, 0),
            v_vec=(0, 0, 1),
            u_len=L,
            v_len=H,
            name="front",
            u_beam_logic="SECONDARY",
            v_beam_logic="MAIN",
        )

        # Back (X=0): Same as Front.
        self._build_face(
            origin=(0, 0, 0),
            u_vec=(0, 1, 0),
            v_vec=(0, 0, 1),
            u_len=L,
            v_len=H,
            name="back",
            u_beam_logic="SECONDARY",
            v_beam_logic="MAIN",
        )

        # Left (Y=0): U=X(W), V=Z(H). (端盖)
        self._build_face(
            origin=(0, 0, 0),
            u_vec=(1, 0, 0),
            v_vec=(0, 0, 1),
            u_len=W,
            v_len=H,
            name="left",
            u_beam_logic="SECONDARY",
            v_beam_logic="FRAME_GRID",
            # virtual_beam=BeamType.BRACE,
        )

        # Right (Y=L): Same as Left.
        self._build_face(
            origin=(0, L, 0),
            u_vec=(1, 0, 0),
            v_vec=(0, 0, 1),
            u_len=W,
            v_len=H,
            name="right",
            u_beam_logic="SECONDARY",
            v_beam_logic="FRAME_GRID",
            # virtual_beam=BeamType.BRACE,
        )
        self._generate_extension()

        ext_x_start = self.params.width * 0.5
        ext_x_end = self.params.width

        # 调用删除: 在 'left' 面上，删除 X 在 [start, end], Z 在 [0, H] 范围内的元素
        self._remove_elements_in_face_area(
            face_name="left",
            u_range=(ext_x_start, ext_x_end),  # 对应 x
            v_range=(0, self.params.height),  # 对应 z
            axis_val=0,  # Y=0 面
        )

    def _generate_extension(self):
        """
        生成左侧凸出区域 (Extension Block)
        逻辑：在 Y < 0 的区域生成一个盒子，通过 Y=0 面与主结构连接。
        """
        # 获取参数
        L_ext = 1.1
        W_ext = self.params.width * 0.5
        H_ext = self.params.height  # 假设高度相同
        Off_x = self.params.width * 0.5

        # 1. 获取主结构在宽度方向(X轴)的完整网格线
        # 主结构 Top/Bottom 面沿 U(X) 轴分布的是 "SECONDARY" 类型的梁逻辑 (即V方向梁)
        full_width_lines = self._calculate_grid_lines(self.params.width, "SECONDARY")

        # 2. 截取落在凸出部分范围内的网格线
        # 凸出部分的 X 范围是 [Off_x, Off_x + W_ext]
        aligned_u_lines = []

        # 添加左边界 (强制 Frame)
        # 即使主网格在这个位置没有梁，凸出部分作为实体也需要边界梁
        # 使用 check_and_add 逻辑防止重复
        def add_line(lines_list, pos, b_type):
            # 检查是否已有近似的点
            for p, _ in lines_list:
                if math.isclose(p, pos, abs_tol=1e-4):
                    return
            lines_list.append((pos, b_type))

        # 先把主结构中落在范围内的线加进来
        for pos, b_type in full_width_lines:
            if pos >= Off_x - 1e-4 and pos <= (Off_x + W_ext) + 1e-4:
                # 转换为局部坐标 (相对于 Off_x)
                local_pos = pos - Off_x
                # 修正边界类型：如果恰好是凸出部分的边缘，强制为 FRAME
                if math.isclose(local_pos, 0) or math.isclose(local_pos, W_ext):
                    real_type = BeamType.FRAME
                else:
                    real_type = b_type
                add_line(aligned_u_lines, local_pos, real_type)

        # 确保 0 和 W_ext 存在 (防止凸出部分并没有完全对齐主结构的网格线)
        add_line(aligned_u_lines, 0.0, BeamType.FRAME)
        add_line(aligned_u_lines, W_ext, BeamType.FRAME)

        # 排序
        aligned_u_lines.sort(key=lambda x: x[0])

        # 1. Extension Top (Z = H)
        self._build_face(
            origin=(Off_x, -L_ext, H_ext),
            u_vec=(1, 0, 0),  # U 沿 X
            v_vec=(0, 1, 0),  # V 沿 Y
            u_len=W_ext,
            v_len=L_ext,
            name="top",
            u_beam_logic="none",  # 保持与主结构一致
            v_beam_logic="SECONDARY",
            override_u_grid=aligned_u_lines,
        )

        # 2. Extension Bottom (Z = 0)
        self._build_face(
            origin=(Off_x, -L_ext, 0),
            u_vec=(1, 0, 0),
            v_vec=(0, 1, 0),
            u_len=W_ext,
            v_len=L_ext,
            name="bottom",
            u_beam_logic="none",
            v_beam_logic="SECONDARY",
            override_u_grid=aligned_u_lines,
        )

        # 3. Extension Front (X = Off_x + W_ext) - 对应主结构的 Front 侧
        self._build_face(
            origin=(Off_x + W_ext, -L_ext, 0),
            u_vec=(0, 1, 0),  # U 沿 Y
            v_vec=(0, 0, 1),  # V 沿 Z
            u_len=L_ext,
            v_len=H_ext,
            name="front",
            u_beam_logic="SECONDARY",
            v_beam_logic="none",  # 立柱
        )

        # 4. Extension Back (X = Off_x) - 对应主结构的 Back 侧
        self._build_face(
            origin=(Off_x, -L_ext, 0),
            u_vec=(0, 1, 0),
            v_vec=(0, 0, 1),
            u_len=L_ext,
            v_len=H_ext,
            name="back",
            u_beam_logic="SECONDARY",
            v_beam_logic="none",
        )

        # 5. Extension End Cap (Y = -L_ext) - 凸出部分的最外端盖
        left_v_lines_pos = [aligned_u_lines[0][0], aligned_u_lines[1][0], aligned_u_lines[-1][0]]
        left_v_lines = []
        for line in left_v_lines_pos:
            left_v_lines.append((line, BeamType.FRAME))

        self._build_face(
            origin=(Off_x, -L_ext, 0),
            u_vec=(1, 0, 0),  # U 沿 X
            v_vec=(0, 0, 1),  # V 沿 Z
            u_len=W_ext,
            v_len=H_ext,
            name="left",
            u_beam_logic="SECONDARY",
            v_beam_logic="FRAME",  # 类似主结构的端盖
            override_u_grid=left_v_lines,  # 仅使用边界线
        )

    def _build_face(
        self,
        origin,
        u_vec,
        v_vec,
        u_len,
        v_len,
        name,
        u_beam_logic,
        v_beam_logic,
        virtual_beam=None,
        override_u_grid=None,  # 新增参数：强制指定U轴网格线
        override_v_grid=None,  # 新增参数：强制指定V轴网格线
    ):
        """
        构建一个面的网格、梁和板。

        Args:
            u_beam_logic: 沿着U方向走的梁属于什么类型? (MAIN / SECONDARY / FRAME_GRID)
            v_beam_logic: 沿着V方向走的梁属于什么类型?
            virtual_beam: 虚拟梁（应对左右端盖不需要布置支撑梁但是划分网格时需要与上下面协同）
        """

        # 1. 生成网格线坐标和该线对应的“正交梁类型”
        # 注意：这里有点绕。
        # 如果 u_beam_logic 是 MAIN，意味着沿着 U 轴画的线是主梁。
        # 这些线是分布在 V 轴上的。所以我们需要计算 V 轴上的分割点，并标记这些点引出的 U 向线是 MAIN。

        # V轴上的点 (决定 U方向梁的位置)
        if override_v_grid is not None:
            v_lines = override_v_grid
        else:
            v_lines = self._calculate_grid_lines(v_len, u_beam_logic)

        # U轴上的点 (决定 V方向梁的位置)
        if override_u_grid is not None:
            u_lines = override_u_grid
        else:
            u_lines = self._calculate_grid_lines(u_len, v_beam_logic)

        # 2. 网格生成 (双重循环)
        # i 对应 U 轴索引, j 对应 V 轴索引
        for i in range(len(u_lines)):
            for j in range(len(v_lines)):
                # 当前节点
                u_curr, type_v_beam_here = u_lines[i]  # U坐标, 以及在此处沿着V方向走的梁的类型
                v_curr, type_u_beam_here = v_lines[j]  # V坐标, 以及在此处沿着U方向走的梁的类型

                # 计算绝对坐标
                pos = np.array(origin) + u_curr * np.array(u_vec) + v_curr * np.array(v_vec)
                node_id = self.node_manager.get_or_create(*pos)

                # 生成向右的梁 (U-Direction Beam) -> 连接 (i, j) 和 (i+1, j)
                # 这根梁的类型由它所在的 V 坐标 (type_u_beam_here) 决定
                if i < len(u_lines) - 1:
                    u_next = u_lines[i + 1][0]
                    pos_next = np.array(origin) + u_next * np.array(u_vec) + v_curr * np.array(v_vec)
                    next_id = self.node_manager.get_or_create(*pos_next)

                    # 确定梁类型：如果是在边缘，强制为 FRAME，否则由 grid line 属性决定
                    b_type = type_u_beam_here
                    if math.isclose(v_curr, 0) or math.isclose(v_curr, v_len):
                        b_type = BeamType.FRAME

                    is_virtual = b_type == virtual_beam
                    sec_prop = self._get_section_for_beam(b_type)
                    self.beam_manager.add_or_update_beam(node_id, next_id, b_type, sec_prop, is_virtual, name)

                # 生成向上的梁 (V-Direction Beam) -> 连接 (i, j) 和 (i, j+1)
                # 这根梁的类型由它所在的 U 坐标 (type_v_beam_here) 决定
                if j < len(v_lines) - 1:
                    v_next = v_lines[j + 1][0]
                    pos_next = np.array(origin) + u_curr * np.array(u_vec) + v_next * np.array(v_vec)
                    next_id = self.node_manager.get_or_create(*pos_next)

                    b_type = type_v_beam_here
                    if math.isclose(u_curr, 0) or math.isclose(u_curr, u_len):
                        b_type = BeamType.FRAME

                    is_virtual = b_type == virtual_beam
                    sec_prop = self._get_section_for_beam(b_type)
                    self.beam_manager.add_or_update_beam(node_id, next_id, b_type, sec_prop, is_virtual, name)

                # 生成板 (Plate) -> (i,j), (i+1,j), (i+1,j+1), (i,j+1)
                if i < len(u_lines) - 1 and j < len(v_lines) - 1:
                    # 获取四个角点
                    n1 = node_id

                    # (i+1, j)
                    u_next = u_lines[i + 1][0]
                    p2 = np.array(origin) + u_next * np.array(u_vec) + v_curr * np.array(v_vec)
                    n2 = self.node_manager.get_or_create(*p2)

                    # (i+1, j+1)
                    v_next = v_lines[j + 1][0]
                    p3 = np.array(origin) + u_next * np.array(u_vec) + v_next * np.array(v_vec)
                    n3 = self.node_manager.get_or_create(*p3)

                    # (i, j+1)
                    p4 = np.array(origin) + u_curr * np.array(u_vec) + v_next * np.array(v_vec)
                    n4 = self.node_manager.get_or_create(*p4)

                    self.add_plate([n1, n2, n3, n4], name)

    def _calculate_grid_lines(self, length: float, logic_type: str) -> List[Tuple[float, BeamType]]:
        """
        根据逻辑计算轴上的分割点。
        返回: List[(坐标, 穿过该坐标的正交梁类型)]
        """
        lines = []

        # 始终添加起点和终点 (FRAME)
        # 注意：这里的 FRAME 是默认属性，如果它位于中间，可能会被后续逻辑覆盖，
        # 但在这个函数里，我们主要标记“如果梁在这个位置生成，它是什么类型”

        if logic_type == "MAIN":
            # 按间距均匀布置: 0, s, 2s, ... L
            # 除了0和L是Frame，中间都是Main
            current = 0.0
            while current <= length + 1e-5:
                if math.isclose(current, 0) or math.isclose(current, length):
                    b_type = BeamType.FRAME
                else:
                    b_type = BeamType.MAIN

                lines.append((current, b_type))
                current += self.params.main_beam_spacing

            # 修正：如果最后一个点没对齐 L，强制把 L 加进去作为 Frame
            # (通常参数化建模要求 L 是间距的整数倍，或者最后一跨作为非标)
            # 这里的逻辑是：如果 while 循环最后 current 超过了 L，
            # 说明 L 处没有生成线。需要强制修正最后一个点或者添加 L。
            # 简单处理：如果最后一个点不是 L，添加 L
            if not math.isclose(lines[-1][0], length):
                lines.append((length, BeamType.FRAME))
            else:
                # 确保最后一个是 Frame
                lines[-1] = (length, BeamType.FRAME)

        elif logic_type in ["SECONDARY", "FRAME_GRID"]:
            if math.isclose(length, self.params.width):
                n_segs = self.params.num_secondary_width
            elif math.isclose(length, self.params.height):
                n_segs = self.params.num_secondary_height
            else:
                n_segs = 1  # 默认至少一段

            seg_len = length / n_segs

            # 基础分段点
            base_points = [i * seg_len for i in range(n_segs + 1)]

            for i, pos in enumerate(base_points):
                is_edge = i == 0 or i == n_segs
                # 如果逻辑是 FRAME_GRID (端盖竖向梁)，则中间也是 FRAME
                # 否则是 SECONDARY
                if logic_type == "FRAME_GRID":
                    b_type = BeamType.FRAME
                else:
                    b_type = BeamType.FRAME if is_edge else BeamType.SECONDARY

                lines.append((pos, b_type))

                # 处理支撑梁 (Brace)
                # 在当前点和下一点之间加中点
                if self.params.add_support_beam and i < n_segs:
                    mid_pos = pos + seg_len / 2
                    lines.append((mid_pos, BeamType.BRACE))

            # 按位置排序，以防 Brace 插入顺序问题
            lines.sort(key=lambda x: x[0])

        else:
            # 默认仅起点和终点
            lines.append((0.0, BeamType.FRAME))
            lines.append((length, BeamType.FRAME))

        return lines

    def add_plate(self, node_list: List[int], on_face: str):
        # calculate area of the plate
        p1 = self.points[node_list[0]]
        p2 = self.points[node_list[1]]
        p4 = self.points[node_list[3]]
        vec1 = p2.coordinates() - p1.coordinates()
        vec2 = p4.coordinates() - p1.coordinates()
        area = np.linalg.norm(np.cross(vec1, vec2))
        self.plates[self.plate_counter] = Plate(self.plate_counter, node_list, on_face, area)
        self.plate_counter += 1

    def _remove_elements_in_face_area(
        self,
        face_name: str,
        u_range: Tuple[float, float],
        v_range: Tuple[float, float],
        axis_val: float = 0.0,
        tolerance: float = 0.01,
    ):
        """
        基于 on_face 属性，删除指定面上、指定矩形范围内的梁和板。

        Args:
            face_name: 面的名称 (e.g., 'left', 'front', 'top')
            u_range: 该面局部坐标系下第一维度的范围 (min, max)
            v_range: 该面局部坐标系下第二维度的范围 (min, max)
            axis_val: 该面的法向坐标值 (e.g., 对于 'left' 面，X=0)
        """

        # 1. 确定局部坐标轴映射
        # 约定：
        # left/right (法向Y) -> u=x, v=z
        # front/back (法向X) -> u=y, v=z
        # top/bottom (法向Z) -> u=x, v=y

        if face_name in ["left", "right"]:
            axis_u, axis_v, axis_n = 0, 2, 1  # x, z, y
        elif face_name in ["front", "back"]:
            axis_u, axis_v, axis_n = 1, 2, 0  # y, z, x
        elif face_name in ["top", "bottom"]:
            axis_u, axis_v, axis_n = 0, 1, 2  # x, y, z

        u_min, u_max = sorted(u_range)
        v_min, v_max = sorted(v_range)

        # 辅助函数：判断点是否在范围内 (严格内部，保留边界)
        def is_in_rect(p_coords):
            u = p_coords[axis_u]
            v = p_coords[axis_v]
            n = p_coords[axis_n]
            if not math.isclose(n, axis_val, abs_tol=tolerance):
                return False
            return (
                (u > u_min + tolerance)
                and (u < u_max - tolerance)
                and (v > v_min + tolerance)
                and (v < v_max - tolerance)
            )

        # ---------------------------
        # 2. 处理梁 (Beams)
        # ---------------------------
        beams_to_remove = []
        points_to_remove = set()
        for bid, beam in self.beams.items():
            # 利用 on_face 快速筛选
            if beam.on_face != face_name:
                continue

            p1 = self.points[beam.start_node].coordinates()
            p2 = self.points[beam.end_node].coordinates()
            mid_point = (p1 + p2) / 2

            if is_in_rect(mid_point):
                beams_to_remove.append(bid)
                points_to_remove.add(beam.start_node)
                points_to_remove.add(beam.end_node)

        for bid in beams_to_remove:
            del self.beams[bid]

        # ---------------------------
        # 3. 处理板 (Plates)
        # ---------------------------
        plates_to_remove = []
        for pid, plate in self.plates.items():
            # 利用 on_face 快速筛选
            if plate.on_face != face_name:
                continue

            # 计算板的质心 (简单平均)
            # 注意：plate.nodes 是节点ID列表
            node_coords = [self.points[nid].coordinates() for nid in plate.nodes]
            centroid = np.mean(node_coords, axis=0)

            if is_in_rect(centroid):
                plates_to_remove.append(pid)

        for pid in plates_to_remove:
            del self.plates[pid]

        for pid in points_to_remove:
            if pid in self.points and is_in_rect(self.points[pid].coordinates()):
                del self.points[pid]

    def _get_section_for_beam(self, b_type: BeamType) -> SectionProperties:
        """根据梁类型和参数配置，从库中获取对应的截面对象"""

        # 1. 确定截面类型
        sec_type = BEAM_TYPE_MAP.get(b_type)
        if not sec_type:
            raise ValueError(f"No section mapping for {b_type}")

        # 2. 确定索引 (从 params 中读取)
        # 假设 StructuralParameters 里有 frame_section_idx, main_beam_idx 等字段
        if b_type == BeamType.FRAME:
            idx = self.params.frame_section_index
        elif b_type == BeamType.MAIN:
            idx = self.params.main_beam_section_index
        elif b_type == BeamType.SECONDARY:
            idx = self.params.secondary_beam_section_index
        elif b_type == BeamType.BRACE:
            idx = self.params.brace_section_index
        else:
            idx = 0

        # 3. 从库中获取
        return self.section_lib.get_section(sec_type, idx)

    def calculate_mass(self):
        """计算结构总质量，单位 kg"""
        total_mass = 0.0

        # 计算梁质量
        for beam in self.beams.values():
            if beam.is_virtual:
                continue  # 虚拟梁不计质量
            start_pt = self.points[beam.start_node]
            end_pt = self.points[beam.end_node]
            length = start_pt.distance_to(end_pt)
            density = beam.section.density
            area = beam.section.area
            mass = density * area * length
            total_mass += mass

        # 计算板质量
        for plate in self.plates.values():
            density = 7850.0  # 钢板密度 (kg/m^3)，假设为常数
            thickness = self.params.plate_thickness
            area = plate.area
            mass = density * area * thickness
            total_mass += mass

        return total_mass

    def get_beam_stats(self):
        """
        按照梁类型统计数量和总长度（m）
        """
        stats = {}
        for b_type in BeamType:
            stats[b_type] = {"count": 0, "total_length": 0.0}

        for beam in self.beams.values():
            b_type = beam.beam_type
            start_pt = self.points[beam.start_node]
            end_pt = self.points[beam.end_node]
            length = start_pt.distance_to(end_pt)

            stats[b_type]["count"] += 1
            stats[b_type]["total_length"] += length

        return stats


# --- 使用示例 ---
if __name__ == "__main__":
    from src.structure.mesh_generator import MeshGenerator
    from src.utils.model_utils import StructureVisualizer, ViewType

    # 配置参数
    params = StructuralParameters(
        length=12.0,  # 10米长
        width=3.7,  # 3米宽
        height=3.3,  # 3米高
        main_beam_spacing=1.05,  # 主梁每1.5米一道
        num_secondary_width=3,  # 宽度方向分3段 (2条中间次梁)
        num_secondary_height=3,  # 高度方向分3段
        add_support_beam=True,  # 添加支撑梁
    )

    model = StructuralModel(params, SectionLibrary())
    model.generate()

    # mesh_generator = MeshGenerator(model, 0.5)
    # model = mesh_generator.generate()

    visualizer = StructureVisualizer(model)
    visualizer.show_face(ViewType.FULL_3D)
    # visualizer.show_all_views(grid=False)
