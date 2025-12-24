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
class ExtensionParameters:
    """侧面凸出部分参数"""

    active: bool = True
    length: float = 1.1  # 凸出长度 (Y负方向)
    width: float = 1.85  # 凸出宽度 (X方向)
    height: float = 0.0  # 凸出高度
    offset_x: float = 0.0  # 在主结构X方向的起始位置 (离X=0的距离)


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

    extension: ExtensionParameters = field(default_factory=ExtensionParameters)  # 侧面凸出部分参数

    windows: List[Window] = field(default_factory=list)  # 窗户参数列表
    win_bottom: float = 1.0  # 窗户底部离地高度
    win_height: float = 1.5  # 窗户高度
    plate_thickness: float = 0.01  # 钢板厚度
    pressure: float = 30  # 内压 (kPa)

    # extension的后两个参数计算得出
    def __post_init__(self):
        if self.extension.active:
            self.extension.height = self.height
            self.extension.offset_x = self.width - self.extension.width


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
        ext = self.params.extension

        # -----------------------------------------------------
        # 1. 准备 Extension 的边界信息 (用于网格对齐)
        # -----------------------------------------------------
        # 在 X 方向上的强制分割点 (对应 Top/Bottom 面)
        # 这些点确保主结构在 Extension 的墙角处有节点
        x_mandatory = []
        hole_definition = None

        if ext.active:
            x_start = ext.offset_x
            x_end = ext.offset_x + ext.width
            x_mandatory = [x_start, x_end]

            # 定义连接面(Left)上的开洞区域 (u=x, v=z)
            # 格式: (u_min, u_max, v_min, v_max)
            hole_definition = (x_start, x_end, 0, ext.height)

        # -----------------------------------------------------
        # 2. 生成主结构面
        # -----------------------------------------------------

        # Top (Z=H)
        # 传入 mandatory_u_points，使得顶部梁在 extension 边界处断开
        self._build_face(
            origin=(0, 0, H),
            u_vec=(1, 0, 0),
            v_vec=(0, 1, 0),
            u_len=W,
            v_len=L,
            name="top",
            u_beam_logic="MAIN",
            v_beam_logic="SECONDARY",
            mandatory_u_points=x_mandatory,
        )

        # Bottom (Z=0)
        self._build_face(
            origin=(0, 0, 0),
            u_vec=(1, 0, 0),
            v_vec=(0, 1, 0),
            u_len=W,
            v_len=L,
            name="bottom",
            u_beam_logic="MAIN",
            v_beam_logic="SECONDARY",
            mandatory_u_points=x_mandatory,
        )

        # Front (X=W)
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

        # Back (X=0)
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

        # Left (Y=0) - 核心连接面
        # 使用 hole_rect 参数直接挖洞，不再事后删除
        self._build_face(
            origin=(0, 0, 0),
            u_vec=(1, 0, 0),
            v_vec=(0, 0, 1),
            u_len=W,
            v_len=H,
            name="left",
            u_beam_logic="SECONDARY",
            v_beam_logic="FRAME_GRID",
            hole_rect=hole_definition,  # 传入洞口定义
            mandatory_u_points=x_mandatory,  # 确保洞口边缘有立柱
        )

        # Right (Y=L)
        self._build_face(
            origin=(0, L, 0),
            u_vec=(1, 0, 0),
            v_vec=(0, 0, 1),
            u_len=W,
            v_len=H,
            name="right",
            u_beam_logic="SECONDARY",
            v_beam_logic="FRAME_GRID",
        )

        # -----------------------------------------------------
        # 3. 生成 Extension (如果激活)
        # -----------------------------------------------------
        if ext.active:
            self._generate_extension_block()

        self._cleanup_orphans()

    def _generate_extension_block(self):
        """生成左侧凸出区域，并直接与主结构节点连接"""
        ext = self.params.extension

        # 局部参数
        L_ext = ext.length
        W_ext = ext.width
        H_ext = ext.height
        Off_x = ext.offset_x

        # 获取主结构在 X 方向的网格线 (复用逻辑以保证对齐)
        # 重点：只取 Extension 范围内的线
        full_u_lines = self._calculate_grid_lines(
            self.params.width, "SECONDARY", mandatory_points=[Off_x, Off_x + W_ext]
        )

        # 筛选出属于 Extension 范围内的 X 向网格
        aligned_u_lines = []
        for pos, b_type in full_u_lines:
            # 容差判断是否在范围内
            if pos >= Off_x - 1e-4 and pos <= (Off_x + W_ext) + 1e-1:
                local_pos = pos - Off_x
                # 边界强制为 FRAME
                if math.isclose(local_pos, 0) or math.isclose(local_pos, W_ext):
                    real_type = BeamType.FRAME
                else:
                    real_type = b_type
                aligned_u_lines.append((local_pos, real_type))

        # 1. Top (Z = H_ext)
        self._build_face(
            origin=(Off_x, -L_ext, H_ext),
            u_vec=(1, 0, 0),
            v_vec=(0, 1, 0),
            u_len=W_ext,
            v_len=L_ext,
            name="top",
            u_beam_logic="none",
            v_beam_logic="SECONDARY",
            override_u_grid=aligned_u_lines,  # 强制对齐主结构
        )

        # 2. Bottom (Z = 0)
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

        # 3. Front (Y 方向侧面)
        self._build_face(
            origin=(Off_x + W_ext, -L_ext, 0),
            u_vec=(0, 1, 0),
            v_vec=(0, 0, 1),
            u_len=L_ext,
            v_len=H_ext,
            name="front",
            u_beam_logic="SECONDARY",
            v_beam_logic="none",  # 立柱
        )

        # 4. Back (Y 方向侧面)
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

        # 5. End Cap (Extension的最外端面)
        # 仅在边界有立柱
        end_u_grid = [(0.0, BeamType.FRAME), (W_ext, BeamType.FRAME)]
        # 如果中间还有其他Frame线也加上
        for pos, btype in aligned_u_lines:
            if not (math.isclose(pos, 0) or math.isclose(pos, W_ext)):
                end_u_grid.append((pos, BeamType.FRAME))
        end_u_grid.sort(key=lambda x: x[0])

        self._build_face(
            origin=(Off_x, -L_ext, 0),
            u_vec=(1, 0, 0),
            v_vec=(0, 0, 1),
            u_len=W_ext,
            v_len=H_ext,
            name="left",
            u_beam_logic="SECONDARY",
            v_beam_logic="FRAME",
            override_u_grid=end_u_grid,
        )

        # ---------------------------------------------------------
        # 2. 设备区 (Equipment Area) - 新增代码
        # ---------------------------------------------------------
        # 位于 X [0, Off_x] 区间，填补左上角缺口
        # 逻辑类型使用 "none" (或任意非标准字符串)，只生成四周的 Frame 梁

        # Equip Top (Z = H)
        self._build_face(
            origin=(0, -L_ext, H_ext),  # 起点在最左后角
            u_vec=(1, 0, 0),
            v_vec=(0, 1, 0),  # X, Y 方向
            u_len=Off_x,
            v_len=L_ext,  # 长度填满 Off_x
            name="top",
            u_beam_logic="none",  # 仅外框
            v_beam_logic="none",
            beam_only=True,  # 仅生成边框梁
        )

        # Equip Bottom (Z = 0)
        self._build_face(
            origin=(0, -L_ext, 0),
            u_vec=(1, 0, 0),
            v_vec=(0, 1, 0),
            u_len=Off_x,
            v_len=L_ext,
            name="bottom",
            u_beam_logic="none",
            v_beam_logic="none",
            beam_only=True,  # 仅生成边框梁
        )

        # Equip Back (X = 0) - 最外侧的背面墙
        self._build_face(
            origin=(0, -L_ext, 0),
            u_vec=(0, 1, 0),
            v_vec=(0, 0, 1),  # Y, Z 方向
            u_len=L_ext,
            v_len=H_ext,
            name="back",
            u_beam_logic="none",
            v_beam_logic="none",
            beam_only=True,  # 仅生成边框梁
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
        override_u_grid: List[Tuple[float, BeamType]] = None,
        override_v_grid: List[Tuple[float, BeamType]] = None,
        hole_rect: Tuple[float, float, float, float] = None,  # (u_min, u_max, v_min, v_max)
        mandatory_u_points: List[float] = None,
        beam_only: bool = False,
    ):
        """
        构建面，支持定义空洞 (Hole) 以便拼接子结构。
        hole_rect: 定义在局部(u, v)坐标系下的矩形范围，范围内的元素将不生成。
        """
        if override_v_grid is not None:
            v_lines = override_v_grid
        else:
            v_lines = self._calculate_grid_lines(v_len, u_beam_logic)

        if override_u_grid is not None:
            u_lines = override_u_grid
        else:
            # 这里传入强制点，确保主结构网格在Extension边界处断开/生成节点
            u_lines = self._calculate_grid_lines(u_len, v_beam_logic, mandatory_points=mandatory_u_points)

        # 辅助函数: 检查线段或区域是否在洞内
        def is_in_hole(u1, v1, u2, v2):
            if not hole_rect:
                return False
            h_u_min, h_u_max, h_v_min, h_v_max = hole_rect
            # 简单的全包含检测 (容差 1e-3)
            tol = 1e-3
            # 检查中心点是否在洞内
            c_u, c_v = (u1 + u2) / 2, (v1 + v2) / 2
            in_u = (c_u > h_u_min + tol) and (c_u < h_u_max - tol)
            in_v = (c_v > h_v_min + tol) and (c_v < h_v_max - tol)
            return in_u and in_v

        # 2. 网格生成循环
        for i in range(len(u_lines)):
            for j in range(len(v_lines)):
                u_curr, type_v_beam_here = u_lines[i]
                v_curr, type_u_beam_here = v_lines[j]

                pos = np.array(origin) + u_curr * np.array(u_vec) + v_curr * np.array(v_vec)
                node_id = self.node_manager.get_or_create(*pos)

                # --- 生成 U方向梁 ---
                if i < len(u_lines) - 1:
                    u_next = u_lines[i + 1][0]
                    # 检查是否在洞内，在洞内则跳过
                    if not is_in_hole(u_curr, v_curr, u_next, v_curr):
                        pos_next = np.array(origin) + u_next * np.array(u_vec) + v_curr * np.array(v_vec)
                        next_id = self.node_manager.get_or_create(*pos_next)

                        b_type = type_u_beam_here
                        if math.isclose(v_curr, 0) or math.isclose(v_curr, v_len):
                            b_type = BeamType.FRAME

                        is_virtual = b_type == virtual_beam
                        sec_prop = self._get_section_for_beam(b_type)
                        self.beam_manager.add_or_update_beam(
                            node_id, next_id, b_type, sec_prop, is_virtual, name
                        )

                # --- 生成 V方向梁 ---
                if j < len(v_lines) - 1:
                    v_next = v_lines[j + 1][0]
                    # 检查是否在洞内
                    if not is_in_hole(u_curr, v_curr, u_curr, v_next):
                        pos_next = np.array(origin) + u_curr * np.array(u_vec) + v_next * np.array(v_vec)
                        next_id = self.node_manager.get_or_create(*pos_next)

                        b_type = type_v_beam_here
                        if math.isclose(u_curr, 0) or math.isclose(u_curr, u_len):
                            b_type = BeamType.FRAME

                        is_virtual = b_type == virtual_beam
                        sec_prop = self._get_section_for_beam(b_type)
                        ext_x_offset = self.params.extension.offset_x

                        if (
                            name in ["top", "bottom"]
                            and math.isclose(pos[0], ext_x_offset)
                            and pos[1] > -1e-4
                        ):
                            self.beam_manager.add_or_update_beam(
                                node_id, next_id, b_type, sec_prop, True, name
                            )
                        else:
                            self.beam_manager.add_or_update_beam(
                                node_id, next_id, b_type, sec_prop, is_virtual, name
                            )

                if beam_only:
                    continue

                # --- 生成 板 ---
                if i < len(u_lines) - 1 and j < len(v_lines) - 1:
                    u_next = u_lines[i + 1][0]
                    v_next = v_lines[j + 1][0]

                    # 检查板中心是否在洞内
                    if not is_in_hole(u_curr, v_curr, u_next, v_next):
                        p2 = np.array(origin) + u_next * np.array(u_vec) + v_curr * np.array(v_vec)
                        p3 = np.array(origin) + u_next * np.array(u_vec) + v_next * np.array(v_vec)
                        p4 = np.array(origin) + u_curr * np.array(u_vec) + v_next * np.array(v_vec)

                        n2 = self.node_manager.get_or_create(*p2)
                        n3 = self.node_manager.get_or_create(*p3)
                        n4 = self.node_manager.get_or_create(*p4)

                        self.add_plate([node_id, n2, n3, n4], name)

    def _calculate_grid_lines(
        self, length: float, logic_type: str, mandatory_points: List[float] = None
    ) -> List[Tuple[float, BeamType]]:
        """
        根据逻辑计算轴上的分割点。
        新增功能: mandatory_points 用于强制在特定位置生成节点 (通常为 FRAME 类型)
        """
        # 使用字典来去重: {position: beam_type}
        points_map: Dict[float, BeamType] = {}

        # --- 1.原有生成逻辑 (生成的点存入 points_map) ---
        if logic_type == "MAIN":
            current = 0.0
            while current <= length + 1e-5:
                # 边界强制为 FRAME
                if math.isclose(current, 0) or math.isclose(current, length):
                    b_type = BeamType.FRAME
                else:
                    b_type = BeamType.MAIN
                points_map[current] = b_type
                current += self.params.main_beam_spacing

            # 修正尾部: 确保 length 处有点
            if not any(math.isclose(k, length) for k in points_map):
                points_map[length] = BeamType.FRAME
            else:
                # 确保终点是 Frame
                for k in points_map:
                    if math.isclose(k, length):
                        points_map[k] = BeamType.FRAME

        elif logic_type in ["SECONDARY", "FRAME_GRID"]:
            # 确定分段数
            if math.isclose(length, self.params.width):
                n_segs = self.params.num_secondary_width
            elif math.isclose(length, self.params.height):
                n_segs = self.params.num_secondary_height
            else:
                n_segs = 1

            seg_len = length / n_segs
            base_points = [i * seg_len for i in range(n_segs + 1)]

            for i, pos in enumerate(base_points):
                is_edge = i == 0 or i == n_segs
                if logic_type == "FRAME_GRID":
                    b_type = BeamType.FRAME
                else:
                    b_type = BeamType.FRAME if is_edge else BeamType.SECONDARY

                points_map[pos] = b_type

                # 支撑梁逻辑
                if self.params.add_support_beam and i < n_segs:
                    mid_pos = pos + seg_len / 2
                    if mid_pos not in points_map:
                        points_map[mid_pos] = BeamType.BRACE

        else:
            points_map[0.0] = BeamType.FRAME
            points_map[length] = BeamType.FRAME

        # --- 2. 注入强制点 (Extension 边界) ---
        if mandatory_points:
            for pt in mandatory_points:
                # 检查点是否已存在 (容差判断)
                found_key = None
                for k in points_map:
                    if math.isclose(k, pt, abs_tol=1e-4):
                        found_key = k
                        break

                if found_key is not None:
                    # 如果点已存在，强制升级为 FRAME (因为这是连接处，需要强梁/柱)
                    points_map[found_key] = BeamType.FRAME
                else:
                    # 如果点不存在，插入新点，类型为 FRAME
                    points_map[pt] = BeamType.FRAME

        # --- 3. 排序并输出 ---
        sorted_pos = sorted(points_map.keys())
        return [(pos, points_map[pos]) for pos in sorted_pos]

    def _cleanup_orphans(self):
        """清理没有任何梁或板连接的孤立节点"""
        used_nodes = set()

        # 收集梁用到的节点
        for beam in self.beams.values():
            used_nodes.add(beam.start_node)
            used_nodes.add(beam.end_node)

        # 收集板用到的节点
        for plate in self.plates.values():
            for n in plate.nodes:
                used_nodes.add(n)

        # 找出所有现存节点
        all_node_ids = list(self.points.keys())
        deleted_count = 0

        for nid in all_node_ids:
            if nid not in used_nodes:
                del self.points[nid]
                deleted_count += 1

        # print(f"Cleaned up {deleted_count} isolated nodes.")

    def add_plate(self, node_list: List[int], on_face: str):
        # calculate area of the plate
        p1 = self.points[node_list[0]]
        p2 = self.points[node_list[1]]
        p4 = self.points[node_list[3]]
        vec1 = p2.coordinates() - p1.coordinates()
        vec2 = p4.coordinates() - p1.coordinates()
        area = np.linalg.norm(np.cross(vec1, vec2))
        if area < 1e-6:
            return  # 忽略面积过小的板
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
