import bisect
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.structure.enums import BeamType
from src.structure.section_library import SectionLibrary, SectionProperties, SectionType

# ==========================================
# 1. 基础数据结构保持不变
# ==========================================


@dataclass
class ExtensionParameters:
    """侧面凸出部分参数"""

    active: bool = True
    length: float = 1.1  # Y负方向长度
    width: float = 1.85  # X方向宽度
    height: float = 0.0  # 自动计算
    offset_x: float = 0.0  # 自动计算


@dataclass
class StructuralParameters:
    """结构参数类"""

    length: float
    width: float
    height: float
    main_beam_spacing: float
    num_secondary_width: int
    num_secondary_height: int
    add_support_beam: bool = True

    # 截面索引
    frame_section_index: int = 0
    main_beam_section_index: int = 0
    secondary_beam_section_index: int = 0
    brace_section_index: int = 0

    extension: ExtensionParameters = field(default_factory=ExtensionParameters)

    # 其他物理参数
    plate_thickness: float = 0.005

    def __post_init__(self):
        if self.extension.active:
            self.extension.width = self.width / 2
            self.extension.height = self.height
            self.extension.offset_x = self.width - self.extension.width


@dataclass
class Point:
    id: int
    x: float
    y: float
    z: float

    def coordinates(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def distance_to(self, other: "Point") -> float:
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
    is_virtual: bool
    on_face: str
    parent_id: int = None
    response: BeamResponse = None  # 存储分析结果


@dataclass
class Plate:
    id: int
    nodes: List[int]
    on_face: str
    area: float


# ==========================================
# 2. 管理器类 (复用)
# ==========================================


class NodeManager:
    def __init__(self):
        self._node_map: Dict[Tuple[int, int, int], int] = {}
        self.points: Dict[int, Point] = {}
        self.counter = 0
        self.precision = 1000

    def get_or_create(self, x: float, y: float, z: float) -> int:
        key = (int(round(x * self.precision)), int(round(y * self.precision)), int(round(z * self.precision)))
        if key in self._node_map:
            return self._node_map[key]
        node_id = self.counter
        self.points[node_id] = Point(node_id, x, y, z)
        self._node_map[key] = node_id
        self.counter += 1
        return node_id


class BeamManager:
    def __init__(self, node_manager: NodeManager):
        self.node_manager = node_manager
        self.beams: Dict[int, Beam] = {}
        self._beam_map: Dict[Tuple[int, int], int] = {}
        self.counter = 0
        # 优先级: FRAME > MAIN > SECONDARY > BRACE
        self.priority = {BeamType.FRAME: 0, BeamType.MAIN: 1, BeamType.SECONDARY: 2, BeamType.BRACE: 3}

    def add_or_update_beam(
        self,
        n1: int,
        n2: int,
        b_type: BeamType,
        section: SectionProperties,
        is_virtual: bool,
        face: str,
        parent_id: int = None,
    ):
        if n1 == n2:
            return
        key = tuple(sorted((n1, n2)))

        if key in self._beam_map:
            # 冲突解决：保留高优先级类型
            exist_beam = self.beams[self._beam_map[key]]
            if self.priority[b_type] < self.priority[exist_beam.beam_type]:
                exist_beam.beam_type = b_type
                exist_beam.section = section
                exist_beam.is_virtual = is_virtual
            # 如果新类型和旧类型一样，或者优先级低，则不需要做任何事
            # 除非我们需要合并 face 属性 (例如 "top/front")，这里简化处理
            return

        beam_id = self.counter
        self.beams[beam_id] = Beam(beam_id, n1, n2, b_type, section, is_virtual, face, parent_id)
        self._beam_map[key] = beam_id
        self.counter += 1

    def get_beam_length(self, beam_id: int) -> float:
        beam = self.beams[beam_id]
        p1 = self.node_manager.points[beam.start_node]
        p2 = self.node_manager.points[beam.end_node]
        return p1.distance_to(p2)


# ==========================================
# 3. 核心：几何生成引擎 (The New Architecture)
# ==========================================


@dataclass
class Zone:
    name: str  # "main", "ext", "equip"
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    z_range: Tuple[float, float]
    is_hollow: bool = False

    def contains(self, x, y, z, tol=1e-4) -> bool:
        """点是否在闭包围盒内"""
        return (
            self.x_range[0] - tol <= x <= self.x_range[1] + tol
            and self.y_range[0] - tol <= y <= self.y_range[1] + tol
            and self.z_range[0] - tol <= z <= self.z_range[1] + tol
        )

    def on_surface(self, x, y, z, tol=1e-4) -> bool:
        """点是否在表面（包含棱边）"""
        if not self.contains(x, y, z, tol):
            return False
        on_x = math.isclose(x, self.x_range[0], abs_tol=tol) or math.isclose(x, self.x_range[1], abs_tol=tol)
        on_y = math.isclose(y, self.y_range[0], abs_tol=tol) or math.isclose(y, self.y_range[1], abs_tol=tol)
        on_z = math.isclose(z, self.z_range[0], abs_tol=tol) or math.isclose(z, self.z_range[1], abs_tol=tol)
        return on_x or on_y or on_z

    def on_edge(self, x, y, z, tol=1e-4) -> bool:
        """点是否在棱线上（即 FRAME 的位置）"""
        if not self.contains(x, y, z, tol):
            return False
        on_x = math.isclose(x, self.x_range[0], abs_tol=tol) or math.isclose(x, self.x_range[1], abs_tol=tol)
        on_y = math.isclose(y, self.y_range[0], abs_tol=tol) or math.isclose(y, self.y_range[1], abs_tol=tol)
        on_z = math.isclose(z, self.z_range[0], abs_tol=tol) or math.isclose(z, self.z_range[1], abs_tol=tol)

        # 只有满足至少两个面约束，才是在棱上
        return (on_x and on_y) or (on_x and on_z) or (on_y and on_z)

    def get_face(self, x, y, z, tol=1e-4) -> Optional[str]:
        """
        返回面名称，用于后续荷载施加

        约定：
        结构为一三维立方体，全局坐标系定义如下：
        - X轴：水平横向，正方向指向前方
        - Y轴：水平纵向，正方向指向右侧
        - Z轴：竖向，正方向向上

        假定观察者站在结构外部，X轴正向一定距离处，基于观察者视角，将面按如下规则命名：
        - top：z = max
        - bottom：z = min
        - front：x = max
        - back：x = min
        - left：y = min
        - right：y = max
        """
        if math.isclose(z, self.z_range[1], abs_tol=tol):
            return "top"
        if math.isclose(z, self.z_range[0], abs_tol=tol):
            return "bottom"
        if math.isclose(x, self.x_range[0], abs_tol=tol):
            return "back"
        if math.isclose(x, self.x_range[1], abs_tol=tol):
            return "front"
        if math.isclose(y, self.y_range[0], abs_tol=tol):
            return "left"
        if math.isclose(y, self.y_range[1], abs_tol=tol):
            return "right"
        return None


class LayoutEngine:
    def __init__(self, params: StructuralParameters):
        self.p = params
        self.zones: List[Zone] = []
        self.main_zone = None

        # 坐标网格集合
        self.grids_x = set()
        self.grids_y = set()
        self.grids_z = set()

        # 支撑梁位置标记 (X 和 Z 方向都需要)
        self.secondary_x_locs = set()
        self.brace_x_locs = set()
        self.brace_z_locs = set()

        self._init_geometry()

    def _init_geometry(self):
        L, W, H = self.p.length, self.p.width, self.p.height
        ext = self.p.extension

        # 1. 定义区域 (Zone)
        self.main_zone = Zone("main", (0, W), (0, L), (0, H))
        self.zones.append(self.main_zone)

        if ext.active:
            # 假设 extension 在 X 大值侧，向 Y 负方向生长
            # Extension: [offset_x, W]
            ext_x_start = ext.offset_x
            ext_x_end = ext.offset_x + ext.width  # 其实就是 W
            self.zones.append(Zone("ext", (ext_x_start, ext_x_end), (-ext.length, 0), (0, ext.height)))
            self.zones.append(
                Zone("equip", (0, ext_x_start), (-ext.length, 0), (0, ext.height), is_hollow=True)
            )

        # 2. 生成边界网格
        for z in self.zones:
            self.grids_x.update(z.x_range)
            self.grids_y.update(z.y_range)
            self.grids_z.update(z.z_range)

        # 3. 填充内部细分网格
        self._fill_main_beam_y_grids()  # Y轴: 主梁间距
        self._fill_secondary_x_grids()  # X轴: 顶/底面次梁分布
        self._fill_secondary_z_grids()  # Z轴: 前/后面次梁分布 (新增)

        self.sorted_x = sorted(list(self.grids_x))
        self.sorted_y = sorted(list(self.grids_y))
        self.sorted_z = sorted(list(self.grids_z))

    def _fill_main_beam_y_grids(self):
        """Y方向切片：控制 Top/Bottom/Front/Back 面的竖向肋或主梁"""
        curr = 0.0
        # 主舱体长 L
        while curr <= self.p.length + 1e-5:
            self.grids_y.add(curr)
            curr += self.p.main_beam_spacing
        self.grids_y.add(self.p.length)

        # Extension 区域 Y 方向 (通常比较短，只加头尾即可，暂不加密)
        if self.p.extension.active:
            self.grids_y.add(-self.p.extension.length)

    def _fill_secondary_x_grids(self):
        """X方向切片：控制 Top/Bottom 面的次梁"""
        n_segs = self.p.num_secondary_width
        step = self.p.width / n_segs

        for i in range(1, n_segs):
            pos = i * step
            self.grids_x.add(pos)
            self.secondary_x_locs.add(pos)
            # 支撑梁 (次梁中间)
            if self.p.add_support_beam:
                brace_pos = pos - (step / 2)
                self.grids_x.add(brace_pos)
                self.brace_x_locs.add(brace_pos)

        # 补齐最后一格的支撑
        if self.p.add_support_beam:
            self.grids_x.add(self.p.width - (step / 2))
            self.brace_x_locs.add(self.p.width - (step / 2))

    def _fill_secondary_z_grids(self):
        """Z方向切片：控制 Front/Back 面的次梁 (沿高度分布)"""
        # 假设高度方向分段逻辑与宽度一致
        n_segs = self.p.num_secondary_height
        step = self.p.height / n_segs

        for i in range(1, n_segs):
            pos = i * step
            self.grids_z.add(pos)
            # 支撑梁逻辑同上
            if self.p.add_support_beam:
                brace_pos = pos - (step / 2)
                self.grids_z.add(brace_pos)
                self.brace_z_locs.add(brace_pos)

        if self.p.add_support_beam:
            self.grids_z.add(self.p.height - (step / 2))
            self.brace_z_locs.add(self.p.height - (step / 2))

    def is_in_opening(self, x, y, z, tol=1e-4) -> bool:
        """检查点是否位于 Ext 和 Main 的连通口 (y=0, x > offset_x)"""
        # 连通口定义：平面 y=0, x区间 [offset_x, width], z区间 [0, height]
        # 注意：这里我们放宽一点 z 的判定，只处理中间部分，保留上下边框梁(Frame)
        ext = self.p.extension
        if not ext.active:
            return False

        # 判定面：y=0
        if not math.isclose(y, 0, abs_tol=tol):
            return False

        # 判定 X 范围 (严格在内部，保留两侧立柱)
        # 如果 x == offset_x 或 x == width，那是立柱位置，不能是洞
        in_x = (x > ext.offset_x + tol) and (x < self.p.width - tol)

        # 判定 Z 范围 (严格在内部，保留上下主梁)
        in_z = (z > 0 + tol) and (z < ext.height - tol)

        return in_x and in_z

    def determine_beam(
        self, p_start: np.ndarray, p_end: np.ndarray, axis: str
    ) -> Tuple[Optional[BeamType], str, bool]:
        """
        判断梁类型
        Returns: (BeamType, face_name, is_virtual)
        """
        mid = (p_start + p_end) / 2
        x, y, z = mid

        # 1. 区域判定
        active_zones = [zn for zn in self.zones if zn.contains(x, y, z)]
        if not active_zones:
            return None, "", False

        # 开洞检查
        if self.is_in_opening(x, y, z):
            return None, "", False

        # 2. 表面检查 (只生成表面的构件)
        # 注意：虚拟梁(Split Line)通常位于主结构的表面(Top/Bottom)，所以也能通过此检查
        is_on_any_surface = any(zn.on_surface(x, y, z) for zn in active_zones)
        if not is_on_any_surface:
            return None, "", False

        # 3. 确定 Face 名称
        face_name = active_zones[0].get_face(x, y, z)

        # 4. FRAME (棱边) 判定 - 优先级最高
        is_frame_edge = any(zn.on_edge(x, y, z) for zn in active_zones)

        # Equip空心区域特殊处理
        in_hollow = all(zn.is_hollow for zn in active_zones)
        if in_hollow:
            return (BeamType.FRAME, face_name, False) if is_frame_edge else (None, "", False)

        if is_frame_edge:
            return BeamType.FRAME, face_name, False

        # ==========================================================
        # 5. [重构] 梁类型判定 (包含虚拟梁逻辑)
        # 策略：先按正常规则判定。如果判定为无梁，但又在 split_line 上，则设为虚拟梁。
        # ==========================================================

        # 预先检查：是否是 Extension 引入的分割线
        is_split_line = False
        if self.p.extension.active:
            is_split_line = math.isclose(x, self.p.extension.offset_x, abs_tol=1e-4)

        # 仅针对 Main 区域的主结构部分 (Y > 0) 启用虚拟梁逻辑
        # Ext区域 (Y < 0) 的 offset_x 线是实墙，会在下面的常规逻辑中被捕获
        check_virtual_necessity = is_split_line and (y > 1e-4)

        proposed_type = None
        is_virtual = False

        # --- 常规规则判定 (复制自你原有的第6步，稍作整理) ---

        # === Axis X (水平横向) ===
        if axis == "x":
            # Case A: Top/Bottom 面 -> 主梁 (Main)
            if math.isclose(z, 0) or math.isclose(z, self.p.height):
                proposed_type = BeamType.MAIN

            # Case B: Front/Back 面 -> 次梁或支撑
            else:
                is_brace = any(math.isclose(z, bz) for bz in self.brace_z_locs)
                if is_brace:
                    proposed_type = BeamType.BRACE
                else:
                    proposed_type = BeamType.SECONDARY

        # === Axis Y (水平纵向) ===
        elif axis == "y":
            # Case A: Top/Bottom 面 -> 次梁或支撑
            if math.isclose(z, 0) or math.isclose(z, self.p.height):
                is_brace = any(math.isclose(x, bx) for bx in self.brace_x_locs)
                if is_brace:
                    proposed_type = BeamType.BRACE
                elif any(math.isclose(x, sx) for sx in self.secondary_x_locs):
                    proposed_type = BeamType.SECONDARY

            # Case B: Front/Back/Side 面 (包含 Ext 的侧墙) -> 次梁或支撑
            else:
                # 检查是否在垂直方向的支撑位置
                is_brace = any(math.isclose(z, bz) for bz in self.brace_z_locs)
                if is_brace:
                    proposed_type = BeamType.BRACE
                else:
                    proposed_type = BeamType.SECONDARY

        # === Axis Z (竖向) ===
        elif axis == "z":
            # 只要在表面且是竖向的，除非是Frame(已处理)，否则定义为主梁
            if math.isclose(y, self.p.length):
                if any(math.isclose(x, sx) for sx in self.secondary_x_locs):
                    proposed_type = BeamType.FRAME
                elif any(math.isclose(x, bx) for bx in self.brace_x_locs):
                    proposed_type = BeamType.BRACE
                else:
                    proposed_type = BeamType.MAIN
                    is_virtual = True
                # is_virtual = not any(math.isclose(x, sx) for sx in self.secondary_x_locs)
            else:
                proposed_type = BeamType.MAIN

        # ==========================================================
        # 6. 最终裁决：实体梁 vs 虚拟梁 vs 无梁
        # ==========================================================

        if proposed_type is not None:
            # 规则说这里有梁，那就是实梁 (即使它在 split_line 上，那也是恰好重合)
            return proposed_type, face_name, is_virtual

        elif check_virtual_necessity:
            # 规则说这里没梁 (proposed_type is None)
            # 但它位于分割线上，且在主结构区域 -> 必须生成虚拟梁以保证网格连接
            # 默认类型设为 MAIN (刚度由 is_virtual=True 消除)
            return BeamType.MAIN, face_name, True

        else:
            # 既没规则梁，也不是分割线 -> 空气
            return None, "", False


# ==========================================
# 4. 主模型类 (入口)
# ==========================================

BEAM_TYPE_MAP = {
    BeamType.FRAME: SectionType.BOX,
    BeamType.MAIN: SectionType.H_MAIN,
    BeamType.SECONDARY: SectionType.H_SEC,
    BeamType.BRACE: SectionType.CHANNEL,
}


class StructuralModel:
    def __init__(self, params: StructuralParameters, section_lib: SectionLibrary):
        self.params = params
        self.section_lib = section_lib
        self.node_manager = NodeManager()
        self.beam_manager = BeamManager(self.node_manager)

        self.plates: Dict[int, Plate] = {}
        self.plate_counter = 10000

        # 快捷访问
        self.points = self.node_manager.points
        self.beams = self.beam_manager.beams

    def generate(self):
        engine = LayoutEngine(self.params)
        xs, ys, zs = engine.sorted_x, engine.sorted_y, engine.sorted_z

        # --- 1. 生成 X 方向梁 ---
        for y in ys:
            for z in zs:
                for i in range(len(xs) - 1):
                    p1 = np.array([xs[i], y, z])
                    p2 = np.array([xs[i + 1], y, z])

                    b_type, face, is_virt = engine.determine_beam(p1, p2, "x")
                    if b_type:
                        self._add_beam(p1, p2, b_type, face, is_virt)

        # --- 2. 生成 Y 方向梁 ---
        for x in xs:
            for z in zs:
                for i in range(len(ys) - 1):
                    p1 = np.array([x, ys[i], z])
                    p2 = np.array([x, ys[i + 1], z])

                    b_type, face, is_virt = engine.determine_beam(p1, p2, "y")
                    if b_type:
                        self._add_beam(p1, p2, b_type, face, is_virt)

        # --- 3. 生成 Z 方向梁 ---
        for x in xs:
            for y in ys:
                for i in range(len(zs) - 1):
                    p1 = np.array([x, y, zs[i]])
                    p2 = np.array([x, y, zs[i + 1]])

                    b_type, face, is_virt = engine.determine_beam(p1, p2, "z")
                    if b_type:
                        self._add_beam(p1, p2, b_type, face, is_virt)

        # --- 4. 生成板 ---
        self._generate_plates_from_grid(engine)
        self._cleanup_orphans()

    def _add_beam(self, p1_arr, p2_arr, b_type, face, is_virtual_override=False):
        n1 = self.node_manager.get_or_create(*p1_arr)
        n2 = self.node_manager.get_or_create(*p2_arr)

        # 允许外部 override 虚拟属性，或者梁本身类型决定（例如以后 BRACE 想设为虚拟）
        is_virtual = is_virtual_override

        sec_prop = self._get_section(b_type)
        self.beam_manager.add_or_update_beam(n1, n2, b_type, sec_prop, is_virtual, face)

    def _generate_plates_from_grid(self, engine: LayoutEngine):
        """生成所有板单元"""

        # 定义通用扫描函数
        def scan_plane(u_vals, v_vals, w_vals, plane_type):
            """
            u_vals: 平面内的横向坐标列表 (如 x)
            v_vals: 平面内的纵向坐标列表 (如 y)
            w_vals: 平面法向坐标列表 (如 z)
            plane_type: 'XY', 'YZ', or 'XZ'
            """
            for w in w_vals:
                for i in range(len(u_vals) - 1):
                    for j in range(len(v_vals) - 1):
                        u1, u2 = u_vals[i], u_vals[i + 1]
                        v1, v2 = v_vals[j], v_vals[j + 1]

                        # 计算中心点和角点
                        uc, vc = (u1 + u2) / 2, (v1 + v2) / 2

                        # 坐标映射回 (x, y, z)
                        if plane_type == "XY":
                            cx, cy, cz = uc, vc, w
                            # 构造板的四个角点 (用于面积/法向)
                            pts = [(u1, v1, w), (u2, v1, w), (u2, v2, w), (u1, v2, w)]
                        elif plane_type == "YZ":
                            cx, cy, cz = w, uc, vc  # w is x
                            pts = [(w, u1, v1), (w, u1, v2), (w, u2, v2), (w, u2, v1)]
                        elif plane_type == "XZ":
                            cx, cy, cz = uc, w, vc  # w is y
                            pts = [(u1, w, v1), (u2, w, v1), (u2, w, v2), (u1, w, v2)]

                        # 1. 检查是否在连通口 (Opening)
                        if engine.is_in_opening(cx, cy, cz):
                            continue

                        # 2. 检查区域包含性
                        active_zones = [zn for zn in engine.zones if zn.contains(cx, cy, cz)]
                        # 过滤掉 Hollow (设备间)
                        active_real_zones = [zn for zn in active_zones if not zn.is_hollow]
                        if not active_real_zones:
                            continue

                        # 3. 检查是否在表面
                        is_surface = any(zn.on_surface(cx, cy, cz) for zn in active_real_zones)

                        if is_surface:
                            face = active_real_zones[0].get_face(cx, cy, cz)
                            # 创建板
                            nids = [self.node_manager.get_or_create(*p) for p in pts]
                            self.add_plate(nids, face)

        # 调用通用函数
        xs, ys, zs = engine.sorted_x, engine.sorted_y, engine.sorted_z

        scan_plane(xs, ys, zs, "XY")  # Top / Bottom
        scan_plane(ys, zs, xs, "YZ")  # Front / Back
        scan_plane(xs, zs, ys, "XZ")  # Left / Right

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

    def _get_section(self, b_type: BeamType) -> SectionProperties:
        sec_type = BEAM_TYPE_MAP.get(b_type, SectionType.BOX)
        idx_map = {
            BeamType.FRAME: self.params.frame_section_index,
            BeamType.MAIN: self.params.main_beam_section_index,
            BeamType.SECONDARY: self.params.secondary_beam_section_index,
            BeamType.BRACE: self.params.brace_section_index,
        }
        return self.section_lib.get_section(sec_type, idx_map.get(b_type, 0))

    def _cleanup_orphans(self):
        used = set()
        for b in self.beams.values():
            used.add(b.start_node)
            used.add(b.end_node)
        for p in self.plates.values():
            used.update(p.nodes)

        to_del = [k for k in self.points if k not in used]
        for k in to_del:
            del self.points[k]

    def calculate_mass_kg(self):
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
        length=8.0,  # 10米长
        width=3.6,  # 3米宽
        height=3.6,  # 3米高
        main_beam_spacing=0.9,  # 主梁每1.5米一道
        num_secondary_width=2,  # 宽度方向分3段 (2条中间次梁)
        num_secondary_height=3,  # 高度方向分3段
        add_support_beam=False,  # 添加支撑梁
        frame_section_index=1,
        main_beam_section_index=16,
        secondary_beam_section_index=17,
        brace_section_index=18,
    )

    model = StructuralModel(params, SectionLibrary())
    model.generate()

    # mesh_generator = MeshGenerator(model, 0.5)
    # model = mesh_generator.generate()

    visualizer = StructureVisualizer(model)
    visualizer.show_face(ViewType.FULL_3D)
    # visualizer.show_all_views(grid=False)
