import json
import math
import os
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import openseespy.opensees as ops

# 假设这些类依然存在于你的项目中
from src.analysis.fem_utils import calc_basic_stresses
from src.analysis.unit import (E_STEEL_MPA, G_M_S2, G_STEEL_MPA, M2_TO_MM2,
                               M4_TO_MM4, M_TO_MM, N_M_TO_N_MM, NU_STEEL,
                               RHO_STEEL_KG_M3, RHO_STEEL_TONNE_MM3)
from src.structure.section_library import SectionLibrary
from src.structure.structural_model import (Beam, StructuralModel,
                                            StructuralParameters)
from src.utils.logger import logger

# ================= 配置常量 =================
# 荷载输入 (kN/m2 -> N/mm2 = MPa)
LOAD_DEAD_BOTTOM_KPA = 5.0  # 底部恒载 5kN/m2
LOAD_LIVE_BOTTOM_KPA = 2.0  # 底部活载 2kN/m2
KPA_TO_MPA = 1e-3  # 单位转换系数


class AnalysisResults:
    """数据类，用于存储单次工况的原始结果"""

    def __init__(self):
        self.displacements: Dict[int, np.ndarray] = {}  # {node_id: [dx, dy, dz, rx, ry, rz]}
        self.beam_forces: Dict[int, np.ndarray] = {}  # {beam_id: array}
        self.plate_stresses: Dict[int, np.ndarray] = {}  # {plate_id: array}
        self.reactions: Dict[int, np.ndarray] = {}  # {node_id, array}

    def __add__(self, other):
        """支持结果直接相加: res = res1 + res2"""
        if not isinstance(other, AnalysisResults):
            return None
        new_res = AnalysisResults()
        # 向量化相加逻辑 (示例简化版，实际需处理键值对齐)
        common_nodes = set(self.displacements.keys()) & set(other.displacements.keys())
        for k in common_nodes:
            new_res.displacements[k] = self.displacements[k] + other.displacements[k]

        common_plates = set(self.plate_stresses.keys()) & set(other.plate_stresses.keys())
        for k in common_plates:
            # 处理 None 的情况
            v1 = self.plate_stresses.get(k)
            v2 = other.plate_stresses.get(k)
            if v1 is not None and v2 is not None:
                new_res.plate_stresses[k] = v1 + v2
        return new_res

    def scale(self, factor: float):
        """支持结果乘系数: res = res * 1.5"""
        new_res = AnalysisResults()
        for k, v in self.displacements.items():
            new_res.displacements[k] = v * factor
        for k, v in self.plate_stresses.items():
            if v is not None:
                new_res.plate_stresses[k] = v * factor
        return new_res


class OpenSeesSolver:
    """
    单一职责：负责将 StructuralModel 转换为 OpenSees 模型，
    施加指定工况的荷载，并返回原始结果。
    """

    def __init__(self, model: StructuralModel):
        self.model = model
        self.node_map_mm: Dict[int, List[float]] = {}  # 缓存毫米单位坐标

        # 预计算：识别底部板 (用于施加 5kN/m2 和 2kN/m2)
        self.bottom_plate_ids = self._identify_bottom_plates()
        self.support_node_ids = self._identify_support_nodes()

    def _identify_bottom_plates(self) -> List[int]:
        """识别位于 z ≈ 0 (或模型最低处) 的板"""
        return [pid for pid, plate in self.model.plates.items() if plate.on_face == "bottom"]

    def _identify_support_nodes(self) -> List[int]:
        """
        确定支座节点逻辑：
        1. 必须包含底部4个角点。
        2. 若长边 > 10m，需在长边中部均匀布置支座，间距 <= 2.5m。
        该函数会将结果存储在 self.support_node_ids 中并返回。
        """

        # 1. 筛选底部节点 (容差 0.1m)
        all_z = [p.z for p in self.model.points.values()]
        min_z = min(all_z)
        bottom_nodes = {nid: p for nid, p in self.model.points.items() if abs(p.z - min_z) < 0.001}

        if not bottom_nodes:
            logger.warning("No bottom nodes found!")
            return []

        # 2. 获取几何边界
        coords = np.array([[p.x, p.y] for p in bottom_nodes.values()])
        min_x, max_x = np.min(coords[:, 0]), np.max(coords[:, 0])
        min_y, max_y = np.min(coords[:, 1]), np.max(coords[:, 1])

        supports = set()

        # 辅助函数：寻找坐标最接近 (target_x, target_y) 的节点
        def find_nearest_node(tx, ty, node_pool):
            best_nid = None
            min_dist = float("inf")
            for nid, p in node_pool.items():
                dist = (p.x - tx) ** 2 + (p.y - ty) ** 2
                if dist < min_dist:
                    min_dist = dist
                    best_nid = nid
            # 如果距离太远（例如超过0.5m），说明此处可能没有节点（模型网格未对齐），需警示
            if min_dist > 0.5**2:
                logger.warning(f"Warning: No node found near support target ({tx:.2f}, {ty:.2f})")
            return best_nid

        # 3. 添加4个角点
        corners = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
        for cx, cy in corners:
            nid = find_nearest_node(cx, cy, bottom_nodes)
            if nid is not None:
                supports.add(nid)

        # 4. 处理长边加密逻辑
        long_edge_len = max_y - min_y  # 长边沿y轴方向

        if long_edge_len > 10.0:
            # 计算分段数，确保间距 <= 2.5m
            # 例如 12m: ceil(12/2.5) = 5段 -> 间距 2.4m
            num_segments = math.ceil(long_edge_len / 2.5)
            step = long_edge_len / num_segments

            # 生成中间点坐标 (排除首尾 0 和 long_edge_len，因为角点已加)
            for i in range(1, num_segments):
                dist = min_y + i * step

                targets = [(min_x, dist), (max_x, dist)]

                for tx, ty in targets:
                    nid = find_nearest_node(tx, ty, bottom_nodes)
                    if nid is not None:
                        supports.add(nid)

        return list[int](supports)

    def _setup_model_geometry(self):
        """建立几何模型：节点、材料、截面、单元"""
        ops.wipe()
        ops.model("basic", "-ndm", 3, "-ndf", 6)

        # 1. Nodes
        for nid, point in self.model.points.items():
            coords = point.coordinates() * M_TO_MM
            ops.node(nid, *coords)
            self.node_map_mm[nid] = coords

        # 2. Materials & Sections
        # Steel Material
        ops.nDMaterial("ElasticIsotropic", 1000, E_STEEL_MPA, NU_STEEL, RHO_STEEL_TONNE_MM3)
        # Plate Section
        plate_thk_mm = self.model.params.plate_thickness * M_TO_MM
        ops.section("ElasticMembranePlateSection", 2000, E_STEEL_MPA, NU_STEEL, plate_thk_mm)

        # 4. Elements - Beams
        transf_tag = 1
        for bid, beam in self.model.beams.items():
            if beam.is_virtual:
                continue
            p1 = np.array(self.node_map_mm[beam.start_node])
            p2 = np.array(self.node_map_mm[beam.end_node])

            # 计算局部坐标向量 (简化版，复用你原有的逻辑概念)
            vecxz = self._calculate_vecxz(p1, p2, beam.on_face)
            ops.geomTransf("Linear", transf_tag, *vecxz)

            # 获取截面属性
            sec = beam.section

            # 创建单元
            ops.element(
                "elasticBeamColumn",
                bid,
                beam.start_node,
                beam.end_node,
                sec.area * M2_TO_MM2,
                E_STEEL_MPA,
                G_STEEL_MPA,
                sec.Iz * M4_TO_MM4,
                sec.Iy * M4_TO_MM4,
                sec.Ix * M4_TO_MM4,
                transf_tag,
            )
            transf_tag += 1

        # 3. Elements - Plates (ShellDKGQ)
        for pid, plate in self.model.plates.items():
            # print(pid)
            # print(plate.nodes)
            # if pid not in self.model.points.keys():
            #     print("node not in key")
            ops.element("ShellDKGQ", int(pid + 10000), *plate.nodes, 2000)

        # 5. Boundary Conditions (Pin 底部节点)
        for nid in self.support_node_ids:
            ops.fix(nid, 1, 1, 1, 0, 0, 0)

    def _calculate_vecxz(self, p1: np.ndarray, p2: np.ndarray, on_face: str) -> np.ndarray:
        """计算梁的局部坐标方向向量 (简化实现)"""
        dx_vec = p2 - p1
        length = np.linalg.norm(dx_vec)
        if length < 1e-6:  # Zero length beam
            return np.array([0, 0, 1])

        x_local = dx_vec / length

        if np.isclose(abs(x_local[0]), 1.0) and on_face in ["top", "bottom"]:
            vecxz = np.array([0.0, 1.0, 0.0])
        elif np.isclose(abs(x_local[0]), 1.0) and on_face in ["left", "right"]:
            vecxz = np.array([0.0, 0.0, 1.0])
        elif np.isclose(abs(x_local[1]), 1.0) and on_face in ["top", "bottom"]:
            vecxz = np.array([1.0, 0.0, 0.0])
        elif np.isclose(abs(x_local[1]), 1.0) and on_face in ["front", "back"]:
            vecxz = np.array([0.0, 0.0, 1.0])
        elif np.isclose(abs(x_local[2]), 1.0) and on_face in ["front", "back"]:
            vecxz = np.array([0.0, 1.0, 0.0])
        elif np.isclose(abs(x_local[2]), 1.0) and on_face in ["left", "right"]:
            vecxz = np.array([1.0, 0.0, 0.0])

        return vecxz

    def _apply_distributed_plate_load(
        self, plate_ids: List[int], load_mpa: float, direction_local_z: bool = True
    ):
        """将面荷载 (MPa) 转化为节点力施加"""
        load_dir_vec = (
            np.array([0, 0, -1.0]) if direction_local_z else np.array([0, 0, 1.0])
        )  # 简化：默认向下

        for pid in plate_ids:
            plate = self.model.plates[pid]
            # 这里调用 plate 对象的方法计算面积 (m2) 并转为 mm2
            area_mm2 = plate.area * M2_TO_MM2

            total_force = load_mpa * area_mm2
            num_nodes = len(plate.nodes)
            nodal_force_val = total_force / num_nodes

            # 施加力
            fx, fy, fz = load_dir_vec * nodal_force_val
            for nid in plate.nodes:
                ops.load(nid, fx, fy, fz, 0, 0, 0)

    def _apply_pressure_load_by_face(self, pressure_mpa: float):
        """根据面板所在的 'face' 属性施加法向气压"""
        face_dirs = {
            "top": [0, 0, 1],
            "bottom": [0, 0, -1],
            "left": [0, -1, 0],
            "right": [0, 1, 0],
            "front": [-1, 0, 0],
            "back": [1, 0, 0],
        }

        for pid, plate in self.model.plates.items():
            direction = face_dirs.get(plate.on_face, [0, 0, 1])
            area_mm2 = plate.area(self.model.points) * M2_TO_MM2
            total_force = pressure_mpa * area_mm2
            nodal_force = total_force / len(plate.nodes)

            fx, fy, fz = np.array(direction) * nodal_force
            for nid in plate.nodes:
                ops.load(nid, fx, fy, fz, 0, 0, 0)

    def run_case(self, case_type: Literal["DL", "LL", "PL"], pressure_pa: float = 0) -> AnalysisResults:
        """
        核心对外接口：运行指定工况
        """
        logger.info(f"Running Analysis Case: {case_type}")

        # 1. 重建模型 (确保纯净)
        self._setup_model_geometry()

        # 2. 定义荷载模式
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)

        # 3. 施加荷载
        if case_type == "DL":
            # A. 结构自重 (梁)
            for bid, beam in self.model.beams.items():
                if beam.is_virtual:
                    continue
                sec = beam.section
                # w = A * rho * g (N/mm)
                w = (sec.area * RHO_STEEL_KG_M3 * G_M_S2) * N_M_TO_N_MM
                ops.eleLoad("-ele", bid, "-type", "-beamUniform", 0, 0, -w)  # Global Z down

            # B. 结构自重 (板) - 分配到节点
            for pid, plate in self.model.plates.items():
                plate_thk_m = self.model.params.plate_thickness
                mass_kg = plate.area * plate_thk_m * RHO_STEEL_KG_M3
                force_n = mass_kg * G_M_S2
                fz = -force_n / len(plate.nodes)
                for nid in plate.nodes:
                    ops.load(nid, 0, 0, fz, 0, 0, 0)

            # C. 底部恒载 (5 kN/m2)
            self._apply_distributed_plate_load(self.bottom_plate_ids, LOAD_DEAD_BOTTOM_KPA * KPA_TO_MPA)

        elif case_type == "LL":
            # 底部活载 (2 kN/m2)
            self._apply_distributed_plate_load(self.bottom_plate_ids, LOAD_LIVE_BOTTOM_KPA * KPA_TO_MPA)

        elif case_type == "PL":
            # 气压荷载
            self._apply_pressure_load_by_face(pressure_pa / 1e6)  # Pa -> MPa

        # 4. 分析求解
        ops.system("BandSPD")
        ops.numberer("RCM")
        ops.constraints("Plain")
        ops.integrator("LoadControl", 1.0)
        ops.algorithm("Linear")
        ops.analysis("Static")

        ok = ops.analyze(1)
        if ok != 0:
            logger.error(f"Analysis Failed for case {case_type}")
            return None

        # 5. 提取结果
        res = AnalysisResults()

        # 提取位移
        for nid in self.node_map_mm:
            res.displacements[nid] = np.array(ops.nodeDisp(nid))

        # 提取板应力
        for pid in self.model.plates:
            try:
                # 只有成功运行且单元支持时才提取
                s = ops.eleResponse(pid, "stresses")
                if s:
                    res.plate_stresses[pid] = np.array(s)
            except:
                pass

        # 提取支座反力
        ops.reactions()
        for nid in self.support_node_ids:
            res.reactions[nid] = np.array(ops.nodeReaction(nid))

        return res


class ResultCombiner:
    """负责结果的线性组合"""

    @staticmethod
    def get_max_displacement_combo(res_dl: AnalysisResults, res_ll: AnalysisResults, res_pl: AnalysisResults):
        """组合: 1.0 DL + 1.0 LL + 1.0 PL"""
        # 注意：AnalysisResults 类需要实现 __add__ 和 scale
        # 这里用一种更 Pythonic 的写法
        return res_dl.scale(1.0) + res_ll.scale(1.0) + res_pl.scale(1.0)

    @staticmethod
    def get_max_stress_combo(res_dl: AnalysisResults, res_ll: AnalysisResults, res_pl: AnalysisResults):
        """组合: 1.3 DL + 1.5 LL + 1.25 PL"""
        return res_dl.scale(1.3) + res_ll.scale(1.5) + res_pl.scale(1.25)

    @staticmethod
    def extract_max_stats(combined_res: AnalysisResults):
        """从组合结果中提取最大值统计"""
        max_disp = 0
        for v in combined_res.displacements.values():
            d = np.linalg.norm(v[:3])  # 平动位移
            if d > max_disp:
                max_disp = d

        # 应力提取逻辑需要根据 Shell 单元的应力输出格式解析
        # 这里仅做框架示例
        max_stress = 0
        # ... logic to parse Mises or Principal stress from tensor ...

        return {"max_disp_mm": max_disp, "max_stress_mpa": max_stress}


# ================= 主流程调用 =================


def run_full_design_check(model: StructuralModel, internal_pressure_pa: float = 30000):
    solver = OpenSeesSolver(model)

    # 1. 运行三个独立工况
    # DL: 自重 + 底部恒载 5kN/m2
    res_dl = solver.run_case("DL")

    # LL: 底部活载 2kN/m2
    res_ll = solver.run_case("LL")

    # PL: 气压荷载
    res_pl = solver.run_case("PL", pressure_pa=internal_pressure_pa)

    # 2. 结果组合
    combiner = ResultCombiner()

    # 工况 1: 最大位移 (1.0DL + 1.0LL + 1.0PL)
    res_disp_check = combiner.get_max_displacement_combo(res_dl, res_ll, res_pl)
    stats_disp = combiner.extract_max_stats(res_disp_check)
    logger.info(f"Check 1 (Max Displacement): {stats_disp['max_disp_mm']:.2f} mm")

    # 工况 2: 最大应力 (1.3DL + 1.5LL + 1.25PL)
    res_stress_check = combiner.get_max_stress_combo(res_dl, res_ll, res_pl)
    stats_stress = combiner.extract_max_stats(res_stress_check)
    logger.info(f"Check 2 (Max Stress Load): {stats_stress['max_stress_mpa']:.2f} MPa")

    # 3. (可选) 保存详细结果到 JSON
    # ...


if __name__ == "__main__":
    from src.structure.mesh_generator import MeshGenerator

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

    mesh_generator = MeshGenerator(model, 0.5)
    model = mesh_generator.generate()

    solver = OpenSeesSolver(model)
    res = solver.run_case("DL")
    print(res.displacements[0])
