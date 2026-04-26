import math
from typing import Dict, List, Optional

import numpy as np
import openseespy.opensees as ops

from src.analysis.results import AnalysisResults, ModalAnalysisResults
from src.analysis.unit import (
    E_STEEL_MPA,
    G_M_S2,
    G_STEEL_MPA,
    M2_TO_MM2,
    M4_TO_MM4,
    M_TO_MM,
    N_M_TO_N_MM,
    NU_STEEL,
    RHO_STEEL_KG_M3,
    RHO_STEEL_TONNE_MM3,
)
from src.structure.layout.params import StructuralParameters
from src.structure.section_library import SectionLibrary
from src.structure.structural_model import Beam, StructuralModel
from src.utils.constants import KPA_TO_MPA, LOAD_DEAD_BOTTOM_KPA, LOAD_LIVE_BOTTOM_KPA
from src.utils.logger import logger


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

    def _calculate_node_mass(self) -> Dict[int, float]:
        # Nodal Masses (including beam and plate contributions)
        nodal_masses_tonne: Dict[int, float] = {nid: 0 for nid in self.model.points.keys()}

        # Add beam masses
        for beam_id, beam_obj in self.model.beams.items():
            section_props = beam_obj.section

            start_node_model = self.model.points[beam_obj.start_node]
            end_node_model = self.model.points[beam_obj.end_node]
            length_m = start_node_model.distance_to(end_node_model)
            length_mm = length_m * M_TO_MM

            area_mm2 = section_props.area * M2_TO_MM2
            beam_mass_tonne = area_mm2 * length_mm * RHO_STEEL_TONNE_MM3
            mass_per_node = beam_mass_tonne / 2.0

            start_node_id = beam_obj.start_node
            end_node_id = beam_obj.end_node

            nodal_masses_tonne[start_node_id] += mass_per_node
            nodal_masses_tonne[end_node_id] += mass_per_node

        # Add plate masses
        for plate_id, plate_obj in self.model.plates.items():
            # Calculate plate mass
            plate_area_m2 = plate_obj.area
            plate_thickness_m, plate_density_kg_m3 = self._get_plate_mass_properties(plate_obj.panel_type)
            plate_volume_m3 = plate_area_m2 * plate_thickness_m
            plate_mass_kg = plate_volume_m3 * plate_density_kg_m3
            plate_mass_tonne = plate_mass_kg / 1000.0

            # Distribute mass equally to 4 nodes
            mass_per_node = plate_mass_tonne / len(plate_obj.nodes)
            for node_id in plate_obj.nodes:
                nodal_masses_tonne[node_id] += mass_per_node

        return nodal_masses_tonne

    def _get_plate_mass_properties(self, panel_type: str) -> tuple[float, float]:
        if panel_type == "opening":
            return self.model.params.opening_panel.thickness, self.model.params.opening_panel.density_kg_m3
        return self.model.params.plate_thickness, RHO_STEEL_KG_M3

    def _get_plate_section_tag(self, panel_type: str) -> int:
        return 2001 if panel_type == "opening" else 2000

    def _define_plate_sections(self):
        ops.nDMaterial("ElasticIsotropic", 1000, E_STEEL_MPA, NU_STEEL, RHO_STEEL_TONNE_MM3)
        plate_thk_mm = self.model.params.plate_thickness * M_TO_MM
        ops.section("ElasticMembranePlateSection", 2000, E_STEEL_MPA, NU_STEEL, plate_thk_mm)

        opening_density_tonne_mm3 = self.model.params.opening_panel.density_kg_m3 / 1e12
        ops.nDMaterial(
            "ElasticIsotropic",
            1001,
            self.model.params.opening_panel.elastic_modulus_mpa,
            self.model.params.opening_panel.poisson_ratio,
            opening_density_tonne_mm3,
        )
        opening_plate_thk_mm = self.model.params.opening_panel.thickness * M_TO_MM
        ops.section(
            "ElasticMembranePlateSection",
            2001,
            self.model.params.opening_panel.elastic_modulus_mpa,
            self.model.params.opening_panel.poisson_ratio,
            opening_plate_thk_mm,
        )

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
        self._define_plate_sections()

        # 3. Elements - Beams
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

        # 4. Elements - Plates (ShellDKGQ)
        for pid, plate in self.model.plates.items():
            ops.element("ShellDKGQ", int(pid), *plate.nodes, self._get_plate_section_tag(plate.panel_type))

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
        else:
            raise ValueError(
                f"Cannot determine vecxz for beam on face {on_face}, with direction {x_local}, p1: {p1}, p2: {p2}"
            )

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
        """根据面板所在的 'face' 属性施加法向内压"""
        face_dirs = {
            "top": [0, 0, 1],
            "bottom": [0, 0, -1],
            "left": [0, -1, 0],
            "right": [0, 1, 0],
            "front": [1, 0, 0],
            "back": [-1, 0, 0],
        }

        for pid, plate in self.model.plates.items():
            direction = face_dirs[plate.on_face]
            area_mm2 = plate.area * M2_TO_MM2
            total_force = pressure_mpa * area_mm2
            nodal_force = total_force / len(plate.nodes)

            fx, fy, fz = np.array(direction) * nodal_force
            for nid in plate.nodes:
                ops.load(nid, fx, fy, fz, 0, 0, 0)

    def _apply_dead_load(self):
        """施加恒载 (DL)"""
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
            plate_thk_m, plate_density_kg_m3 = self._get_plate_mass_properties(plate.panel_type)
            mass_kg = plate.area * plate_thk_m * plate_density_kg_m3
            force_n = mass_kg * G_M_S2
            fz = -force_n / len(plate.nodes)
            for nid in plate.nodes:
                ops.load(nid, 0, 0, fz, 0, 0, 0)

        # C. 底部恒载 (5 kN/m2)
        self._apply_distributed_plate_load(self.bottom_plate_ids, LOAD_DEAD_BOTTOM_KPA * KPA_TO_MPA)

    def _apply_live_load(self):
        """施加活载 (LL)"""
        self._apply_distributed_plate_load(self.bottom_plate_ids, LOAD_LIVE_BOTTOM_KPA * KPA_TO_MPA)

    def _apply_pressure_load(self, pressure_pa: float):
        """施加内压 (PL)"""
        self._apply_pressure_load_by_face(pressure_pa / 1e6)  # Pa -> MPa

    def run_load_combination(self, factors: Dict[str, float], pressure_pa: float = 0, odb_tag: int = 1):
        """
        修正后的荷载组合计算
        """
        combo_name = "+".join([f"{v}{k}" for k, v in factors.items()])
        logger.info(f"Running Load Combination: {combo_name}")

        self._setup_model_geometry()  # 确保模型几何已建立/重置

        # =========================================================
        # 修正核心：通过 TimeSeries 的 -factor 参数控制系数
        # =========================================================

        # 1. 恒载 (DL)
        # 创建一个专属于 DL 的 TimeSeries，直接带入系数
        dl_factor = factors.get("DL", 0.0)
        ops.timeSeries("Linear", 101, "-factor", dl_factor)
        ops.pattern("Plain", 201, 101)  # 使用 tag 101
        self._apply_dead_load()

        # 2. 活载 (LL)
        # 创建一个专属于 LL 的 TimeSeries，带入 LL 的系数
        ll_factor = factors.get("LL", 0.0)
        ops.timeSeries("Linear", 102, "-factor", ll_factor)
        ops.pattern("Plain", 202, 102)  # 使用 tag 102
        self._apply_live_load()

        # 3. 内压 (PL)
        pl_factor = factors.get("PL", 0.0)
        if pressure_pa > 0 and pl_factor != 0:
            # 创建专属于 PL 的 TimeSeries
            ops.timeSeries("Linear", 103, "-factor", pl_factor)
            ops.pattern("Plain", 203, 103)  # 使用 tag 103
            self._apply_pressure_load(pressure_pa)

        # =========================================================
        # 分析设置
        # =========================================================
        ops.system("BandSPD")  # 或 BandGeneral, UmfPack
        ops.numberer("RCM")
        ops.constraints("Plain")

        # 这里的 LoadControl 只负责让时间往前走 1.0
        # 此时：
        # DL 的实际倍数 = Time(1.0) * Factor(1.3) = 1.3
        # LL 的实际倍数 = Time(1.0) * Factor(1.5) = 1.5
        ops.integrator("LoadControl", 1.0)

        ops.algorithm("Linear")
        ops.analysis("Static")

        ok = ops.analyze(1)

        if ok != 0:
            logger.error(f"Analysis Failed for load combination {combo_name}")
            return None

        return self._extract_results()

    def _extract_results(self) -> AnalysisResults:
        """从 OpenSees 模型中提取结果"""
        res = AnalysisResults()

        # 先触发支座反力计算。ShellDKGQ 的应力响应在此之后才会被正确更新。
        ops.reactions()

        # (1) 提取位移
        for nid in self.node_map_mm:
            res.displacements[nid] = np.array(ops.nodeDisp(nid))

        for nid in self.support_node_ids:
            res.reactions[nid] = np.array(ops.nodeReaction(nid))

        # (2) 提取梁单元内力
        for bid in self.model.beams:
            try:
                f = ops.eleResponse(bid, "localForce")
                if f:
                    res.beam_forces[bid] = np.array(f)
            except:
                pass

        # (4) 提取板单元内力
        for pid in self.model.plates:
            try:
                _ = ops.eleResponse(pid, "force")  # 先调用一次 force 来确保内部状态更新
                s = ops.eleResponse(pid, "stresses")
                if s:
                    res.plate_stresses[pid] = np.array(s)
            except:
                pass

        return res

    def run_modal_case(self, num_modes: int, odb_tag: int = 1) -> Optional[ModalAnalysisResults]:
        """Performs modal analysis."""
        logger.info(f"Running Modal Analysis for {num_modes} modes (odb_tag={odb_tag})")
        self._setup_model_geometry()
        nodal_masses = self._calculate_node_mass()
        for node_id, mass in nodal_masses.items():
            ops.mass(node_id, mass, mass, mass, 0, 0, 0)

        eigenvalues_omega_sq = ops.eigen(num_modes)

        frequencies_hz = [np.sqrt(val) / (2 * np.pi) for val in eigenvalues_omega_sq]
        periods_s = [1 / f for f in frequencies_hz if f > 0]
        mode_shapes = {}
        for mode in range(1, len(frequencies_hz) + 1):
            mode_shapes[mode] = {}
            for node_id in self.model.points:
                try:
                    mode_shapes[mode][node_id] = np.array(ops.nodeEigenvector(node_id, mode))
                except Exception:
                    continue
        return ModalAnalysisResults(
            frequencies_hz=frequencies_hz, periods_s=periods_s, mode_shapes=mode_shapes
        )


# ================= 主流程调用 =================
if __name__ == "__main__":
    from src.analysis.post_processor import ResultProcessor
    from src.structure.layout.params import EngineeringLayoutParameters, WindowParameters
    from src.structure.mesh_generator import MeshGenerator
    from src.utils.util import Timer

    pressure = 30_000  # 内压 30kPa
    layout = 1
    if layout == 1:
        # 配置参数
        params = StructuralParameters(
            length=8.9,  # 10米长
            width=3.7,  # 3米宽
            height=3.3,  # 3米高
            main_beam_spacing=0.95,  # 主梁每1.5米一道
            num_secondary_width=3,  # 宽度方向分3段 (2条中间次梁)
            num_secondary_height=3,  # 高度方向分3段
            add_support_beam=True,  # 添加支撑梁
            frame_section_index=2,
            main_beam_section_index=8,
            secondary_beam_section_index=7,
            brace_section_index=14,
        )
        model = StructuralModel(params, SectionLibrary("src/data/steel_sections_large.json"))
        model.generate()
    else:
        # 配置参数
        windows = WindowParameters(
            front_spans=[],  # 前墙第1段有窗
            back_spans=[],
            right_spans=[1, 2],  # 右墙第2段有窗
        )
        params = EngineeringLayoutParameters(
            length=8.9,  # 10米长
            width=3.7,  # 3米宽
            height=3.3,  # 3米高
            main_beam_spacing=0.95,  # 主梁每1.5米一道
            add_support_beam=True,  # 添加支撑梁
            support_beam_max_spacing=0.75,
            frame_section_index=2,
            main_beam_section_index=8,
            secondary_beam_section_index=7,
            brace_section_index=14,
            window=windows,
        )
        model = StructuralModel(params, SectionLibrary())
        model.generate()

    print("Total mass：", model.calculate_mass_kg(), "kg")

    mesh_generator = MeshGenerator(model, 0.3)
    model = mesh_generator.generate()

    with Timer("OpenSees Analysis Cost: "):
        solver = OpenSeesSolver(model)
        combo_strength = {"DL": 1.3, "LL": 1.5, "PL": 1.25}
        combo_service = {"DL": 1.0, "LL": 1.0, "PL": 1.0}
        res_strength = solver.run_load_combination(combo_strength, pressure_pa=pressure, odb_tag=91)
        strength_check = ResultProcessor(model).run_gb50017_check(res_strength)
        res_service = solver.run_load_combination(combo_service, pressure_pa=pressure, odb_tag=52)
        disp_check = ResultProcessor(model).run_gb50017_check(res_service)

    print(f"Max Stress Ratio: {strength_check.max_stress_ratio:.3f}")
    print(f"Max Stress: {strength_check.max_stress:.3f} MPa")
    print(f"Max Stress Pos: {strength_check.max_stress_pos}")
    print(f"Max Deflection Ratio: {disp_check.max_deflection_ratio:.3f}")
    print(f"Max Deflection: {disp_check.max_deflection:.3f} mm")
    print(f"Max Deflection Pos: {disp_check.max_deflection_pos}")
    print(model.points[disp_check.max_deflection_pos].coordinates())

    # with Timer("Modal Analysis Cost: "):
    #     frequencies, periods = solver.run_modal_case(6)
    # for i, (f, T) in enumerate(zip(frequencies, periods), start=1):
    #     print(f"Mode {i}: Frequency = {f:.3f} Hz, Period = {T:.4f} s")
