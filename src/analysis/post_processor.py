from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.analysis.metrics import (
    DESIGN_STRENGTH_F,
    DESIGN_SHEAR_FV,
    beam_stress_metric,
    plate_stress_metric,
)
from src.analysis.opensees_analysis import OpenSeesSolver
from src.analysis.results import AnalysisResults
from src.analysis.unit import M_TO_MM
from src.structure.layout.params import StructuralParameters
from src.structure.section_library import SectionLibrary
from src.structure.structural_model import BeamType, StructuralModel

@dataclass
class DesignCheckResult:
    max_stress: float  # 最大应力值 (N/mm2)
    max_stress_ratio: float  # 最大应力比 (calculated / limit)
    max_stress_pos: Tuple[int, str]  # (元素ID, "Beam" or "Plate")
    max_deflection: float  # 最大挠度值 (mm)
    max_deflection_ratio: float  # 最大挠度比 (calculated / limit)
    max_deflection_pos: int  # 节点ID
    critical_beam_id: int  # 最危险梁ID
    critical_plate_id: int  # 最危险板ID
    failure_mode: str  # "Bending", "Shear", "PlateMises", "Deflection"


class ResultProcessor:
    """后处理器：将FEM结果转换为工程指标并进行校核"""

    def __init__(self, model: StructuralModel):
        self.model = model

    def run_gb50017_check(self, results: AnalysisResults) -> DesignCheckResult:
        """执行 GB50017 校核主流程"""

        # 1. 梁应力校核
        beam_max_ratio, beam_max_stress, crit_beam, beam_mode = self._check_beams(results)

        # 2. 板应力校核 (Mises)
        plate_max_ratio, plate_max_stress, crit_plate = self._check_plates(results)

        # 3. 挠度校核
        disp_max_ratio, max_deflection, max_deflection_pos = self._check_deflection(results)

        # 汇总最不利情况
        max_stress_ratio = max(beam_max_ratio, plate_max_ratio)
        max_stress = max(beam_max_stress, plate_max_stress)

        mode = beam_mode
        crit_elem = crit_beam
        max_stress_pos = ("Beam", crit_beam)
        if plate_max_ratio > beam_max_ratio:
            mode = "PlateMises"
            crit_elem = crit_plate
            max_stress_pos = ("Plate", crit_plate)

        return DesignCheckResult(
            max_stress=max_stress,
            max_stress_ratio=max_stress_ratio,
            max_stress_pos=max_stress_pos,
            max_deflection=max_deflection,
            max_deflection_ratio=disp_max_ratio,
            max_deflection_pos=max_deflection_pos,
            critical_beam_id=crit_elem if mode != "PlateMises" else -1,
            critical_plate_id=crit_elem if mode == "PlateMises" else -1,
            failure_mode=mode,
        )

    def _check_single_beam_stress(self, bid, forces):

        beam = self.model.beams[bid]
        metric = beam_stress_metric(beam, forces)
        if beam.is_virtual or metric.ratio <= 0.0:
            return
        return metric.ratio, metric.stress_mpa, bid, metric.mode

    def _check_beams(self, res: AnalysisResults) -> Tuple[float, int, str]:
        """校核所有梁的强度 (正应力 + 剪应力)"""
        max_ratio = 0.0
        max_stress = 0.0
        crit_id = -1
        mode = "OK"

        for bid, forces in res.beam_forces.items():
            if bid not in self.model.beams:
                continue
            res = self._check_single_beam_stress(bid, forces)
            if res is None:
                continue
            b_ratio, b_stress, b_id, b_mode = res
            if b_ratio > max_ratio:
                max_ratio = b_ratio
                max_stress = b_stress
                crit_id = b_id
                mode = b_mode

        return max_ratio, max_stress, crit_id, mode

    def _check_plates(self, res: AnalysisResults) -> Tuple[float, int]:
        """
        校核所有板的 Von Mises 应力.
        修正逻辑: 遍历 ShellDKGQ 单元的 4 个高斯积分点 x (顶面+底面)
        """
        max_ratio = 0.0
        max_stress = 0.0
        crit_id = -1

        for pid, stress_raw in res.plate_stresses.items():
            metric = plate_stress_metric(self.model, pid, stress_raw)
            if metric.ratio > max_ratio:
                max_ratio = metric.ratio
                max_stress = metric.stress_mpa
                crit_id = pid

        return max_ratio, max_stress, crit_id

    def _check_deflection(self, res: AnalysisResults) -> Tuple[float, float, int]:
        """校核最大挠度比 (Disp / L)"""
        # 这里做一个简化：计算所有节点的最大位移，除以结构最短跨度或特征长度
        # 严格来说应该按构件计算 (Mid-span deflection relative to ends)

        max_disp = 0.0
        max_disp_node = -1
        for node_id, d in res.displacements.items():
            disp_norm = np.linalg.norm(d[:3])  # XYZ displacement
            if disp_norm > max_disp:
                max_disp = disp_norm
                max_disp_node = node_id

        L_ref_mm = self.model.params.width * M_TO_MM

        # 规范限值: L/250
        limit = L_ref_mm / 250.0
        return max_disp / limit, max_disp, max_disp_node


def run_full_design_check(model: StructuralModel, internal_pressure_pa: float = 30000):
    solver = OpenSeesSolver(model)

    # 1. 定义荷载组合
    combo_strength = {"DL": 1.3, "LL": 1.5, "PL": 1.25}  # 承载能力极限状态
    combo_service = {"DL": 1.0, "LL": 1.0, "PL": 1.0}  # 正常使用极限状态

    # 2. 直接运行荷载组合分析
    res_strength = solver.run_load_combination(combo_strength, pressure_pa=internal_pressure_pa)
    res_service = solver.run_load_combination(combo_service, pressure_pa=internal_pressure_pa)

    # 3. 后处理校核
    processor = ResultProcessor(model)

    # 临时修正：我们用强度组合结果跑一次，主要看 stress
    strength_check = processor.run_gb50017_check(res_strength)

    # 用位移组合结果跑一次，主要看 deflection
    service_check = processor.run_gb50017_check(res_service)

    print("====== Design Check Report (GB50017) ======")
    print(
        f"Max Stress Ratio: {strength_check.max_stress_ratio:.3f} "
        f"(Elem: {strength_check.critical_beam_id}/{strength_check.critical_plate_id}, "
        f"Mode: {strength_check.failure_mode})"
    )

    print(f"Max Deflection Ratio: {service_check.max_deflection_ratio:.3f}")

    is_safe = (strength_check.max_stress_ratio <= 1.0) and (service_check.max_deflection_ratio <= 1.0)
    print(f"Design Status: {'PASSED' if is_safe else 'FAILED'}")

    return is_safe


if __name__ == "__main__":
    from src.structure.mesh_generator import MeshGenerator

    # 配置参数
    params = StructuralParameters(
        length=8,
        width=3.7,
        height=3.3,
        main_beam_spacing=1.05,
        num_secondary_width=3,  # 宽度方向分3段 (2条中间次梁)
        num_secondary_height=3,  # 高度方向分3段
        add_support_beam=True,  # 添加支撑梁
        frame_section_index=0,
        main_beam_section_index=0,
        secondary_beam_section_index=0,
        brace_section_index=0,
        plate_thickness=0.005,
    )

    model = StructuralModel(params, SectionLibrary())
    model.generate()

    mesh_generator = MeshGenerator(model, 0.3)
    model = mesh_generator.generate()

    run_full_design_check(model, internal_pressure_pa=30000)
