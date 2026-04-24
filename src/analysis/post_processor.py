import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.analysis.opensees_analysis import OpenSeesSolver
from src.analysis.results import AnalysisResults
from src.analysis.unit import M2_TO_MM2, M4_TO_MM4, M_TO_MM
from src.structure.layout.params import StructuralParameters
from src.structure.section_library import SectionLibrary
from src.structure.structural_model import BeamType, StructuralModel

# 钢材设计强度 (Q345 示例)
DESIGN_STRENGTH_F = 314  # MPa (N/mm2)
DESIGN_SHEAR_FV = 175.0  # MPa (N/mm2)


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
        if beam.is_virtual:
            return

        # 获取截面属性 (SI单位 -> MM单位)
        sec = beam.section
        A = sec.area * M2_TO_MM2
        # 简化计算模量 W = I / (H/2)
        # 注意：OpenSees Iz 是强轴(绕z), Iy 是弱轴(绕y) - 需根据 SectionLibrary 具体定义校对
        # 通常 H型钢库中: Ix是强轴, Iy是弱轴。
        # 在 ops_analysis 中我们映射了: OPS_Iz = Lib_Ix (强), OPS_Iy = Lib_Iy (弱)

        # 计算截面模量 W (mm3)
        # 假设 section 类有 H 和 B 属性 (如果是 Box/H/Channel)
        # 如果是通用 SectionProperties，可能需要估算或增加字段
        # 这里做保守估计：假设它是对称截面
        if hasattr(sec, "H") and sec.H > 0:
            Wz = (sec.Ix * M4_TO_MM4) / (sec.H * M_TO_MM / 2)  # 强轴模量
        else:
            Wz = (sec.Ix * M4_TO_MM4) / math.sqrt(sec.area * M2_TO_MM2)  # 粗略兜底

        if hasattr(sec, "B") and sec.B > 0:
            Wy = (sec.Iy * M4_TO_MM4) / (sec.B * M_TO_MM / 2)  # 弱轴模量
        else:
            Wy = (sec.Iy * M4_TO_MM4) / math.sqrt(sec.area * M2_TO_MM2)

        # --- 提取内力 (OpenSees 局部坐标) ---
        # 0:Fx(Axial), 1:Fy(ShearY), 2:Fz(ShearZ), 3:T, 4:My(Weak), 5:Mz(Strong)
        # 我们检查 I端(0-5) 和 J端(6-11)

        max_ratio = 0.0
        max_stress = 0.0
        crit_id = -1
        mode = "OK"

        for i in [0, 6]:  # 两个端点
            N = forces[i + 0]  # 轴力 (+拉 -压)
            Vy = forces[i + 1]  # 剪力Y
            Vz = forces[i + 2]  # 剪力Z
            My = forces[i + 4]  # 弯矩Y (绕弱轴)
            Mz = forces[i + 5]  # 弯矩Z (绕强轴)

            # 1. 正应力校核: sigma = N/A + Mx/Wx + My/Wy
            sigma = abs(N / A) + abs(Mz / Wz) + abs(My / Wy)
            ratio_sigma = sigma / DESIGN_STRENGTH_F

            # 2. 剪应力校核 (简化: tau = V / A_web)
            # 假设剪力主要由腹板承担，粗略用 A/2 或 A_web
            tau = math.sqrt(Vy**2 + Vz**2) / (A * 0.5)  # 保守估计
            ratio_tau = tau / DESIGN_SHEAR_FV

            curr_max = max(ratio_sigma, ratio_tau)
            curr_max_stress = max(sigma, tau)
            if curr_max > max_ratio:
                max_ratio = curr_max
                max_stress = curr_max_stress
                crit_id = bid
                mode = "Bending" if ratio_sigma > ratio_tau else "Shear"
        return max_ratio, max_stress, crit_id, mode

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
            plate = self.model.plates.get(pid)
            if plate is None:
                continue
            thickness_mm = (
                self.model.params.opening_panel.thickness if plate.panel_type == "opening" else self.model.params.plate_thickness
            ) * M_TO_MM
            t2 = thickness_mm**2
            c_N = 1.0 / thickness_mm
            c_M = 6.0 / t2
            design_strength = (
                self.model.params.opening_panel.design_strength_mpa
                if plate.panel_type == "opening"
                else DESIGN_STRENGTH_F
            )
            # 遍历 4 个高斯点
            for i in range(4):
                offset = i * 8

                # 提取内力分量 (N: N/mm, M: N*mm/mm)
                # 忽略 Vxz, Vyz (通常在薄板 Mises 校核中忽略层间剪切，或作为单独项校核)
                Nxx = stress_raw[offset + 0]
                Nyy = stress_raw[offset + 1]
                Nxy = stress_raw[offset + 2]
                Mxx = stress_raw[offset + 3]
                Myy = stress_raw[offset + 4]
                Mxy = stress_raw[offset + 5]

                # 检查该高斯点的 顶面(Top, +) 和 底面(Bottom, -)
                for sign in [1.0, -1.0]:
                    # 计算正应力与剪应力
                    # sigma = N/t ± 6M/t^2
                    sig_x = Nxx * c_N + sign * Mxx * c_M
                    sig_y = Nyy * c_N + sign * Myy * c_M
                    tau_xy = Nxy * c_N + sign * Mxy * c_M

                    # Von Mises Stress 计算公式
                    # sigma_vm = sqrt(sx^2 + sy^2 - sx*sy + 3*txy^2)
                    mises = math.sqrt(sig_x**2 + sig_y**2 - sig_x * sig_y + 3 * tau_xy**2)

                    ratio = mises / design_strength

                    if ratio > max_ratio:
                        max_ratio = ratio
                        max_stress = mises
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
