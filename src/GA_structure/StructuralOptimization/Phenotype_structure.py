
import numpy as np
# from src.analysis.opensees_analysis import OpenSeesAnalyzer
from ...analysis.opensees_analysis import OpenSeesAnalyzer


# from src.utils.model_utils import create_structural_model
from ...utils.model_utils import create_structural_model
# from src.structure.mesh_generator import MeshGenerator, MeshParameters
from ...structure.mesh_generator import MeshGenerator, MeshParameters

class PhenotypeStructure:
    MAX_STRESS = 355.0  # MPa
    MAX_DISPLACEMENT = 14.4  # mm
    MAX_DRIFT_RATIO = 0.005  # H/200

    def __init__(self, params: dict):
        """
        初始化结构表现型（接收结构设计参数）
        :param params: 所有结构参数（包括固定参数 + 优化参数）
        """
        self.params = params
        self.fitness = None
        self._mass_kg = None
        self._constraints_satisfied = None
        self._analysis_result = None
        self._run_analysis()

    def _run_analysis(self):
        """
        执行结构分析（建模 → 网格划分 → OpenSees 分析）
        """
        try:
            # 1. 基于参数生成结构模型
            model = create_structural_model({"structural_params": self.params})  # 伪造嵌套结构

            # model = create_structural_model(self.params)

            # 2. 网格划分
            mesh_params = MeshParameters(max_grid_size=0.2)
            mesh_generator = MeshGenerator(model, mesh_params)
            mesh_generator.generate_mesh()
            meshed_model = mesh_generator.meshed_model

            # 3. 执行 OpenSees 分析
            analyzer = OpenSeesAnalyzer()
            analyzer.setup_opensees_model(meshed_model)
            analyzer.apply_pressure_loads(self.params["pressure"]*1e3)
            success = analyzer.run_static_analysis()

            if not success:
                print("\n [警告] OpenSees 静力分析未成功运行")
                print(f" 当前设计参数: {self.params}")
                self._constraints_satisfied = False
                self.fitness = 1e9  # 失败个体惩罚
                return

            # 4. 获取分析指标
            indicators = analyzer.get_max_indicators()

            max_stress = indicators["max_abs_stress"]       # MPa
            max_disp = indicators["max_abs_displacement"]   # mm
            
            # 5. 计算质量（优化目标）以 kg 为单位
            self._mass_kg = meshed_model.calculate_mass()
            self.fitness = self._mass_kg

            # print("\n [分析结果指标]")
            # print(f"  - 最大应力 (MPa)：{max_stress:.2f}")
            # print(f"  - 最大位移 (mm)：{max_disp:.2f}")
            # drift_ratio_ok = self._check_drift_ratio(analyzer)
            # print(f"  - 层间位移角满足限制：{'✔️' if drift_ratio_ok else '❌'}")

            # 可选：显示质量
            # print(f"  - 结构总质量 (kg)：{self._mass_kg:.2f}")


            # 6. 验证约束条件
            satisfies = (
                max_stress <= self.MAX_STRESS and
                max_disp <= self.MAX_DISPLACEMENT 
                # self._check_drift_ratio(analyzer)
            )
            self._constraints_satisfied = satisfies

        except Exception as e:
            # 任意错误都认为不可行，严重惩罚
            print("\n [错误] OpenSees 分析过程发生异常")
            import traceback
            traceback.print_exc()
            print(f" 当前设计参数: {self.params}")
            self.fitness = 1e9
            self._constraints_satisfied = False

    def _check_drift_ratio(self, analyzer: OpenSeesAnalyzer) -> bool:
        """
        简化层间位移角检查（根据节点高度差近似估算）
        加入 OpenSees 收敛判断，确保分析有效
        """
        # 检查 OpenSees 分析是否成功收敛
        if not analyzer._static_analysis_success:
            print("Warning: OpenSees 静态分析未运行或未收敛！")
            return False  # 分析失败，直接返回 False

        displacements = analyzer.get_node_displacements()
        if not displacements:
            print("Warning: 无位移数据，可能分析失败！")
            return False  # 无位移数据，分析可能失败

        z_coords = {}
        lateral_disp = {}

        for node_id, disp in displacements.items():
            node = analyzer._model.points[node_id]
            z = node.z
            z_coords.setdefault(z, []).append(node_id)
            lateral_disp[node_id] = disp[0]  # 仅用 X 方向位移（或 disp[1] 用 Y 方向）

        if len(z_coords) < 2:
            return True  # 只有一层则满足

        sorted_heights = sorted(z_coords.keys())
        max_ratio = 0

        for i in range(1, len(sorted_heights)):
            z1, z2 = sorted_heights[i - 1], sorted_heights[i]
            h = z2 - z1  # 假设 h 单位是米
            if h <= 1e-6:  # 避免除零或极小高度差
                continue

            nodes_upper = z_coords[z2]
            nodes_lower = z_coords[z1]

            disp_upper = np.mean([lateral_disp[nid] for nid in nodes_upper])
            disp_lower = np.mean([lateral_disp[nid] for nid in nodes_lower])
            drift = abs(disp_upper - disp_lower)
            ratio = drift / (h * 1000)  # 将 h 转换为毫米，确保单位一致
            max_ratio = max(max_ratio, ratio)

        # print("max_ratio:", max_ratio)
        return max_ratio <= self.MAX_DRIFT_RATIO

    def is_feasible(self) -> bool:
        """
        判断该结构解是否满足全部约束条件
        """
        return self._constraints_satisfied

    def get_objective(self) -> float:
        """
        获取优化目标（结构总质量，越小越好）
        """
        return self.fitness

    def __str__(self):
        status = "Feasible" if self.is_feasible() else "Infeasible"
        return f"[{status}] Mass = {self.fitness:.2f} kg"
