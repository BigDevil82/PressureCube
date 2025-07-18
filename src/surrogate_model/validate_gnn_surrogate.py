import os
import sys
from pathlib import Path
from traceback import print_exc

import joblib
import numpy as np
import torch
from torch_geometric.data import Data

# 将项目根目录添加到Python路径中
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.analysis.opensees_analysis import OpenSeesAnalyzer
from src.graph.structural_graph import BeamAsNodeGraph
from src.structure.mesh_generator import MeshGenerator, MeshParameters
from src.structure.random_structure import RandomStructure
from src.structure.section_library import SectionLibrary
from src.structure.structural_model import StructuralModel, StructuralParameters

# 导入训练脚本中定义的模型类
from src.surrogate_model.train_gnn_surrogate import EfficientGNNRegressor


class GNNPredictor:
    """封装了GNN模型加载和预测的逻辑"""

    def __init__(self, model_dir: str, device: torch.device):
        self.device = device
        self.section_library = SectionLibrary()

        model_path = os.path.join(model_dir, "efficient_gnn_surrogate_model.pth")
        node_scaler_path = os.path.join(model_dir, "gnn_node_scaler.joblib")
        target_scaler_path = os.path.join(model_dir, "gnn_target_scaler.joblib")

        if not all(os.path.exists(p) for p in [model_path, node_scaler_path, target_scaler_path]):
            raise FileNotFoundError("找不到模型或scaler文件。请先运行训练脚本。")

        # 加载scalers
        self.node_scaler = joblib.load(node_scaler_path)
        self.target_scaler = joblib.load(target_scaler_path)

        # 加载模型
        # 从scalers推断维度
        input_dim = self.node_scaler.n_features_in_
        output_dim = self.target_scaler.n_features_in_

        self.model = EfficientGNNRegressor(
            input_dim=input_dim, hidden_dim=64, output_dim=output_dim, num_heads=4
        )
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(self.device)
        self.model.eval()
        print("GNN预测器已成功初始化。")

    def _create_graph_from_params(self, params: StructuralParameters) -> Data:
        """根据结构参数创建图数据对象"""
        model = StructuralModel(params, self.section_library)
        model.generate_grid_mesh()
        graph_converter = BeamAsNodeGraph(model)
        pyg_data = graph_converter.to_pyg_data({})
        return pyg_data

    def predict(self, params: StructuralParameters) -> tuple[float, float]:
        """对给定的结构参数进行预测"""
        # 1. 创建图
        graph_data = self._create_graph_from_params(params)

        # 2. 标准化节点特征
        node_features_scaled = self.node_scaler.transform(graph_data.x.numpy())
        graph_data.x = torch.from_numpy(node_features_scaled).float()

        # 3. 移动到设备并预测
        graph_data = graph_data.to(self.device)
        with torch.no_grad():
            prediction_scaled = self.model(graph_data)

        # 4. 反标准化得到最终结果
        prediction = self.target_scaler.inverse_transform(prediction_scaled.cpu().numpy())

        return prediction[0][0], prediction[0][1]


def generate_and_validate_sample(predictor: GNNPredictor):
    """生成一个随机样本，运行有限元分析，并与GNN预测进行比较。"""
    try:
        # 1. 生成随机参数
        random_structure_gen = RandomStructure(predictor.section_library)
        params = random_structure_gen.generate_random_parameters(generate_windows=False)

        # 2. 运行有限元分析获取真实值
        model = StructuralModel(params, predictor.section_library)
        model.generate_grid_mesh()
        analyzer = OpenSeesAnalyzer()
        analyzer.setup_opensees_model(model)
        analyzer.apply_pressure_loads(internal_pressure_pa=params.pressure * 1000.0)
        analyzer.run_static_analysis()
        max_indicators = analyzer.get_max_indicators()
        true_stress = max_indicators["max_abs_stress"]
        true_disp = max_indicators["max_abs_displacement"]

        # 3. 使用GNN模型进行预测
        pred_stress, pred_disp = predictor.predict(params)

        return (true_stress, true_disp), (pred_stress, pred_disp)

    except Exception:
        print("生成验证样本时出错:")
        print_exc()
        return None, None


def main(num_samples_to_validate: int = 10):
    """主函数，运行验证流程。"""
    model_dir = "results/gnn_surrogate_model"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        predictor = GNNPredictor(model_dir=model_dir, device=device)
    except FileNotFoundError as e:
        print(e)
        return

    total_stress_err = 0
    total_disp_err = 0
    success_count = 0

    for i in range(num_samples_to_validate):
        print(f"\n--- 正在验证样本 {i + 1}/{num_samples_to_validate} ---")
        true_values, pred_values = generate_and_validate_sample(predictor)

        if true_values and pred_values:
            true_stress, true_disp = true_values
            pred_stress, pred_disp = pred_values

            # 避免除以零
            if true_stress == 0 or true_disp == 0:
                print("真实值为零，跳过误差计算。")
                continue

            stress_error = abs(pred_stress - true_stress) / true_stress * 100
            disp_error = abs(pred_disp - true_disp) / true_disp * 100

            total_stress_err += stress_error
            total_disp_err += disp_error
            success_count += 1

            print(f"  真实值:     应力 = {true_stress:.2f} MPa, 位移 = {true_disp:.3f} mm")
            print(f"  预测值:     应力 = {pred_stress:.2f} MPa, 位移 = {pred_disp:.3f} mm")
            print(f"  相对误差:   应力 = {stress_error:.2f}%,   位移 = {disp_error:.2f}%")

    if success_count > 0:
        print("\n--- 验证总结 ---")
        print(f"成功验证样本数: {success_count}/{num_samples_to_validate}")
        print(f"平均相对误差 (应力): {total_stress_err / success_count:.2f}%")
        print(f"平均相对误差 (位移): {total_disp_err / success_count:.2f}%")


if __name__ == "__main__":
    main()
