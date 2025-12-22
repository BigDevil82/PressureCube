from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from src.structure.section_library import SectionLibrary

# 假设 structural_model.py 已经按照之前的讨论重构完成
from src.structure.structural_model import Beam, BeamType, StructuralModel

# Ensuring imports are relative to the 'src' directory structure.
# This assumes that 'src' is part of PYTHONPATH or the execution context.
from ..structure.structural_model import Beam as ModelBeam
from ..structure.structural_model import Point as ModelPoint
from ..structure.structural_model import StructuralModel


class BeamAsNodeGraph:
    """
    Beam-as-Node 图表示法生成器。

    图定义:
    - 节点 (Nodes): 对应结构中的物理梁 (Beam)。
    - 边 (Edges): 如果两根梁共享同一个物理连接点 (Point)，则它们之间存在边。
    """

    def __init__(self, model: StructuralModel):
        self.model = model
        self.nodes: Dict[int, Beam] = self.model.beams

        # 预计算归一化因子 (基于模型边界框)
        self.norm_scale = self._calculate_normalization_factors()

        # 邻接表缓存
        self.adj: Dict[int, List[int]] = defaultdict(list)
        self._build_adjacency()

    def _calculate_normalization_factors(self) -> Dict[str, float]:
        """计算用于特征归一化的缩放因子"""
        # 几何归一化
        L, W, H = self.model.params.length, self.model.params.width, self.model.params.height
        max_dim = max(L, W, H)

        # 截面属性归一化 (从SectionLibrary获取最大值)
        # 注意：这里需要确保 section_lib 已经加载
        lib = self.model.section_lib
        # 简单遍历库中所有截面找最大值，或者硬编码一个较大的经验值
        # 这里为了稳健性，使用硬编码的经验上限，防止除零
        max_area = 0.01  # m^2 (100 cm^2)
        max_I = 0.0005  # m^4 (50000 cm^4)

        # 尝试从库中动态获取（如果库已填充）
        try:
            # 这是一个简化的获取逻辑，实际需遍历库
            pass
        except:
            pass

        return {
            "coord": max_dim,
            "area": max_area,
            "I": max_I,
            "load": 10000.0,  # 10 kN 作为荷载归一化基准
            "stress": 355.0,  # MPa, 屈服强度
            "disp": W * 1000 / 250.0,  # 容许挠度作为基准
        }

    def _build_adjacency(self):
        """构建图的邻接关系：节点共享即相连"""
        # 1. 建立 Node -> [BeamID, ...] 的映射
        node_to_beams = defaultdict(list)
        for beam_id, beam in self.nodes.items():
            if beam.is_virtual:
                continue  # 跳过虚拟梁（如果有）

            node_to_beams[beam.start_node].append(beam_id)
            node_to_beams[beam.end_node].append(beam_id)

        # 2. 构建 Beam -> [Beam, ...] 的邻接表
        for _, connected_beams in node_to_beams.items():
            if len(connected_beams) > 1:
                # 全连接：同一个节点上的所有梁互为邻居
                for i in range(len(connected_beams)):
                    for j in range(i + 1, len(connected_beams)):
                        b1, b2 = connected_beams[i], connected_beams[j]
                        self.adj[b1].append(b2)
                        self.adj[b2].append(b1)

    def to_pyg_data(self) -> Data:
        """
        转换为 PyTorch Geometric 的 Data 对象。
        此函数应在每次 RL step 之后调用，以获取最新的状态（应力、位移）。
        """
        # 过滤掉虚拟梁，并确保 ID 排序一致以保证特征矩阵对应的物理意义不变
        valid_beam_ids = sorted([b.id for b in self.nodes.values() if not b.is_virtual])
        beam_id_to_idx = {bid: i for i, bid in enumerate(valid_beam_ids)}

        num_nodes = len(valid_beam_ids)
        if num_nodes == 0:
            return Data()

        # --- 1. 构建节点特征矩阵 X ---
        # 维度: [num_beams, feature_dim]
        node_features = []

        for bid in valid_beam_ids:
            beam = self.nodes[bid]

            # A. 几何特征 (6 dims)
            # 质心坐标 (3) + 方向向量 (3)
            p_start = self.model.points[beam.start_node]
            p_end = self.model.points[beam.end_node]

            center = (p_start.coordinates() + p_end.coordinates()) / 2.0
            direction = p_end.coordinates() - p_start.coordinates()
            length = np.linalg.norm(direction)
            direction_norm = direction / (length + 1e-6)

            feat_geom = list(center / self.norm_scale["coord"]) + list(direction_norm)

            # B. 拓扑/类型特征 (4 dims - One Hot)
            # Frame, Main, Secondary, Brace
            feat_type = [0, 0, 0, 0]
            if beam.beam_type.value < 4:
                feat_type[beam.beam_type.value] = 1

            # C. 物理/设计变量特征 (3 dims)
            # Area, Ix, Iy (这是 RL Agent 改变截面索引后直接影响的物理量)
            sec = beam.section
            feat_phys = [
                sec.area / self.norm_scale["area"],
                sec.Ix / self.norm_scale["I"],
                sec.Iy / self.norm_scale["I"],
            ]

            # D. 响应特征 (4 dims) - RL 的关键观测值
            # 假设 model 中已经有了分析结果，或者通过外部注入
            # 这里需要你在 StructuralModel 或 Beam 类中维护一个 response 字段
            # response 包含: max_stress_ratio, max_disp_ratio
            # 如果还没有分析，默认为 0
            if beam.response:
                stress_r = beam.response.max_stress_ratio
                disp_r = beam.response.max_disp_ratio

                # [Fix] 对输入特征进行 Log 缩放或 Clip 截断，防止数值爆炸
                # 建议使用 Clip，限制在 [0, 5] 范围内，因为大于 5 已经是极度危险，
                # 对神经网络来说 5 和 20 都是“非常大”，没必要保留那么大的方差。
                stress_r_clamped = min(stress_r, 5.0)
                disp_r_clamped = min(disp_r, 5.0)

                stress_viol = max(0, stress_r - 1.0)
                disp_viol = max(0, disp_r - 1.0)

                feat_resp = [stress_r_clamped, disp_r_clamped, stress_viol, disp_viol]
            else:
                feat_resp = [0.0, 0.0, 0.0, 0.0]

            # 合并所有特征
            # Total dim = 6 + 4 + 3 + 4 = 17
            node_features.append(feat_geom + feat_type + feat_phys + feat_resp)

        x = torch.tensor(node_features, dtype=torch.float)

        # --- 2. 构建边索引 Edge Index ---
        # 维度: [2, num_edges]
        edge_indices = []
        edge_attrs = []  # 可选：边特征（例如连接节点的类型：刚接/铰接，或坐标）

        for bid_u in valid_beam_ids:
            if bid_u not in self.adj:
                continue

            u_idx = beam_id_to_idx[bid_u]

            for bid_v in self.adj[bid_u]:
                if bid_v not in beam_id_to_idx:
                    continue  # 过滤掉可能的虚拟梁邻居
                v_idx = beam_id_to_idx[bid_v]

                # 添加有向边 (PyG 默认为有向，无向图需添加双向)
                edge_indices.append([u_idx, v_idx])

                # 边特征：共享节点的坐标 (归一化)
                # 找到共享节点
                beam_u = self.nodes[bid_u]
                beam_v = self.nodes[bid_v]
                shared_node_id = None
                if beam_u.start_node in (beam_v.start_node, beam_v.end_node):
                    shared_node_id = beam_u.start_node
                else:
                    shared_node_id = beam_u.end_node

                p_shared = self.model.points[shared_node_id].coordinates()
                edge_attrs.append(list(p_shared / self.norm_scale["coord"]))

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


if __name__ == "__main__":
    from src.utils.model_utils import create_structural_model, load_config

    print("Testing StructuralGraph to PyG Data conversion...")
    config_path = "config.yaml"
    config = load_config(config_path)
    model = create_structural_model(config)

    # Test BeamAsNodeGraph (beams as nodes, connections as edges)
    print("\n--- Testing BeamAsNodeGraph (beams as nodes) ---")
    graph_type2 = BeamAsNodeGraph(model)
    print(f"Created: {graph_type2}")
    pyg_data_type2 = graph_type2.to_pyg_data()
    print("PyG Data (Type 2):")
    print(pyg_data_type2)
    if pyg_data_type2:
        print(f"  Node features shape: {pyg_data_type2.x.shape if pyg_data_type2.x is not None else 'N/A'}")
        print(
            f"  Edge index shape: {pyg_data_type2.edge_index.shape if pyg_data_type2.edge_index is not None else 'N/A'}"
        )
        print(
            f"  Edge features shape: {pyg_data_type2.edge_attr.shape if pyg_data_type2.edge_attr is not None else 'N/A'}"
        )

    print("\nTest script finished.")
