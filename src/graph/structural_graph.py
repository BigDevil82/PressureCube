from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from src.structure.section_library import SectionLibrary, SectionType
from src.structure.structural_model import Beam, BeamType, StructuralModel


class BeamAsNodeGraph:
    def __init__(self, model: StructuralModel, current_pressure: float = 30000.0):
        self.model = model
        self.nodes: Dict[int, Beam] = self.model.beams
        self.current_pressure = current_pressure  # 接收当前环境的压力值

        # 1. 动态计算更精准的归一化因子
        self.norm_scale = self._calculate_dynamic_normalization()

        # 2. 缓存邻接关系
        self.adj: Dict[int, List[int]] = defaultdict(list)
        self._build_adjacency()

    def _calculate_dynamic_normalization(self) -> Dict[str, float]:
        """动态扫描截面库，获取真实的物理属性边界"""
        lib = self.model.section_lib

        max_area = 1e-8
        max_I = 1e-8

        # 遍历库中所有类型的截面
        for sec_list in lib._sections.values():
            for sec in sec_list:
                if sec.area > max_area:
                    max_area = sec.area
                # 取强轴惯性矩作为基准，保守一点
                if sec.Ix > max_I:
                    max_I = sec.Ix

        # 几何归一化：取包围盒最大边长
        L, W, H = self.model.params.length, self.model.params.width, self.model.params.height
        max_dim = max(L, W, H, 1.0)  # 防止为0

        return {
            "coord": max_dim,
            "area": max_area,  # 动态最大面积
            "I": max_I,  # 动态最大惯性矩
            "load": 50000.0,  # 假设最大设计气压 50kPa
            "stress_ratio": 5.0,  # 截断阈值
            "disp_ratio": 5.0,  # 截断阈值
        }

    def _build_adjacency(self):
        """构建邻接表 (不变)"""
        node_to_beams = defaultdict(list)
        for beam_id, beam in self.nodes.items():
            if beam.is_virtual:
                continue
            node_to_beams[beam.start_node].append(beam_id)
            node_to_beams[beam.end_node].append(beam_id)

        for _, connected_beams in node_to_beams.items():
            if len(connected_beams) > 1:
                for i in range(len(connected_beams)):
                    for j in range(i + 1, len(connected_beams)):
                        b1, b2 = connected_beams[i], connected_beams[j]
                        self.adj[b1].append(b2)
                        self.adj[b2].append(b1)

    def to_pyg_data(self) -> Data:
        """生成增强版特征的 PyG Data"""
        valid_beam_ids = sorted([b.id for b in self.nodes.values() if not b.is_virtual])
        beam_id_to_idx = {bid: i for i, bid in enumerate(valid_beam_ids)}

        if not valid_beam_ids:
            return Data()

        node_features = []

        # 预先获取库的大小，用于归一化 Index
        lib = self.model.section_lib
        counts = lib.section_counts
        # 映射 BeamType 到对应的 SectionType
        type_map = {
            BeamType.FRAME: SectionType.BOX,
            BeamType.MAIN: SectionType.H_MAIN,
            BeamType.SECONDARY: SectionType.H_SEC,
            BeamType.BRACE: SectionType.CHANNEL,
        }

        for bid in valid_beam_ids:
            beam = self.nodes[bid]

            # --- A. 几何特征 (6 dims) ---
            p1 = self.model.points[beam.start_node].coordinates()
            p2 = self.model.points[beam.end_node].coordinates()
            center = (p1 + p2) / 2.0
            direction = p2 - p1
            length = np.linalg.norm(direction)
            direction_norm = direction / (length + 1e-6)

            feat_geom = list(center / self.norm_scale["coord"]) + list(direction_norm)

            # --- B. 类型特征 (4 dims - One Hot) ---
            feat_type = [0.0] * 4
            if beam.beam_type.value < 4:
                feat_type[beam.beam_type.value] = 1.0

            # --- C. 物理与状态特征 (改进核心) ---
            sec = beam.section

            # 1. 归一化的物理属性 (相对整个库的强弱)
            # Area/Max, I/Max. 这告诉智能体"我现在的绝对强度是多少"
            feat_phys_abs = [
                sec.area / self.norm_scale["area"],
                sec.Ix / self.norm_scale["I"],
                sec.Iy / self.norm_scale["I"],
            ]

            # 2. [New] 归一化的索引位置 (Relative Index Position)
            # 这告诉智能体"我还有多少升级/降级的空间"
            # 如果 idx_norm 接近 0，智能体就能学会不要输出 -1 动作
            s_type = type_map.get(beam.beam_type, SectionType.BOX)
            max_idx = counts.get(s_type, 1) - 1
            idx_norm = sec.idx / max(max_idx, 1)  # 0.0 ~ 1.0

            # 3. [New] 全局荷载上下文
            # 广播到每个节点，让智能体知道当前的恶劣程度
            load_norm = self.current_pressure / self.norm_scale["load"]

            feat_state = [idx_norm, load_norm]

            # --- D. 响应特征 (4 dims) ---
            if beam.response:
                # 使用 Tanh 或 Log 压缩，或者硬截断，防止梯度爆炸
                sr = beam.response.max_stress_ratio
                dr = beam.response.max_disp_ratio

                # 特征工程：显式提供 Violation 信号
                sr_viol = max(0.0, sr - 1.0)
                dr_viol = max(0.0, dr - 1.0)

                # 截断到 [0, 5]
                feat_resp = [
                    min(sr, self.norm_scale["stress_ratio"]),
                    min(dr, self.norm_scale["disp_ratio"]),
                    min(sr_viol, 4.0),  # 违约量
                    min(dr_viol, 4.0),
                ]
            else:
                feat_resp = [0.0, 0.0, 0.0, 0.0]

            # 合并: 6 + 4 + 3 + 2 + 4 = 19 dims
            # 请记得修改 train_main.py 中的 NODE_DIM = 19
            node_features.append(feat_geom + feat_type + feat_phys_abs + feat_state + feat_resp)

        x = torch.tensor(node_features, dtype=torch.float)

        # --- E. 边特征 (保持原样或增加角度特征) ---
        # 简单起见，目前保持共享节点坐标即可。
        # 如果后续想优化拓扑，可以在这里计算两根梁的夹角余弦值作为边特征。
        edge_indices = []
        edge_attrs = []

        for bid_u in valid_beam_ids:
            if bid_u not in self.adj:
                continue
            u_idx = beam_id_to_idx[bid_u]

            for bid_v in self.adj[bid_u]:
                if bid_v not in beam_id_to_idx:
                    continue
                v_idx = beam_id_to_idx[bid_v]

                edge_indices.append([u_idx, v_idx])

                # 边特征：共享节点坐标
                b_u, b_v = self.nodes[bid_u], self.nodes[bid_v]
                if b_u.start_node in (b_v.start_node, b_v.end_node):
                    shared_p = self.model.points[b_u.start_node].coordinates()
                else:
                    shared_p = self.model.points[b_u.end_node].coordinates()

                edge_attrs.append(list(shared_p / self.norm_scale["coord"]))

        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 3), dtype=torch.float)

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
