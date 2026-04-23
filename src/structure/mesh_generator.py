import math
from typing import Dict, List, Optional

import numpy as np

from src.structure.structural_model import Beam, BeamType, NodeManager, Plate, StructuralModel


class MeshGenerator:
    """
    适配新架构的结构模型网格生成器.
    核心逻辑：
    1. 初始划分：根据长度和 max_mesh_size 计算每根梁的段数。
    2. 约束传播：确保每个板的对边（U方向对边，V方向对边）划分数一致。
    3. 网格生成：生成新节点、细分梁、细分板。
    """

    def __init__(self, model: StructuralModel, max_mesh_size: float = 0.5):
        self.original_model = model
        self.max_mesh_size = max_mesh_size

        self.meshed_model = StructuralModel(model.params, model.section_lib)

        # 映射字典: Old ID -> New ID(s)
        self._node_map: Dict[int, int] = {}  # 旧节点 -> 新节点（端点）
        self._beam_divisions: Dict[int, int] = {}  # 梁ID -> 划分段数
        self._virtual_edge_divisions: Dict[tuple[int, int], int] = {}

        # 缓存：记录每根旧梁细分后产生的所有新节点序列（包含首尾）
        self._beam_new_nodes: Dict[int, List[int]] = {}

    def generate(self):
        # print("开始网格划分流程...")

        # 1. 初始划分计算
        self._init_beam_divisions()

        # 2. 约束传播 (Constraint Propagation)
        # 确保四边形板的对边具有相同的划分数
        self._propagate_constraints()

        # 3. 生成几何体
        self._generate_geometry()

        # print(
        #     f"网格划分完成. 节点数: {len(self.meshed_model.node_manager.points)}, "
        #     f"板单元数: {len(self.meshed_model.plates)}"
        # )
        return self.meshed_model

    def _init_beam_divisions(self):
        """根据长度计算初始划分数"""
        points = self.original_model.node_manager.points

        for beam in self.original_model.beam_manager.beams.values():
            p1 = points[beam.start_node]
            p2 = points[beam.end_node]
            length = np.linalg.norm([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])

            # 至少分1段
            divs = max(1, math.ceil(length / self.max_mesh_size))
            self._beam_divisions[beam.id] = divs

    def _propagate_constraints(self):
        """
        迭代松弛法：
        对于每个板，Top边和Bottom边的划分数必须相等，Left边和Right边必须相等。
        """
        max_iterations = 20
        for it in range(max_iterations):
            changed = False

            for plate in self.original_model.plates.values():
                # 板的节点顺序是逆时针: n0 -> n1 -> n2 -> n3
                # 边0: n0-n1 (Bottom/U)
                # 边1: n1-n2 (Right/V)
                # 边2: n2-n3 (Top/U)
                # 边3: n3-n0 (Left/V)

                nodes = plate.nodes
                edges = [
                    (nodes[0], nodes[1]),  # Edge 0
                    (nodes[1], nodes[2]),  # Edge 1
                    (nodes[2], nodes[3]),  # Edge 2
                    (nodes[3], nodes[0]),  # Edge 3
                ]

                # 获取每条边的当前划分数
                # 注意：在新模型中，两个节点之间可能没有物理梁，
                # 但为了网格兼容性，我们必须给这条“虚边”分配一个划分数。
                # 如果有梁，取梁的划分数；如果没有，我们需要临时记录并参与计算。
                edge_divs = []
                edge_defs = []

                for n_start, n_end in edges:
                    edge_def = self._get_edge_definition(n_start, n_end)
                    div = self._get_edge_division(edge_def)
                    edge_defs.append(edge_def)
                    edge_divs.append(div)

                # 比较 U 方向对边 (Edge 0 vs Edge 2)
                max_u = max(edge_divs[0], edge_divs[2])
                if edge_divs[0] < max_u or edge_divs[2] < max_u:
                    changed = True
                    self._update_edge_div(edge_defs[0], max_u)
                    self._update_edge_div(edge_defs[2], max_u)

                # 比较 V 方向对边 (Edge 1 vs Edge 3)
                max_v = max(edge_divs[1], edge_divs[3])
                if edge_divs[1] < max_v or edge_divs[3] < max_v:
                    changed = True
                    self._update_edge_div(edge_defs[1], max_v)
                    self._update_edge_div(edge_defs[3], max_v)

            if not changed:
                # print(f"约束传播在第 {it+1} 次迭代收敛。")
                break
        else:
            print("警告：约束传播未完全收敛，网格可能不完全对齐。")

    def _find_beam_id(self, n1: int, n2: int) -> Optional[int]:
        """在新模型的 BeamManager 中查找连接 n1, n2 的梁"""
        # 利用 BeamManager 的内部 map 加速查找
        key = tuple(sorted((n1, n2)))
        return self.original_model.beam_manager._beam_map.get(key)

    def _sorted_edge_key(self, n1: int, n2: int) -> tuple[int, int]:
        return tuple(sorted((n1, n2)))

    def _edge_length(self, n1: int, n2: int) -> float:
        p1 = self.original_model.node_manager.points[n1]
        p2 = self.original_model.node_manager.points[n2]
        return np.linalg.norm([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])

    def _get_segment_axis(self, n1: int, n2: int) -> int:
        p1 = self.original_model.node_manager.points[n1]
        p2 = self.original_model.node_manager.points[n2]
        deltas = [abs(p1.x - p2.x), abs(p1.y - p2.y), abs(p1.z - p2.z)]
        return int(np.argmax(deltas))

    def _get_edge_definition(self, n1: int, n2: int) -> dict:
        direct_beam_id = self._find_beam_id(n1, n2)
        if direct_beam_id is not None:
            return {"kind": "beam", "beam_ids": [direct_beam_id], "nodes": [n1, n2]}

        points = self.original_model.node_manager.points
        p1 = points[n1]
        p2 = points[n2]
        axis = self._get_segment_axis(n1, n2)
        tol = 1e-6

        segment_nodes = []
        for nid, point in points.items():
            coords = [point.x, point.y, point.z]
            start = [p1.x, p1.y, p1.z]
            end = [p2.x, p2.y, p2.z]

            if not (
                min(start[axis], end[axis]) - tol <= coords[axis] <= max(start[axis], end[axis]) + tol
            ):
                continue

            other_axes = [0, 1, 2]
            other_axes.remove(axis)
            if all(abs(coords[ax] - start[ax]) <= tol and abs(coords[ax] - end[ax]) <= tol for ax in other_axes):
                segment_nodes.append(nid)

        segment_nodes = sorted(
            set(segment_nodes),
            key=lambda nid: (
                self.original_model.node_manager.points[nid].coordinates()[axis],
                nid,
            ),
        )

        if n1 not in segment_nodes or n2 not in segment_nodes:
            return {"kind": "virtual", "beam_ids": [], "nodes": [n1, n2]}

        start_idx = segment_nodes.index(n1)
        end_idx = segment_nodes.index(n2)
        if start_idx > end_idx:
            segment_nodes = list(reversed(segment_nodes[end_idx : start_idx + 1]))
        else:
            segment_nodes = segment_nodes[start_idx : end_idx + 1]

        beam_ids = []
        for i in range(len(segment_nodes) - 1):
            beam_id = self._find_beam_id(segment_nodes[i], segment_nodes[i + 1])
            if beam_id is None:
                return {"kind": "virtual", "beam_ids": [], "nodes": [n1, n2]}
            beam_ids.append(beam_id)

        if beam_ids:
            return {"kind": "chain", "beam_ids": beam_ids, "nodes": segment_nodes}
        return {"kind": "virtual", "beam_ids": [], "nodes": [n1, n2]}

    def _get_edge_division(self, edge_def: dict) -> int:
        if edge_def["kind"] == "beam":
            return self._beam_divisions[edge_def["beam_ids"][0]]
        if edge_def["kind"] == "chain":
            return sum(self._beam_divisions[beam_id] for beam_id in edge_def["beam_ids"])

        edge_key = self._sorted_edge_key(edge_def["nodes"][0], edge_def["nodes"][-1])
        if edge_key not in self._virtual_edge_divisions:
            self._virtual_edge_divisions[edge_key] = max(
                1, math.ceil(self._edge_length(edge_key[0], edge_key[1]) / self.max_mesh_size)
            )
        return self._virtual_edge_divisions[edge_key]

    def _update_edge_div(self, edge_def, new_div):
        if edge_def["kind"] == "beam":
            self._beam_divisions[edge_def["beam_ids"][0]] = new_div
        elif edge_def["kind"] == "chain":
            beam_ids = edge_def["beam_ids"]
            total_len = sum(
                self._edge_length(
                    self.original_model.beams[beam_id].start_node,
                    self.original_model.beams[beam_id].end_node,
                )
                for beam_id in beam_ids
            )
            assigned = 0
            for idx, beam_id in enumerate(beam_ids):
                if idx == len(beam_ids) - 1:
                    div = max(1, new_div - assigned)
                else:
                    beam = self.original_model.beams[beam_id]
                    ratio = self._edge_length(beam.start_node, beam.end_node) / total_len if total_len > 1e-9 else 0
                    div = max(1, round(new_div * ratio))
                    remaining_segments = len(beam_ids) - idx - 1
                    div = min(div, new_div - assigned - remaining_segments)
                self._beam_divisions[beam_id] = div
                assigned += div
        else:
            edge_key = self._sorted_edge_key(edge_def["nodes"][0], edge_def["nodes"][-1])
            self._virtual_edge_divisions[edge_key] = new_div

    def _generate_geometry(self):
        """生成物理网格"""
        orig_nodes = self.original_model.node_manager.points

        # 1. 映射所有原始节点到新模型
        for nid, p in orig_nodes.items():
            new_id = self.meshed_model.node_manager.get_or_create(p.x, p.y, p.z)
            self._node_map[nid] = new_id

        # 2. 细分梁 (Meshing Beams)
        for beam in self.original_model.beam_manager.beams.values():
            self._mesh_single_beam(beam)

        # 3. 细分板 (Meshing Plates)
        for plate in self.original_model.plates.values():
            self._mesh_single_plate(plate)

    def _mesh_single_beam(self, beam: Beam):
        start_node = self.original_model.node_manager.points[beam.start_node]
        end_node = self.original_model.node_manager.points[beam.end_node]

        divs = self._beam_divisions[beam.id]

        # 获取新模型中的起止节点ID
        new_start_id = self._node_map[beam.start_node]
        new_end_id = self._node_map[beam.end_node]

        # 记录该梁上的所有节点序列（用于板的边界查找）
        node_sequence = [new_start_id]

        # 向量计算
        p1 = np.array([start_node.x, start_node.y, start_node.z])
        p2 = np.array([end_node.x, end_node.y, end_node.z])
        vec = (p2 - p1) / divs

        prev_id = new_start_id
        for i in range(1, divs):
            # 插值生成内部节点
            new_pos = p1 + vec * i
            curr_id = self.meshed_model.node_manager.get_or_create(*new_pos)
            node_sequence.append(curr_id)

            # 添加细分后的梁单元
            # 继承原始梁的属性
            self.meshed_model.beam_manager.add_or_update_beam(
                prev_id, curr_id, beam.beam_type, beam.section, beam.is_virtual, beam.on_face, beam.id
            )
            prev_id = curr_id

        # 连接最后一段
        self.meshed_model.beam_manager.add_or_update_beam(
            prev_id, new_end_id, beam.beam_type, beam.section, beam.is_virtual, beam.on_face, beam.id
        )
        node_sequence.append(new_end_id)

        self._beam_new_nodes[beam.id] = node_sequence

    def _mesh_single_plate(self, plate: Plate):
        # 获取板的四条边对应的梁信息
        # 节点顺序 n0 -> n1 -> n2 -> n3
        nodes = plate.nodes
        edges = [(nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[3]), (nodes[3], nodes[0])]

        edge_node_lists = []

        # 收集四条边的细分节点序列
        for n_start, n_end in edges:
            edge_def = self._get_edge_definition(n_start, n_end)
            edge_node_lists.append(self._build_edge_node_sequence(edge_def, n_start, n_end))

        # edge_node_lists 现在包含了 [Bottom, Right, Top, Left] 四边的节点序列
        # 进行 Coons Patch 插值生成内部节点和板单元
        self._create_coons_patch_mesh(plate, edge_node_lists)

    def _build_edge_node_sequence(self, edge_def: dict, n_start: int, n_end: int) -> List[int]:
        if edge_def["kind"] == "virtual":
            divs = self._get_edge_division(edge_def)
            return self._build_virtual_edge_sequence(n_start, n_end, divs)

        ordered_nodes = edge_def["nodes"]
        if ordered_nodes[0] != n_start:
            ordered_nodes = list(reversed(ordered_nodes))

        node_sequence = [self._node_map[ordered_nodes[0]]]
        for i in range(len(ordered_nodes) - 1):
            beam_id = self._find_beam_id(ordered_nodes[i], ordered_nodes[i + 1])
            seq = self._beam_new_nodes[beam_id]
            if seq[0] != self._node_map[ordered_nodes[i]]:
                seq = seq[::-1]
            node_sequence.extend(seq[1:])
        return node_sequence

    def _build_virtual_edge_sequence(self, n_start: int, n_end: int, divs: int) -> List[int]:
        start_node = self.original_model.node_manager.points[n_start]
        end_node = self.original_model.node_manager.points[n_end]
        p1 = np.array([start_node.x, start_node.y, start_node.z])
        p2 = np.array([end_node.x, end_node.y, end_node.z])
        vec = (p2 - p1) / divs

        sequence = [self._node_map[n_start]]
        for i in range(1, divs):
            pos = p1 + vec * i
            sequence.append(self.meshed_model.node_manager.get_or_create(*pos))
        sequence.append(self._node_map[n_end])
        return sequence

    def _create_coons_patch_mesh(self, original_plate: Plate, edge_nodes: List):
        # edge_nodes: [nodes_bottom, nodes_right, nodes_top, nodes_left]
        nodes_b = edge_nodes[0]  # U direction, v=0
        nodes_r = edge_nodes[1]  # V direction, u=1
        nodes_t = edge_nodes[2]
        nodes_l = edge_nodes[3]

        # 让我们使用归一化的网格坐标生成法
        n_u = len(nodes_b)  # 节点数
        n_v = len(nodes_r)

        # 网格节点矩阵
        grid_ids = np.zeros((n_v, n_u), dtype=int)

        # 填充边界
        # Bottom (Row 0)
        grid_ids[0, :] = nodes_b
        # Top (Row max) - 注意 nodes_t 是 n2->n3 (Right to Left)，需要反转以匹配 u increasing (Left to Right)
        # 只有当 nodes_t 是 n3->n2 时才是 Left to Right。
        # 原始 nodes: 0->1(B), 1->2(R), 2->3(T), 3->0(L).
        # 所以 T 是 2->3. 在 (u,v) 坐标系中，这是 u=1 -> u=0.
        # 为了插值，我们需要 3->2 (u=0 -> u=1).
        grid_ids[-1, :] = nodes_t[::-1]

        # Right (Col max) - nodes_r 是 1->2 (Bottom to Top). Matches V direction.
        grid_ids[:, -1] = nodes_r
        grid_ids[:, 0] = nodes_l[::-1]

        # 获取新模型的 NodeManager 用于坐标查询
        nm = self.meshed_model.node_manager

        # 内部节点插值
        for i in range(1, n_v - 1):  # v index
            for j in range(1, n_u - 1):  # u index
                # 归一化坐标
                u = j / (n_u - 1)
                v = i / (n_v - 1)

                # 边界点坐标
                P_u0 = nm.points[grid_ids[i, 0]].coordinates()  # Left boundary at v
                P_u1 = nm.points[grid_ids[i, -1]].coordinates()  # Right boundary at v
                P_0v = nm.points[grid_ids[0, j]].coordinates()  # Bottom boundary at u
                P_1v = nm.points[grid_ids[-1, j]].coordinates()  # Top boundary at u

                # 角点坐标
                P00 = nm.points[grid_ids[0, 0]].coordinates()
                P10 = nm.points[grid_ids[0, -1]].coordinates()
                P11 = nm.points[grid_ids[-1, -1]].coordinates()
                P01 = nm.points[grid_ids[-1, 0]].coordinates()

                # 双线性 Coons Patch 公式
                pos = (
                    (1 - u) * P_u0
                    + u * P_u1
                    + (1 - v) * P_0v
                    + v * P_1v
                    - ((1 - u) * (1 - v) * P00 + u * (1 - v) * P10 + u * v * P11 + (1 - u) * v * P01)
                )

                new_id = nm.get_or_create(pos[0], pos[1], pos[2])
                grid_ids[i, j] = new_id

        # 生成板单元 (Quad elements)
        for i in range(n_v - 1):
            for j in range(n_u - 1):
                # 逆时针顺序: (u,v) -> (u+1, v) -> (u+1, v+1) -> (u, v+1)
                # 对应 Grid: [i,j] -> [i, j+1] -> [i+1, j+1] -> [i+1, j]
                n1 = int(grid_ids[i, j])
                n2 = int(grid_ids[i, j + 1])
                n3 = int(grid_ids[i + 1, j + 1])
                n4 = int(grid_ids[i + 1, j])

                # 传递原始板的属性
                self.meshed_model.add_plate([n1, n2, n3, n4], original_plate.on_face, original_plate.panel_type)


if __name__ == "__main__":
    from src.structure.layout.params import StructuralParameters
    from src.structure.section_library import SectionLibrary
    from src.utils.model_utils import StructureVisualizer, create_structural_model, load_config

    p = StructuralParameters(
        length=8.8,
        width=4.2,
        height=3.8,
        main_beam_spacing=0.7,
        num_secondary_width=4,
        num_secondary_height=4,
        add_support_beam=True,
    )
    section_lib = SectionLibrary("src/data/steel_sections_large.json")
    model = StructuralModel(
        p,
        section_lib=section_lib,
    )
    mesh_generator = MeshGenerator(model, max_mesh_size=0.5)
    meshed_model = mesh_generator.generate()
