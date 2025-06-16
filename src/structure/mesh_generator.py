from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List

import numpy as np

if TYPE_CHECKING:
    from src.structure.structural_model import Beam, Plate, Point, StructuralModel


class MeshParameters:
    """网格划分参数"""

    def __init__(self, max_grid_size: float):
        """
        初始化网格参数.

        Args:
            max_grid_size (float): 网格单元的最大允许边长.
        """
        self.max_grid_size = max_grid_size


class MeshGenerator:
    """
    结构模型网格生成器.

    根据指定的网格参数，对 StructuralModel 中的梁和板进行细分，
    生成一个新的、网格更精细的 StructuralModel.
    """

    def __init__(self, model: StructuralModel, mesh_params: MeshParameters):
        """
        初始化网格生成器.

        Args:
            model (StructuralModel): 原始结构模型.
            mesh_params (MeshParameters): 网格划分参数.
        """
        self.original_model = model
        self.mesh_params = mesh_params

        # 导入相关类以避免循环导入
        from src.structure.structural_model import Plate as PlateClass
        from src.structure.structural_model import StructuralModel as SM

        self.meshed_model = SM(model.params, model.section_library)
        self.PlateClass = PlateClass

        # 记录每个梁上的节点, 避免板单元重复计算边界网格
        self._nodes_on_beam: Dict[int, List[int]] = {}
        self._old_to_new_node_id: Dict[int, int] = {}
        self._beams_on_plate: Dict[int, List[List[Beam]]] = {}

    def generate_mesh(self) -> StructuralModel:
        """
        执行网格划分并返回新的精细化模型.

        Returns:
            StructuralModel: 网格划分后的新结构模型.
        """
        print("Pre-calculating beams on plate...")
        for plate in self.original_model.plates.values():
            boundary_beams = [self.original_model.beams[beam_id] for beam_id in plate.boundary_beams]
            self._beams_on_plate[plate.id] = self._get_beams_per_edge(plate, boundary_beams)

        print("Pre-calculating beam divisions for plate compatibility...")
        beam_divisions = self._calculate_beam_divisions()

        print("开始划分梁单元...")
        self._mesh_beams(beam_divisions)
        print("梁单元划分完成。")

        print("开始划分板单元...")
        self._mesh_plates()
        print("板单元划分完成。")

        return self.meshed_model

    def _get_beams_per_edge(self, plate: Plate, boundary_beams: List[Beam]) -> List[List[Beam]]:
        """获取构成板的四条逻辑边上的梁序列"""
        all_edge_beams: List[List[Beam]] = []
        original_points = self.original_model.points
        corner_node_ids = plate.nodes

        for i in range(4):
            start_corner_id = corner_node_ids[i]
            end_corner_id = corner_node_ids[(i + 1) % 4]

            if start_corner_id == end_corner_id:
                all_edge_beams.append([])
                continue

            start_corner_pt = original_points[start_corner_id]
            end_corner_pt = original_points[end_corner_id]

            on_segment_beams: List[Beam] = [
                beam
                for beam in boundary_beams
                if self._is_on_segment(original_points[beam.start_node], start_corner_pt, end_corner_pt)
                and self._is_on_segment(original_points[beam.end_node], start_corner_pt, end_corner_pt)
            ]

            sorted_beams: List[Beam] = []
            curr_node_id = start_corner_id
            while curr_node_id != end_corner_id:
                found_next = False
                for beam in on_segment_beams:
                    if beam in sorted_beams:
                        continue
                    if beam.start_node == curr_node_id:
                        sorted_beams.append(beam)
                        curr_node_id = beam.end_node
                        found_next = True
                        break
                    elif beam.end_node == curr_node_id:
                        sorted_beams.append(beam)
                        curr_node_id = beam.start_node
                        found_next = True
                        break
                if not found_next:
                    raise RuntimeError(
                        f"Could not form a continuous path on edge from {start_corner_id} to {end_corner_id}"
                    )
            all_edge_beams.append(sorted_beams)
        return all_edge_beams

    def _increase_edge_divisions(
        self, beams: List[Beam], target_total_divs: int, beam_divisions: Dict[int, int]
    ):
        """按比例增加构成逻辑边的梁的划分数"""
        current_total_divs = sum(beam_divisions.get(b.id, 1) for b in beams)
        delta = target_total_divs - current_total_divs
        if delta <= 0 or not beams:
            return

        beam_lengths = [b.length(self.original_model.points) for b in beams]
        total_length = sum(beam_lengths)

        if total_length < 1e-6:
            beam_divisions[beams[0].id] += delta
            return

        proportions = [l / total_length * delta for l in beam_lengths]
        int_adds = [int(p) for p in proportions]
        remainders = [p - i for p, i in zip(proportions, int_adds)]

        to_distribute = delta - sum(int_adds)
        indices = sorted(range(len(remainders)), key=lambda k: remainders[k], reverse=True)

        for i in range(to_distribute):
            int_adds[indices[i]] += 1

        for beam, adiv in zip(beams, int_adds):
            beam_divisions[beam.id] += adiv

    def _calculate_beam_divisions(self) -> Dict[int, int]:
        """通过迭代松弛法，为所有梁计算一个满足所有板约束的划分方案"""
        beam_divisions: Dict[int, int] = {}
        for beam in self.original_model.beams.values():
            length = beam.length(self.original_model.points)
            min_divs = math.ceil(length / self.mesh_params.max_grid_size) if length > 0 else 1
            beam_divisions[beam.id] = min_divs

        for i in range(10):  # Convergence loop with safety break
            changed_in_iteration = False
            for plate in self.original_model.plates.values():
                if len(plate.nodes) != 4:
                    continue

                edges = self._beams_on_plate[plate.id]

                # U-direction (edge 0 vs edge 2)
                div_u0 = sum(beam_divisions.get(b.id, 1) for b in edges[0])
                div_u2 = sum(beam_divisions.get(b.id, 1) for b in edges[2])
                target_div_u = max(div_u0, div_u2)

                if target_div_u > div_u0:
                    self._increase_edge_divisions(edges[0], target_div_u, beam_divisions)
                    changed_in_iteration = True
                if target_div_u > div_u2:
                    self._increase_edge_divisions(edges[2], target_div_u, beam_divisions)
                    changed_in_iteration = True

                # V-direction (edge 1 vs edge 3)
                div_v1 = sum(beam_divisions.get(b.id, 1) for b in edges[1])
                div_v3 = sum(beam_divisions.get(b.id, 1) for b in edges[3])
                target_div_v = max(div_v1, div_v3)

                if target_div_v > div_v1:
                    self._increase_edge_divisions(edges[1], target_div_v, beam_divisions)
                    changed_in_iteration = True
                if target_div_v > div_v3:
                    self._increase_edge_divisions(edges[3], target_div_v, beam_divisions)
                    changed_in_iteration = True

            if not changed_in_iteration:
                print(f"Beam division calculation converged after {i + 1} iterations.")
                break
        else:
            print("Warning: Beam division calculation did not converge after 10 iterations.")

        return beam_divisions

    def _mesh_beams(self, beam_divisions: Dict[int, int]):
        """对所有梁单元进行网格划分，优先使用预计算的划分数"""
        for beam in self.original_model.beams.values():
            start_node = self.original_model.points[beam.start_node]
            end_node = self.original_model.points[beam.end_node]
            length = start_node.distance_to(end_node)

            num_segments = beam_divisions.get(beam.id, 1)

            # --- Start of new node creation logic ---
            start_id = self._old_to_new_node_id.get(start_node.id)
            if start_id is None:
                start_id = self.meshed_model.add_node(start_node.x, start_node.y, start_node.z)
                self._old_to_new_node_id[start_node.id] = start_id

            end_id = self._old_to_new_node_id.get(end_node.id)
            if end_id is None:
                end_id = self.meshed_model.add_node(end_node.x, end_node.y, end_node.z)
                self._old_to_new_node_id[end_node.id] = end_id

            if length == 0:
                self._nodes_on_beam[beam.id] = [start_id]
                continue
            # --- End of new node creation logic ---

            node_ids = [start_id]
            if num_segments > 1:
                start_coords = start_node.coordinates()
                end_coords = end_node.coordinates()
                direction_vec = (end_coords - start_coords) / num_segments
                for i in range(1, num_segments):
                    new_coords = start_coords + i * direction_vec
                    node_id = self.meshed_model.add_node(new_coords[0], new_coords[1], new_coords[2])
                    node_ids.append(node_id)
            node_ids.append(end_id)

            for i in range(len(node_ids) - 1):
                self._add_beam(node_ids[i], node_ids[i + 1], beam)

            self._nodes_on_beam[beam.id] = node_ids

    def _mesh_plates(self):
        """对所有板单元进行网格划分"""
        for plate in self.original_model.plates.values():
            if len(plate.nodes) != 4:
                continue

            edge_nodes = self._get_edge_nodes(plate)
            self._create_plate_mesh_from_edges(plate, edge_nodes)

    def _is_on_segment(
        self, p_to_check: Point, p_start: Point, p_end: Point, tolerance: float = 1e-6
    ) -> bool:
        """检查一个点是否位于另外两个点定义的线段上（包括端点）"""
        vec_seg = p_end.coordinates() - p_start.coordinates()
        vec_check = p_to_check.coordinates() - p_start.coordinates()
        seg_len_sq = np.dot(vec_seg, vec_seg)
        if seg_len_sq < tolerance**2:
            return np.dot(vec_check, vec_check) < tolerance**2
        cross_product = np.cross(vec_seg, vec_check)
        if np.linalg.norm(cross_product) > tolerance:
            return False
        dot_product = np.dot(vec_check, vec_seg)
        return -tolerance < dot_product < seg_len_sq + tolerance

    def _get_edge_nodes(self, plate: Plate) -> List[List[int]]:
        """获取板的每条边上的所有节点"""
        all_edge_nodes = []
        all_edge_beams = self._beams_on_plate[plate.id]

        for i in range(4):
            sorted_beams = all_edge_beams[i]
            start_corner_id = plate.nodes[i]

            if not sorted_beams:
                new_node_id = self._old_to_new_node_id[start_corner_id]
                all_edge_nodes.append([new_node_id])
                continue

            path_nodes = []

            # The starting node for the current path segment in the *new* model.
            new_curr_node_id = self._old_to_new_node_id[start_corner_id]

            for beam in sorted_beams:
                meshed_nodes = self._nodes_on_beam[beam.id]

                if not meshed_nodes:
                    continue

                # Align the beam's meshed nodes with the path direction
                if meshed_nodes[0] == new_curr_node_id:
                    aligned_nodes = meshed_nodes
                else:
                    # Check if the reversed direction matches
                    if meshed_nodes[-1] != new_curr_node_id:
                        raise RuntimeError(
                            f"Beam {beam.id} doesn't connect to the path at node {new_curr_node_id}"
                        )
                    aligned_nodes = list(reversed(meshed_nodes))

                # Add nodes to the path
                if not path_nodes:
                    path_nodes.extend(aligned_nodes)
                else:
                    path_nodes.extend(aligned_nodes[1:])

                new_curr_node_id = path_nodes[-1]

            all_edge_nodes.append(path_nodes)
        return all_edge_nodes

    def _create_plate_mesh_from_edges(self, original_plate: Plate, edge_nodes: List[List[int]]):
        """根据边界节点创建板的网格（修正版）"""
        if any(not edge for edge in edge_nodes):
            print(f"Warning: Plate {original_plate.id} has a zero-length edge. Skipping mesh generation.")
            return

        n_nodes_u = len(edge_nodes[0])
        n_nodes_v = len(edge_nodes[1])

        if len(edge_nodes[2]) != n_nodes_u or len(edge_nodes[3]) != n_nodes_v:
            # This should no longer happen with the new calculation method, but kept as a safeguard.
            print(
                f"Error: Plate {original_plate.id} opposite edges have non-matching node counts after meshing."
                f" U-dir: {n_nodes_u} vs {len(edge_nodes[2])}, V-dir: {n_nodes_v} vs {len(edge_nodes[3])}"
            )
            return

        node_grid = np.full((n_nodes_u, n_nodes_v), -1, dtype=int)
        meshed_points = self.meshed_model.points

        node_grid[:, 0] = edge_nodes[0]
        node_grid[:, -1] = edge_nodes[2][::-1]
        node_grid[0, :] = edge_nodes[3][::-1]
        node_grid[-1, :] = edge_nodes[1]

        P00 = meshed_points[node_grid[0, 0]].coordinates()
        P01 = meshed_points[node_grid[0, -1]].coordinates()
        P10 = meshed_points[node_grid[-1, 0]].coordinates()
        P11 = meshed_points[node_grid[-1, -1]].coordinates()

        for i in range(n_nodes_u):
            for j in range(n_nodes_v):
                if i == 0 or i == n_nodes_u - 1 or j == 0 or j == n_nodes_v - 1:
                    continue
                u = i / (n_nodes_u - 1)
                v = j / (n_nodes_v - 1)
                A = meshed_points[node_grid[i, 0]].coordinates()
                C = meshed_points[node_grid[i, -1]].coordinates()
                B = meshed_points[node_grid[0, j]].coordinates()
                D = meshed_points[node_grid[-1, j]].coordinates()

                term1 = (1 - v) * np.array(A)
                term2 = v * np.array(C)
                term3 = (1 - u) * np.array(B)
                term4 = u * np.array(D)
                corner_term = (
                    (1 - u) * (1 - v) * np.array(P00)
                    + (1 - u) * v * np.array(P01)
                    + u * (1 - v) * np.array(P10)
                    + u * v * np.array(P11)
                )
                point_coord = (term1 + term2 + term3 + term4 - corner_term).tolist()
                node_id = self.meshed_model.add_node(point_coord[0], point_coord[1], point_coord[2])
                node_grid[i, j] = node_id

        for i in range(n_nodes_u - 1):
            for j in range(n_nodes_v - 1):
                node_ids = np.array(
                    [
                        node_grid[i, j],
                        node_grid[i + 1, j],
                        node_grid[i + 1, j + 1],
                        node_grid[i, j + 1],
                    ]
                )
                self._add_plate(node_ids.tolist(), original_plate)

    def _add_beam(self, start_id: int, end_id: int, original_beam: Beam) -> int:
        """向新模型中添加梁"""
        from src.structure.structural_model import Beam as BeamClass

        new_beam_id = self.meshed_model.beam_counter
        self.meshed_model.beams[new_beam_id] = BeamClass(
            id=new_beam_id,
            start_node=start_id,
            end_node=end_id,
            beam_type=original_beam.beam_type,
            section=original_beam.section,
            on_face=original_beam.on_face,
        )
        self.meshed_model.beam_counter += 1
        return new_beam_id

    def _add_plate(self, node_ids: list[int], original_plate: Plate) -> int:
        """向新模型中添加板"""
        new_plate_id = self.meshed_model.plate_counter
        self.meshed_model.plates[new_plate_id] = self.PlateClass(
            id=new_plate_id,
            nodes=node_ids,
            on_face=original_plate.on_face,
            thickness=original_plate.thickness,
            boundary_beams=[],
        )
        self.meshed_model.plate_counter += 1
        return new_plate_id


if __name__ == "__main__":
    from src.utils.model_utils import StructureVisualizer, create_structural_model, load_config

    model = create_structural_model(load_config("config.yaml"))
    mesh_params = MeshParameters(max_grid_size=0.1)
    mesh_generator = MeshGenerator(model, mesh_params)
    meshed_model = mesh_generator.generate_mesh()
    visualizer = StructureVisualizer(meshed_model)
    visualizer.show_all_views(False)
