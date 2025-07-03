from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

# Ensuring imports are relative to the 'src' directory structure.
# This assumes that 'src' is part of PYTHONPATH or the execution context.
from ..structure.structural_model import Beam as ModelBeam
from ..structure.structural_model import Point as ModelPoint
from ..structure.structural_model import StructuralModel
from ..utils.util import is_close  # For comparing floating point numbers


class StructuralGraph:
    """
    Represents the building structure as a graph.
    In this representation (Type 1):
    - Graph Nodes: Correspond to the Point objects (joints) in the StructuralModel.
    - Graph Edges: Correspond to the Beam objects (structural members) in the StructuralModel.
    """

    def __init__(self, structural_model: StructuralModel):
        """
        Initializes the graph representation from a StructuralModel.

        Args:
            structural_model: The input StructuralModel instance.
        """
        self.model: StructuralModel = structural_model

        # Graph nodes are the Point objects from the model's node dictionary.
        # Key: node_id (int), Value: ModelPoint object
        self.nodes: Dict[int, ModelPoint] = self.model.points

        # Graph edges are the Beam objects from the model's beam dictionary.
        # Key: beam_id (int), Value: ModelBeam object
        self.edges: Dict[int, ModelBeam] = self.model.beams

        # Adjacency list representation of the graph.
        # Key: node_id (int)
        # Value: List of tuples, where each tuple is (neighbor_node_id, beam_id_connecting_them)
        self.adj: Dict[int, List[Tuple[int, int]]] = {node_id: [] for node_id in self.nodes.keys()}

        self._build_adjacency_list()

    def _build_adjacency_list(self):
        """
        Populates the adjacency list based on the beams in the structural model.
        Each beam connects two nodes, forming an edge in the graph.
        """
        for beam_id, beam_obj in self.edges.items():
            start_node_id = beam_obj.start_node
            end_node_id = beam_obj.end_node

            # Ensure both start and end nodes of the beam exist in the graph's node set.
            if start_node_id in self.nodes and end_node_id in self.nodes:
                # Add edge from start_node to end_node
                self.adj[start_node_id].append((end_node_id, beam_id))
                # Add edge from end_node to start_node (for undirected graph representation)
                self.adj[end_node_id].append((start_node_id, beam_id))
            else:
                # Log a warning if a beam refers to non-existent nodes.
                # This might indicate an inconsistency in the StructuralModel.
                print(
                    f"Warning: Beam ID {beam_id} refers to non-existent node(s). "
                    f"Start node: {start_node_id}, End node: {end_node_id}. "
                    "This edge will be skipped in the adjacency list."
                )

    def get_node_attributes(self, node_id: int) -> Optional[ModelPoint]:
        """
        Retrieves the attributes of a specific node in the graph.

        Args:
            node_id: The ID of the node.

        Returns:
            The ModelPoint object representing the node, or None if the node_id is not found.
        """
        return self.nodes.get(node_id)

    def get_edge_attributes(self, beam_id: int) -> Optional[ModelBeam]:
        """
        Retrieves the attributes of a specific edge (beam) in the graph.

        Args:
            beam_id: The ID of the beam (edge).

        Returns:
            The ModelBeam object representing the edge, or None if the beam_id is not found.
        """
        return self.edges.get(beam_id)

    def get_neighbors(self, node_id: int) -> List[Tuple[int, int]]:
        """
        Retrieves the neighbors of a specific node in the graph.

        Args:
            node_id: The ID of the node.

        Returns:
            A list of tuples, where each tuple is (neighbor_node_id, beam_id_connecting_to_neighbor).
            Returns an empty list if the node_id is not found or has no neighbors.
        """
        return self.adj.get(node_id, [])

    def get_all_nodes(self) -> List[ModelPoint]:
        """Returns a list of all node objects in the graph."""
        return list(self.nodes.values())

    def get_all_edges(self) -> List[ModelBeam]:
        """Returns a list of all edge (beam) objects in the graph."""
        return list(self.edges.values())

    def number_of_nodes(self) -> int:
        """Returns the total number of nodes in the graph."""
        return len(self.nodes)

    def number_of_edges(self) -> int:
        """
        Returns the total number of unique edges (beams) in the graph.
        """
        return len(self.edges)

    def __str__(self) -> str:
        return f"StructuralGraph with {self.number_of_nodes()} nodes and " f"{self.number_of_edges()} edges."

    def to_pyg_data(self) -> Data:
        """
        Converts the StructuralGraph to a PyTorch Geometric Data object.

        Node features:
            - x, y, z coordinates (3 features)
            - is_support (1.0 if z approx 0, else 0.0) (1 feature)

        Edge features (for each directed edge u -> v representing one side of a beam):
            - Start node coordinates (of u) (3 features)
            - End node coordinates (of v) (3 features)
            - Beam section index (integer) (1 feature)
            - Direction category (0: horizontal, 1: vertical, 2: diagonal) (1 feature)
        """
        if not self.nodes:
            return Data()  # Return empty Data object if no nodes

        # Node ID mapping from model's potentially sparse IDs to dense 0..N-1 PyG indices
        # Ensure a consistent order for node features and edge_index
        sorted_model_node_ids = sorted(list(self.nodes.keys()))
        model_node_id_to_pyg_idx = {model_id: i for i, model_id in enumerate(sorted_model_node_ids)}

        # 1. Node Features (x)
        node_features_list = []
        tolerance_z_support = 1e-6
        for model_node_id in sorted_model_node_ids:
            node = self.nodes[model_node_id]
            coords = [node.x, node.y, node.z]
            is_support_feature = [1.0 if is_close(node.z, 0.0, abs_tol=tolerance_z_support) else 0.0]
            node_features_list.append(coords + is_support_feature)

        pyg_x = torch.tensor(node_features_list, dtype=torch.float)

        # 2. Edge Index (edge_index) and Edge Attributes (edge_attr)
        edge_list = []
        edge_attr_list = []

        # Create a mapping for section names to integer indices
        section_names = sorted(list(self.model.sections.keys()))  # Consistent ordering
        section_name_to_idx = {name: i for i, name in enumerate(section_names)}

        tolerance_coord_diff = 1e-4  # For determining horizontal/vertical/diagonal

        # To handle undirected edges correctly if iterating self.adj
        # Simpler: iterate self.edges (beams) once

        for beam_id, beam in self.model.beams.items():
            start_model_node_id = beam.start_node
            end_model_node_id = beam.end_node

            if (
                start_model_node_id not in model_node_id_to_pyg_idx
                or end_model_node_id not in model_node_id_to_pyg_idx
            ):
                print(f"Warning: Beam {beam_id} connects to unknown node(s). Skipping.")
                continue

            u_pyg = model_node_id_to_pyg_idx[start_model_node_id]
            v_pyg = model_node_id_to_pyg_idx[end_model_node_id]

            node_u_obj = self.model.points[start_model_node_id]
            node_v_obj = self.model.points[end_model_node_id]

            coords_u = [node_u_obj.x, node_u_obj.y, node_u_obj.z]
            coords_v = [node_v_obj.x, node_v_obj.y, node_v_obj.z]

            # Section index
            section_idx = section_name_to_idx.get(beam.section, -1)  # Default to -1 if section name not found
            if section_idx == -1:
                print(f"Warning: Beam {beam_id} has unknown section '{beam.section}'. Using index -1.")

            # Direction category
            dx = abs(coords_u[0] - coords_v[0])
            dy = abs(coords_u[1] - coords_v[1])
            dz = abs(coords_u[2] - coords_v[2])

            direction_category = 2  # Default to diagonal
            if (
                is_close(dx, 0.0, abs_tol=tolerance_coord_diff)
                and is_close(dy, 0.0, abs_tol=tolerance_coord_diff)
                and not is_close(dz, 0.0, abs_tol=tolerance_coord_diff)
            ):
                direction_category = 1  # Vertical
            elif is_close(dz, 0.0, abs_tol=tolerance_coord_diff) and (
                not is_close(dx, 0.0, abs_tol=tolerance_coord_diff)
                or not is_close(dy, 0.0, abs_tol=tolerance_coord_diff)
            ):
                direction_category = 0  # Horizontal

            # Edge attributes for this beam
            # Features: start_coords (3), end_coords (3), section_idx (1), direction_cat (1)
            # For edge u -> v
            edge_feat_uv = coords_u + coords_v + [section_idx, direction_category]
            # For edge v -> u (features are symmetric in this definition)
            edge_feat_vu = coords_v + coords_u + [section_idx, direction_category]

            # Add edges for PyG (typically expects directed edges, so add both directions for an undirected beam)
            edge_list.append([u_pyg, v_pyg])
            edge_attr_list.append(edge_feat_uv)

            edge_list.append([v_pyg, u_pyg])
            edge_attr_list.append(edge_feat_vu)  # Using specific features for v->u edge

        if not edge_list:  # No edges
            pyg_edge_index = torch.empty((2, 0), dtype=torch.long)
            pyg_edge_attr = torch.empty((0, 8), dtype=torch.float)  # 8 edge features
        else:
            pyg_edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            pyg_edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

        # 3. Create PyG Data object
        data = Data(x=pyg_x, edge_index=pyg_edge_index, edge_attr=pyg_edge_attr)
        data.num_nodes = pyg_x.size(0)

        return data


class BeamAsNodeGraph:
    """
    Represents the building structure as a graph where beams are nodes.
    In this representation (Type 2):
    - Graph Nodes: Correspond to the Beam objects (structural members) in the StructuralModel.
    - Graph Edges: Represent a connection (shared Point) between two Beam objects.
    """

    def __init__(self, structural_model: StructuralModel):
        """
        Initializes the graph representation from a StructuralModel.

        Args:
            structural_model: The input StructuralModel instance.
        """
        self.model: StructuralModel = structural_model

        # Graph nodes are the Beam objects from the model's beam dictionary.
        # Key: beam_id (int), Value: ModelBeam object
        self.nodes: Dict[int, ModelBeam] = self.model.beams

        # Adjacency list representation of the graph.
        # Key: beam_id (acting as node_id)
        # Value: List of beam_ids that are connected to this beam.
        self.adj: Dict[int, List[int]] = defaultdict(list)
        self._build_adjacency_list()

    def _build_adjacency_list(self):
        """
        Populates the adjacency list.
        An edge exists between two beam-nodes if they share a common structural node (Point).
        """
        if not self.model.points:
            # No structural nodes, so no connections possible.
            return

        # 1. Create a map from structural node ID to a list of beam IDs incident to it.
        node_to_incident_beams: Dict[int, List[int]] = defaultdict(list)
        for beam_id, beam_obj in self.model.beams.items():
            if beam_obj.start_node in self.model.points:
                node_to_incident_beams[beam_obj.start_node].append(beam_id)
            if beam_obj.end_node in self.model.points:
                node_to_incident_beams[beam_obj.end_node].append(beam_id)

        # 2. Build the adjacency list for the beam graph.
        # For each structural node, all beams incident to it are connected to each other.
        for structural_node_id, incident_beams in node_to_incident_beams.items():
            if len(incident_beams) > 1:
                for i in range(len(incident_beams)):
                    for j in range(i + 1, len(incident_beams)):
                        beam1_id = incident_beams[i]
                        beam2_id = incident_beams[j]
                        # Add edges in both directions for an undirected graph
                        if beam2_id not in self.adj[beam1_id]:
                            self.adj[beam1_id].append(beam2_id)
                        if beam1_id not in self.adj[beam2_id]:
                            self.adj[beam2_id].append(beam1_id)

    def get_node_attributes(self, beam_id: int) -> Optional[ModelBeam]:
        """
        Retrieves the attributes of a specific node (beam) in the graph.

        Args:
            beam_id: The ID of the beam (which is the node_id in this graph).

        Returns:
            The ModelBeam object, or None if the beam_id is not found.
        """
        return self.nodes.get(beam_id)

    def get_neighbors(self, beam_id: int) -> List[int]:
        """
        Retrieves the neighbors of a specific beam-node in the graph.

        Args:
            beam_id: The ID of the beam-node.

        Returns:
            A list of beam_ids connected to the given beam_id.
            Returns an empty list if the beam_id is not found or has no connections.
        """
        return self.adj.get(beam_id, [])

    def get_all_nodes(self) -> List[ModelBeam]:
        """Returns a list of all node (beam) objects in the graph."""
        return list(self.nodes.values())

    def number_of_nodes(self) -> int:
        """Returns the total number of nodes (beams) in the graph."""
        return len(self.nodes)

    def number_of_edges(self) -> int:
        """
        Returns the total number of unique edges (connections between beams) in the graph.
        Since the graph is undirected, each edge (b1, b2) is counted once.
        """
        count = 0
        visited_edges = set()
        for beam_id, neighbors in self.adj.items():
            for neighbor_id in neighbors:
                # To avoid double counting and self-loops if any were accidentally added
                if tuple(sorted((beam_id, neighbor_id))) not in visited_edges:
                    count += 1
                    visited_edges.add(tuple(sorted((beam_id, neighbor_id))))
        return count

    def __str__(self) -> str:
        return (
            f"BeamAsNodeGraph with {self.number_of_nodes()} nodes (beams) and "
            f"{self.number_of_edges()} edges (connections)."
        )

    def to_pyg_data(self, env_info: Dict) -> Data:
        """
        Converts the BeamAsNodeGraph to a PyTorch Geometric Data object.

        Args:
            env_info: A dictionary containing the environment information.

        Node features (for each beam):
            - Beam Type index (integer) (1 feature)
            - Beam Section index (integer) (1 feature)
            - Beam length (float) (1 feature)
            - Normalized direction vector (dx, dy, dz) (3 features)
            - Centroid coordinates (cx, cy, cz) (3 features)

        Edge features (for each connection between two beams):
            - Coordinates of the shared structural joint (x, y, z) (3 features)
        """
        if not self.nodes:  # self.nodes in BeamAsNodeGraph are the beams
            return Data()

        # 按梁类型分层排序：框架梁 -> 主梁 -> 次梁 -> 支撑梁
        sorted_beam_ids = sorted(self.nodes.keys(), key=lambda bid: self.nodes[bid].beam_type.value)

        # 记录各类型梁的索引范围
        type_ranges = {}
        current_idx = 0
        for beam_type in [0, 1, 2, 3]:  # BeamType枚举值对应: 0-框架梁,1-主梁,2-次梁,3-支撑梁
            type_ids = [bid for bid in sorted_beam_ids if self.nodes[bid].beam_type.value == beam_type]
            if type_ids:
                type_ranges[beam_type] = (current_idx, current_idx + len(type_ids) - 1)
                current_idx += len(type_ids)

        beam_id_to_pyg_idx = {beam_id: i for i, beam_id in enumerate(sorted_beam_ids)}

        # 1. Node Features (x) - for each beam
        node_features_list = []

        for beam_id in sorted_beam_ids:
            beam = self.nodes[beam_id]
            start_node_obj = self.model.points[beam.start_node]
            end_node_obj = self.model.points[beam.end_node]

            # Beam Type Index
            beam_type_idx = beam.beam_type.value

            # Beam Section Index in SectionLibrary
            beam_section_idx = self.model.sections[beam.section].idx

            # Beam Length
            length = start_node_obj.distance_to(end_node_obj)

            # Direction Vector (normalized)
            direction_vec = end_node_obj.coordinates() - start_node_obj.coordinates()
            norm = np.linalg.norm(direction_vec)
            if norm > 1e-9:  # Avoid division by zero for zero-length beams (should not happen)
                direction_vec_normalized = direction_vec / norm
            else:
                direction_vec_normalized = np.array([0.0, 0.0, 0.0])

            # Centroid Coordinates
            centroid = (start_node_obj.coordinates() + end_node_obj.coordinates()) / 2.0

            # response
            max_disp_abs = beam.response.max_disp_abs
            max_stress = beam.response.max_stress

            # section property
            section_property = self.model.sections[beam.section]
            Ix = section_property.Ix
            Iy = section_property.Iy
            area = section_property.area

            features = (
                [beam_type_idx, beam_section_idx, length]
                + list(direction_vec_normalized)
                + list(centroid)
                + [max_disp_abs, max_stress]
                + [Ix, Iy, area]
                + beam.start_node_load.tolist()
                + beam.end_node_load.tolist()
                + [env_info["frame_length"]]
            )
            node_features_list.append(features)

        pyg_x = torch.tensor(node_features_list, dtype=torch.float)

        # 2. Edge Index (edge_index) and Edge Attributes (edge_attr)
        edge_list = []
        edge_attr_list = []

        # Re-iterate through shared structural nodes to find connections for edge attributes
        node_to_incident_beams: Dict[int, List[int]] = defaultdict(list)
        for beam_id_iter, beam_obj_iter in self.model.beams.items():
            if beam_obj_iter.start_node in self.model.points:
                node_to_incident_beams[beam_obj_iter.start_node].append(beam_id_iter)
            if beam_obj_iter.end_node in self.model.points:
                node_to_incident_beams[beam_obj_iter.end_node].append(beam_id_iter)

        # 生成边时记录连接类型
        edge_types = []
        for structural_node_id, incident_beams in node_to_incident_beams.items():
            if len(incident_beams) > 1:
                shared_point_coords = list(self.model.points[structural_node_id].coordinates())
                for i in range(len(incident_beams)):
                    for j in range(i + 1, len(incident_beams)):
                        beam1_id = incident_beams[i]
                        beam2_id = incident_beams[j]

                        # 记录连接类型
                        type1 = self.nodes[beam1_id].beam_type.value
                        type2 = self.nodes[beam2_id].beam_type.value
                        edge_types.extend([(type1, type2), (type2, type1)])

                        # Get PyG indices for these beams
                        u_pyg = beam_id_to_pyg_idx[beam1_id]
                        v_pyg = beam_id_to_pyg_idx[beam2_id]

                        # Add edges in both directions for PyG
                        # Edge from beam1 to beam2
                        edge_list.append([u_pyg, v_pyg])
                        edge_attr_list.append(shared_point_coords)

                        # Edge from beam2 to beam1
                        edge_list.append([v_pyg, u_pyg])
                        edge_attr_list.append(shared_point_coords)  # Edge attribute is symmetric

        if not edge_list:
            pyg_edge_index = torch.empty((2, 0), dtype=torch.long)
            pyg_edge_attr = torch.empty((0, 3), dtype=torch.float)  # 3 edge features (shared point coords)
        else:
            pyg_edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            pyg_edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

        data = Data(x=pyg_x, edge_index=pyg_edge_index, edge_attr=pyg_edge_attr)
        data.num_nodes = pyg_x.size(0)
        data.node_type_ranges = type_ranges  # 各类型梁的索引范围
        data.edge_types = torch.tensor(edge_types, dtype=torch.long)  # 添加连接类型

        return data


if __name__ == "__main__":
    from src.utils.model_utils import create_structural_model, load_config

    print("Testing StructuralGraph to PyG Data conversion...")
    config_path = "config.yaml"
    config = load_config(config_path)
    model = create_structural_model(config)

    # Test StructuralGraph (joints as nodes, beams as edges)
    print("\n--- Testing StructuralGraph (joints as nodes) ---")
    graph_type1 = StructuralGraph(model)
    print(f"Created: {graph_type1}")
    pyg_data_type1 = graph_type1.to_pyg_data()
    print("PyG Data (Type 1):")
    print(pyg_data_type1)
    if pyg_data_type1:
        print(f"  Node features shape: {pyg_data_type1.x.shape if pyg_data_type1.x is not None else 'N/A'}")
        print(
            f"  Edge index shape: {pyg_data_type1.edge_index.shape if pyg_data_type1.edge_index is not None else 'N/A'}"
        )
        print(
            f"  Edge features shape: {pyg_data_type1.edge_attr.shape if pyg_data_type1.edge_attr is not None else 'N/A'}"
        )

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
