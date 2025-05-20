from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# Ensuring imports are relative to the 'src' directory structure.
# This assumes that 'src' is part of PYTHONPATH or the execution context.
from ..structure.structural_model import Beam as ModelBeam
from ..structure.structural_model import Point as ModelPoint
from ..structure.structural_model import StructuralModel


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
        self.nodes: Dict[int, ModelPoint] = self.model.nodes

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
        if not self.model.nodes:
            # No structural nodes, so no connections possible.
            return

        # 1. Create a map from structural node ID to a list of beam IDs incident to it.
        node_to_incident_beams: Dict[int, List[int]] = defaultdict(list)
        for beam_id, beam_obj in self.model.beams.items():
            if beam_obj.start_node in self.model.nodes:
                node_to_incident_beams[beam_obj.start_node].append(beam_id)
            if beam_obj.end_node in self.model.nodes:
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
