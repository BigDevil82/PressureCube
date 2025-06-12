from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import subgraph

# Assuming BeamType enum values from your structural_model correspond to:
# FRAME: 0
# MAIN: 1
# SECONDARY: 2
# BRACE: 3
# Pooling order: BRACE (3) -> SECONDARY (2) -> MAIN (1). FRAME (0) is the most aggregated.


class HEBG(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_beam_types: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_beam_types = num_beam_types  # FRAME, MAIN, SECONDARY, BRACE

        # Initial GCN layers (before first pool)
        self.initial_conv1 = GCNConv(in_channels, hidden_channels)
        self.initial_conv2 = GCNConv(hidden_channels, hidden_channels)

        # --- Encoder (Pooling Layers) ---
        # Ops for pooling Braces (type 3)
        self.gcn_pool_brace_N1 = GCNConv(hidden_channels, hidden_channels)
        self.gcn_pool_brace_N2 = GCNConv(hidden_channels, hidden_channels)

        # Ops for pooling Secondary Beams (type 2)
        self.gcn_pool_secondary_N1 = GCNConv(hidden_channels, hidden_channels)
        self.gcn_pool_secondary_N2 = GCNConv(hidden_channels, hidden_channels)

        # Ops for pooling Main Beams (type 1)
        self.gcn_pool_main_N1 = GCNConv(hidden_channels, hidden_channels)
        self.gcn_pool_main_N2 = GCNConv(hidden_channels, hidden_channels)

        # --- Bottleneck ---
        self.bottleneck_conv1 = GCNConv(hidden_channels, hidden_channels)
        self.bottleneck_conv2 = GCNConv(hidden_channels, hidden_channels)

        # --- Decoder (Unpooling Layers) ---
        # Ops for unpooling Main Beams (type 1)
        self.gcn_unpool_main = GCNConv(hidden_channels, hidden_channels)  # Operates on coarse graph (FRAME)

        # Ops for unpooling Secondary Beams (type 2)
        self.gcn_unpool_secondary = GCNConv(
            hidden_channels, hidden_channels
        )  # Operates on coarse graph (FRAME+MAIN)

        # Ops for unpooling Braces (type 3)
        self.gcn_unpool_brace = GCNConv(
            hidden_channels, hidden_channels
        )  # Operates on coarse graph (FRAME+MAIN+SECONDARY)

        # Final GCN layers
        self.final_conv1 = GCNConv(hidden_channels, hidden_channels)
        self.final_conv2 = GCNConv(hidden_channels, out_channels)

    def _filter_graph(
        self, x_dense_current, edge_index_current, nodes_to_keep_mask_current, edge_attr_current=None
    ):
        """
        Filters a dense graph (x_dense_current, edge_index_current) based on nodes_to_keep_mask_current.
        x_dense_current: Features for the current active set of nodes (dense).
        edge_index_current: Edge index for x_dense_current (relabelled).
        nodes_to_keep_mask_current: Boolean mask for x_dense_current indicating which nodes to keep.
        Returns dense features, relabelled edge_index, and filtered edge_attr for the kept nodes.
        """
        if nodes_to_keep_mask_current.sum() == 0:
            return (
                torch.empty(0, x_dense_current.size(1), device=x_dense_current.device),
                torch.empty(2, 0, dtype=torch.long, device=edge_index_current.device),
                (
                    torch.empty(
                        0,
                        (
                            edge_attr_current.size(1)
                            if edge_attr_current is not None and edge_attr_current.numel() > 0
                            else 0
                        ),
                        device=x_dense_current.device,
                    )
                    if edge_attr_current is not None
                    else None
                ),
            )

        x_filtered_dense = x_dense_current[nodes_to_keep_mask_current]

        edge_index_filtered_relabelled, edge_attr_filtered = subgraph(
            subset=nodes_to_keep_mask_current,
            edge_index=edge_index_current,
            edge_attr=edge_attr_current,
            relabel_nodes=True,
            num_nodes=x_dense_current.size(0),
        )
        return x_filtered_dense, edge_index_filtered_relabelled, edge_attr_filtered

    def get_node_masks_by_type(self, data, num_total_nodes):
        node_masks = {}
        for type_val in range(self.num_beam_types):
            node_masks[type_val] = torch.zeros(num_total_nodes, dtype=torch.bool, device=data.x.device)

        if hasattr(data, "num_graphs") and data.num_graphs > 1:  # Check if it's a Batch object
            # raise NotImplementedError("Batch objects are not supported yet.")
            # It's a Batch object, iterate through individual graphs
            # We need to know the starting index of each graph in the batch
            # data.ptr gives [0, num_nodes_graph1, num_nodes_graph1+num_nodes_graph2, ...]
            current_node_offset = 0
            for i in range(1):
                # data.ptr gives the starting index for each graph in the batch.
                # Example: data_list = [graph0, graph1], graph0 has N0 nodes, graph1 has N1 nodes.
                # data.ptr = [0, N0, N0+N1]
                num_nodes_in_this_graph = data.ptr[i + 1] - data.ptr[i]
                original_graph_data = data.to_data_list()[i]  # Get the original Data object

                if hasattr(original_graph_data, "node_type_ranges") and original_graph_data.node_type_ranges:
                    for type_val_graph, (
                        start_idx_graph,
                        end_idx_graph,
                    ) in original_graph_data.node_type_ranges.items():
                        if isinstance(type_val_graph, int) and 0 <= type_val_graph < self.num_beam_types:
                            # Apply offset for the current graph in the batch
                            batch_start_idx = data.ptr[i] + start_idx_graph
                            batch_end_idx = data.ptr[i] + end_idx_graph
                            node_masks[type_val_graph][batch_start_idx : batch_end_idx + 1] = True
                current_node_offset += num_nodes_in_this_graph

        elif hasattr(data, "node_type_ranges") and data.node_type_ranges:  # Single Data object
            for type_val, (start_idx, end_idx) in data.node_type_ranges.items():
                if isinstance(type_val, int) and 0 <= type_val < self.num_beam_types:
                    node_masks[type_val][start_idx : end_idx + 1] = True
        # else: No node_type_ranges found on the data object or it's an unhandled batch structure
        # print("Warning: node_type_ranges not found on data object or unhandled batch structure in get_node_masks_by_type")

        return node_masks

    def forward(self, data: Data) -> torch.Tensor:
        x_orig, edge_index_orig, edge_attr_orig = data.x, data.edge_index, data.edge_attr
        # patch
        edge_attr_orig = None

        if edge_attr_orig is not None and edge_attr_orig.ndim == 1:
            edge_attr_orig = edge_attr_orig.unsqueeze(-1)

        num_total_nodes = x_orig.size(0)
        node_masks_by_type_global = self.get_node_masks_by_type(data, num_total_nodes)

        # --- Initial Feature Extraction ---
        x_current = F.relu(self.initial_conv1(x_orig, edge_index_orig, edge_attr_orig))
        x_current = F.relu(self.initial_conv2(x_current, edge_index_orig, edge_attr_orig))

        # `map_current_to_original` maps rows in `x_current` (dense) to original node indices
        map_current_to_original = torch.arange(num_total_nodes, device=x_orig.device)
        ei_current = edge_index_orig
        ea_current = edge_attr_orig

        skips_data = {}  # Stores {"N1_skip_features": tensor, "map_to_original": tensor}

        # --- Encoder ---
        # Pool Braces (type 3)
        type_to_pool = 3
        is_type_present_in_current = node_masks_by_type_global[type_to_pool][map_current_to_original]
        if is_type_present_in_current.any():
            N1 = torch.sigmoid(self.gcn_pool_brace_N1(x_current, ei_current, ea_current))
            N2 = torch.sigmoid(self.gcn_pool_brace_N2(x_current, ei_current, ea_current)) * N1
            x_processed = N1 + N2
            skips_data[type_to_pool] = {
                "N1_skip_features": N1.clone(),
                "map_to_original": map_current_to_original.clone(),
            }

            nodes_to_keep_mask = ~is_type_present_in_current
            x_current, ei_current, ea_current = self._filter_graph(
                x_processed, ei_current, nodes_to_keep_mask, ea_current
            )
            map_current_to_original = map_current_to_original[nodes_to_keep_mask]
        else:  # Type not present or already pooled
            skips_data[type_to_pool] = {
                "N1_skip_features": x_current.clone(),
                "map_to_original": map_current_to_original.clone(),
            }

        # Pool Secondary (type 2)
        type_to_pool = 2
        is_type_present_in_current = node_masks_by_type_global[type_to_pool][map_current_to_original]
        if is_type_present_in_current.any():
            N1 = torch.sigmoid(self.gcn_pool_secondary_N1(x_current, ei_current, ea_current))
            N2 = torch.sigmoid(self.gcn_pool_secondary_N2(x_current, ei_current, ea_current)) * N1
            x_processed = N1 + N2
            skips_data[type_to_pool] = {
                "N1_skip_features": N1.clone(),
                "map_to_original": map_current_to_original.clone(),
            }

            nodes_to_keep_mask = ~is_type_present_in_current
            x_current, ei_current, ea_current = self._filter_graph(
                x_processed, ei_current, nodes_to_keep_mask, ea_current
            )
            map_current_to_original = map_current_to_original[nodes_to_keep_mask]
        else:
            skips_data[type_to_pool] = {
                "N1_skip_features": x_current.clone(),
                "map_to_original": map_current_to_original.clone(),
            }

        # Pool Main (type 1)
        type_to_pool = 1
        is_type_present_in_current = node_masks_by_type_global[type_to_pool][map_current_to_original]
        if is_type_present_in_current.any():
            N1 = torch.sigmoid(self.gcn_pool_main_N1(x_current, ei_current, ea_current))
            N2 = torch.sigmoid(self.gcn_pool_main_N2(x_current, ei_current, ea_current)) * N1
            x_processed = N1 + N2
            skips_data[type_to_pool] = {
                "N1_skip_features": N1.clone(),
                "map_to_original": map_current_to_original.clone(),
            }

            nodes_to_keep_mask = ~is_type_present_in_current  # Keep only FRAME (type 0)
            x_current, ei_current, ea_current = self._filter_graph(
                x_processed, ei_current, nodes_to_keep_mask, ea_current
            )
            map_current_to_original = map_current_to_original[nodes_to_keep_mask]
        else:
            skips_data[type_to_pool] = {
                "N1_skip_features": x_current.clone(),
                "map_to_original": map_current_to_original.clone(),
            }

        # --- Bottleneck --- (x_current has features for FRAME nodes)
        x_current = F.relu(self.bottleneck_conv1(x_current, ei_current, ea_current))
        x_current = F.relu(self.bottleneck_conv2(x_current, ei_current, ea_current))

        # --- Decoder ---
        # Unpool Main (type 1)
        type_to_unpool = 1
        skip_info = skips_data[type_to_unpool]
        N1_skip_level = skip_info[
            "N1_skip_features"
        ]  # Dense features for nodes at this skip level (e.g., FRAME+MAIN)
        map_skip_level_to_original = skip_info["map_to_original"]  # Original indices for N1_skip_level

        # GCN on current coarse graph (e.g., FRAME features from bottleneck)
        x_coarse_processed = F.relu(self.gcn_unpool_main(x_current, ei_current, ea_current))

        # Prepare feature matrix for the unpooled level (e.g., FRAME+MAIN)
        x_unpooled_level = torch.zeros_like(N1_skip_level)

        # Place processed coarse features (x_coarse_processed) into x_unpooled_level
        # `map_current_to_original` are original indices of nodes in `x_coarse_processed`
        # `map_skip_level_to_original` are original indices of nodes in `x_unpooled_level` (and `N1_skip_level`)
        # We need to map original indices to dense indices within the skip level's representation
        original_to_dense_skip_map = {
            orig_idx.item(): dense_idx for dense_idx, orig_idx in enumerate(map_skip_level_to_original)
        }

        for dense_idx_coarse, orig_idx_coarse in enumerate(map_current_to_original):
            if orig_idx_coarse.item() in original_to_dense_skip_map:
                dense_idx_in_skip_level = original_to_dense_skip_map[orig_idx_coarse.item()]
                x_unpooled_level[dense_idx_in_skip_level] = x_coarse_processed[dense_idx_coarse]

        x_current = x_unpooled_level + N1_skip_level  # Add skip connection
        map_current_to_original = map_skip_level_to_original  # Update map for current active nodes
        # Update edge_index and edge_attr for this new set of active nodes
        ei_current, ea_current = subgraph(
            subset=map_current_to_original,
            edge_index=edge_index_orig,
            edge_attr=edge_attr_orig,
            relabel_nodes=True,
            num_nodes=num_total_nodes,
        )

        # Unpool Secondary (type 2)
        type_to_unpool = 2
        skip_info = skips_data[type_to_unpool]
        N1_skip_level = skip_info["N1_skip_features"]
        map_skip_level_to_original = skip_info["map_to_original"]
        x_coarse_processed = F.relu(self.gcn_unpool_secondary(x_current, ei_current, ea_current))
        x_unpooled_level = torch.zeros_like(N1_skip_level)
        original_to_dense_skip_map = {
            orig_idx.item(): dense_idx for dense_idx, orig_idx in enumerate(map_skip_level_to_original)
        }
        for dense_idx_coarse, orig_idx_coarse in enumerate(map_current_to_original):
            if orig_idx_coarse.item() in original_to_dense_skip_map:
                dense_idx_in_skip_level = original_to_dense_skip_map[orig_idx_coarse.item()]
                x_unpooled_level[dense_idx_in_skip_level] = x_coarse_processed[dense_idx_coarse]
        x_current = x_unpooled_level + N1_skip_level
        map_current_to_original = map_skip_level_to_original
        ei_current, ea_current = subgraph(
            subset=map_current_to_original,
            edge_index=edge_index_orig,
            edge_attr=edge_attr_orig,
            relabel_nodes=True,
            num_nodes=num_total_nodes,
        )

        # Unpool Braces (type 3)
        type_to_unpool = 3
        skip_info = skips_data[type_to_unpool]
        N1_skip_level = skip_info["N1_skip_features"]
        map_skip_level_to_original = skip_info["map_to_original"]
        x_coarse_processed = F.relu(self.gcn_unpool_brace(x_current, ei_current, ea_current))
        x_unpooled_level = torch.zeros_like(N1_skip_level)
        original_to_dense_skip_map = {
            orig_idx.item(): dense_idx for dense_idx, orig_idx in enumerate(map_skip_level_to_original)
        }
        for dense_idx_coarse, orig_idx_coarse in enumerate(map_current_to_original):
            if orig_idx_coarse.item() in original_to_dense_skip_map:  # Should be true
                dense_idx_in_skip_level = original_to_dense_skip_map[orig_idx_coarse.item()]
                x_unpooled_level[dense_idx_in_skip_level] = x_coarse_processed[dense_idx_coarse]
        x_current = x_unpooled_level + N1_skip_level
        # After unpooling braces, map_current_to_original should ideally cover all original nodes involved.
        # If the initial graph had all types, map_skip_level_to_original here would be arange(num_total_nodes).
        map_current_to_original = map_skip_level_to_original
        # ei_current should now be the original full edge_index (or subgraph if map_current_to_original is not all nodes)
        if torch.all(map_current_to_original == torch.arange(num_total_nodes, device=x_orig.device)):
            ei_current = edge_index_orig
            ea_current = edge_attr_orig
        else:  # Should not happen if all types were present and unpooled.
            ei_current, ea_current = subgraph(
                subset=map_current_to_original,
                edge_index=edge_index_orig,
                edge_attr=edge_attr_orig,
                relabel_nodes=True,
                num_nodes=num_total_nodes,
            )

        # --- Final Output Layers ---
        x_final = F.relu(self.final_conv1(x_current, ei_current, ea_current))
        x_final = self.final_conv2(
            x_final, ei_current, ea_current
        )  # Output layer, activation depends on task

        return x_final


def test_HEBG():
    # Make sure to import BeamAsNodeGraph from the correct location
    from src.graph.structural_graph import BeamAsNodeGraph

    # Make sure to import creation utilities from the correct location
    from src.utils.model_utils import create_structural_model, load_config

    print("Creating a sample structural model and graph...")
    structural_model = None
    config_path = "config.yaml"

    try:
        # Attempt to load existing config and create model
        try:
            config = load_config(config_path)
            structural_model = create_structural_model(config)
        except FileNotFoundError:
            print(f"Warning: {config_path} not found. Creating a dummy config for testing.")
            dummy_config_content = """
structure_parameters:
  length: 10.0
  width: 6.0
  height: 3.0
  bay_spacing: 3.0
  floor_height: 3.0
  main_beam_spacing_x: 3.0
  main_beam_spacing_y: 3.0
  secondary_beam_spacing_x: 1.5
  secondary_beam_spacing_y: 1.5
  brace_spacing: 3.0
  num_floors: 1
  section_indices:
    frame_beam_section_idx: 0
    main_beam_section_idx: 1
    secondary_beam_section_idx: 2
    brace_beam_section_idx: 3
sections:
  - name: "SHS100x5"
    type: "SHS"
    properties: {A: 0.0019, Ix: 0.000003, Iy: 0.000003, Iz: 0.000003, J: 0.000005}
  - name: "H200x100x5x8"
    type: "H"
    properties: {A: 0.0030, Ix: 0.000020, Iy: 0.000005, Iz: 0.000020, J: 0.000008}
  - name: "C100x50x5"
    type: "C"
    properties: {A: 0.0012, Ix: 0.000005, Iy: 0.000001, Iz: 0.000005, J: 0.000003}
  - name: "SHS80x4"
    type: "SHS"
    properties: {A: 0.0012, Ix: 0.0000015, Iy: 0.0000015, Iz: 0.0000015, J: 0.0000025}
"""
            with open(config_path, "w") as f:
                f.write(dummy_config_content)
            print(f"Dummy {config_path} created. Please re-run the script.")
            exit()  # Exit for user to re-run with the new dummy config or provide a real one

        if not structural_model or not structural_model.beams:
            print("Warning: Structural model has no beams. HEBG test might not be meaningful.")
            # Potentially exit or create a very minimal mock if critical for other tests
            # For now, we'll let it proceed to see graph conversion output

    except Exception as e:
        print(f"Error during structural model creation phase: {e}")
        print("Please ensure 'config.yaml' is present and valid, or allow dummy creation.")
        print("Ensure 'create_structural_model' and 'load_config' utilities work correctly.")
        print("Skipping HEBG test.")
        exit()

    graph_converter = BeamAsNodeGraph(structural_model)
    pyg_data = graph_converter.to_pyg_data()

    if pyg_data.num_nodes == 0:
        print("Error: Generated PyG data has 0 nodes. HEBG model cannot be tested.")
        print(
            "This might be due to an empty or problematic structural model, or issues in BeamAsNodeGraph.to_pyg_data()."
        )
    else:
        print(f"Generated PyG data: {pyg_data}")
        print(f"  Number of nodes: {pyg_data.num_nodes}")
        print(
            f"  Node type ranges: {pyg_data.node_type_ranges if hasattr(pyg_data, 'node_type_ranges') else 'Not set'}"
        )

        if pyg_data.x is None or pyg_data.x.shape[0] == 0:
            print(
                "Error: PyG data has no node features (pyg_data.x is None or empty). Cannot determine in_channels."
            )
            print("Skipping HEBG model instantiation and test.")
        else:
            in_channels = pyg_data.x.size(1)
            hidden_channels = 64
            out_channels = 32
            num_beam_types = 4

            print(
                f"\nInstantiating HEBG model with in_channels={in_channels}, hidden_channels={hidden_channels}, out_channels={out_channels}"
            )
            try:
                hebg_model = HEBG(in_channels, hidden_channels, out_channels, num_beam_types)
                print(hebg_model)

                print("\nPerforming forward pass...")
                output_features = hebg_model(pyg_data)
                print("Forward pass successful!")
                print(f"Output feature shape: {output_features.shape}")
                assert output_features.shape[0] == pyg_data.num_nodes, "Output node count mismatch"
                assert output_features.shape[1] == out_channels, "Output feature dimension mismatch"

            except Exception as e:
                print(f"Error during HEBG model operation: {e}")
                import traceback

                traceback.print_exc()

    print("\nHEBG test script finished.")


if __name__ == "__main__":
    test_HEBG()
