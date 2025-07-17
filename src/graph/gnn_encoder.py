import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_networkx


class GraphSAGEEncoder(nn.Module):
    """
    A simple GraphSAGE encoder that takes a graph and returns node embeddings.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 3):
        """
        Initializes the GraphSAGE encoder.

        Args:
            in_channels (int): Number of input features for each node.
            hidden_channels (int): Number of features in the hidden layers.
            out_channels (int): Number of output features for each node (embedding size).
            num_layers (int): The number of GraphSAGE layers.
        """
        super().__init__()
        self.convs = nn.ModuleList()

        if num_layers < 2:
            raise ValueError("Number of layers must be at least 2.")

        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, data: Data) -> torch.Tensor:
        """
        Performs the forward pass of the encoder.

        Args:
            data (Data): The input graph data from PyTorch Geometric.
                         Must contain `x` (node features) and `edge_index`.

        Returns:
            torch.Tensor: The node embeddings.
        """
        x, edge_index = data.x, data.edge_index

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            # No activation on the last layer
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x


def test_GraphSAGEEncoder():
    """
    A simple test function to validate the GraphSAGEEncoder.
    Creates a sample graph, passes it through the encoder, and prints the output shape.
    """
    # Parameters
    num_nodes = 10
    in_channels = 5
    hidden_channels = 16
    out_channels = 8
    num_layers = 3

    # Create a sample graph
    # Node features (random)
    x = torch.randn(num_nodes, in_channels)
    # Edge index (a simple chain graph)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)

    print("--- Testing GraphSAGEEncoder ---")
    print(f"Input graph: {data}")
    print(f"Number of nodes: {data.num_nodes}, Number of features: {data.num_node_features}")

    # Initialize the encoder
    encoder = GraphSAGEEncoder(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
    )
    print("\nEncoder architecture:")
    print(encoder)

    # Get node embeddings
    node_embeddings = encoder(data)

    print(f"\nOutput node embeddings shape: {node_embeddings.shape}")
    assert node_embeddings.shape == (num_nodes, out_channels)

    print("\n--- Test Passed ---")


if __name__ == "__main__":
    test_GraphSAGEEncoder()
