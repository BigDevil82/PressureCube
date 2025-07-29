import random
from collections import deque
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_max_pool, global_mean_pool

from src.graph.gnn_encoder import GraphSAGEEncoder


class GraphQNetwork(nn.Module):
    def __init__(self, state_dim, global_feature_dim):
        super().__init__()
        self.encoder = GraphSAGEEncoder(
            in_channels=state_dim, hidden_channels=64, out_channels=32, num_layers=3
        )
        self.global_feature_encoder = nn.Sequential(
            nn.Linear(global_feature_dim, 64), nn.ReLU(), nn.Linear(64, 32)
        )
        self.q_head = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 8)  # 降低/提高框架梁、主梁、次梁、支撑梁的截面尺寸
        )

    def forward(self, data: Data):
        # Get node embeddings
        node_embeddings = self.encoder(data)
        global_features: torch.Tensor = self.global_feature_encoder(data.global_features)

        # Compute Q-values for each node

        # For batch handling
        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch
            glb_feature_batch = []
            node_nums_per_graph = torch.bincount(batch).tolist()
            for i, node_num in enumerate(node_nums_per_graph):
                glb_feature_batch.append(global_features[i : i + 1].repeat(node_num, 1))
            global_features = torch.cat(glb_feature_batch, dim=0)

            node_embeddings = torch.cat([node_embeddings, global_features], dim=1)
            node_embeddings = global_mean_pool(node_embeddings, batch)
            q_values = self.q_head(node_embeddings)
        else:
            # Single graph
            batch = torch.zeros(node_embeddings.size(0), dtype=torch.long, device=node_embeddings.device)
            global_features = global_features.repeat(node_embeddings.size(0), 1)
            node_embeddings = torch.cat([node_embeddings, global_features], dim=1)
            node_embeddings = global_mean_pool(node_embeddings, batch)
            q_values = self.q_head(node_embeddings)

        return q_values


class DQNSizeAgent:
    def __init__(
        self,
        state_dim,
        glb_feat_dim,
        device,
        gamma=0.99,
        lr=1e-3,
        target_update=100,
    ):

        # Q-network and target network
        self.q_net = GraphQNetwork(state_dim, glb_feat_dim).to(device)
        self.target_net = GraphQNetwork(state_dim, glb_feat_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        # Training components
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Hyperparameters
        self.gamma = gamma
        self.target_update = target_update
        self.device = device

        self.step_count = 0

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, 7)
        else:
            with torch.no_grad():
                q_values = self.q_net(state)
                return q_values.argmax().item()

    def update(self, batch: Dict[str, Any]):
        states, actions, rewards, next_states, dones = (
            batch["state"],
            batch["action"],
            batch["reward"],
            batch["next_state"],
            batch["done"],
        )

        # Compute Q-values
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1))

        # Compute target Q-values
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(dim=1)[0].view(-1, 1)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q_values

        # Compute loss
        loss = self.loss_fn(q_values, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def save_model(self, path: str):
        torch.save(self.q_net.state_dict(), path)

    def load_model(self, path: str):
        self.q_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.q_net.state_dict())

    def set_mode(self, eval=False):
        """Sets the networks to evaluation mode."""
        if eval:
            self.q_net.eval()
            self.target_net.eval()
        else:
            self.q_net.train()
            self.target_net.train()
