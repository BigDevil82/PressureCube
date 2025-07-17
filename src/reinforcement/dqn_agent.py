import random
from collections import deque
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch

from src.graph.gnn_encoder import GraphSAGEEncoder


class GraphQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.encoder = GraphSAGEEncoder(
            in_channels=state_dim, hidden_channels=64, out_channels=32, num_layers=3
        )
        self.q_head = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 1)  # Output Q-value for each node
        )

    def forward(self, data):
        # Get node embeddings
        node_embeddings = self.encoder(data)

        # Compute Q-values for each node
        q_values = self.q_head(node_embeddings)

        # For batch handling
        if hasattr(data, "batch") and data.batch is not None:
            # Split Q-values by graphs in batch
            batch = data.batch
            q_values = torch.split(q_values, torch.bincount(batch).tolist())
        else:
            # Single graph
            q_values = [q_values]

        return q_values


class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        gamma=0.99,
        lr=1e-3,
        target_update=100,
    ):

        # Q-network and target network
        self.q_net = GraphQNetwork(state_dim, action_dim).to(device)
        self.target_net = GraphQNetwork(state_dim, action_dim).to(device)
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
            return random.randint(0, state.num_nodes - 1)
        else:
            with torch.no_grad():
                q_values = self.q_net(state)
                return q_values[0].argmax().item()

    def update(self, batch: Dict[str, Any]):
        states, actions, rewards, next_states, dones = (
            batch["state"],
            batch["action"],
            batch["reward"],
            batch["next_state"],
            batch["done"],
        )

        # Compute Q-values
        q_values = self.q_net(states)
        current_q_values = torch.zeros_like(rewards)
        for i, action in enumerate(actions):
            current_q_values[i] = q_values[i][action.item()]

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            next_max_q_values = torch.zeros_like(rewards)
            for i, action in enumerate(actions):
                next_max_q_values[i] = next_q_values[i].max()
            target_q = rewards + (1 - dones) * self.gamma * next_max_q_values

        # Compute loss
        loss = self.loss_fn(current_q_values, target_q)

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

    def set_eval_mode(self):
        """Sets the networks to evaluation mode."""
        self.q_net.eval()
        self.target_net.eval()
