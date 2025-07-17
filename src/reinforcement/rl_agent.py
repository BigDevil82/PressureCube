from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_max_pool, global_mean_pool

from src.graph.gnn_encoder import GraphSAGEEncoder
from src.graph.hebg import HEBG


class Actor(nn.Module):
    """Actor网络"""

    def __init__(self, hebg_config: Dict[str, int], action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.gnn_encoder = GraphSAGEEncoder(
            in_channels=hebg_config["in_channels"],
            hidden_channels=hebg_config["hidden_channels"],
            out_channels=hebg_config["out_channels"],
            num_layers=hebg_config.get("num_layers", 4),
        )
        self.fc1 = nn.Linear(hebg_config["out_channels"], hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, data: Data) -> torch.Tensor:
        node_features = self.gnn_encoder(data)

        # if hasattr(data, "batch") and data.batch is not None:
        #     batch_vector = data.batch
        # else:
        #     batch_vector = torch.zeros(node_features.size(0), dtype=torch.long, device=node_features.device)

        x = F.relu(self.fc1(node_features))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Critic(nn.Module):
    """Critic网络"""

    def __init__(self, hebg_config: Dict[str, int], action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.hebg = HEBG(
            in_channels=hebg_config["in_channels"],
            hidden_channels=hebg_config["hidden_channels"],
            out_channels=hebg_config["out_channels"],
            num_beam_types=hebg_config.get("num_beam_types", 4),
        )
        self.fc1 = nn.Linear(hebg_config["out_channels"] + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, data: Data, action: torch.Tensor) -> torch.Tensor:
        node_features = self.hebg(data)

        if hasattr(data, "batch") and data.batch is not None:
            batch_vector = data.batch
        else:
            batch_vector = torch.zeros(node_features.size(0), dtype=torch.long, device=node_features.device)

        graph_embedding = global_mean_pool(node_features, batch_vector)

        x = torch.cat([graph_embedding, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DDPGAgent:
    """DDPG智能体"""

    def __init__(
        self,
        hebg_config: Dict[str, int],
        action_dim: int,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.001,
        sigma: float = 0.01,
        hidden_dim_fc: int = 256,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.hebg_config = hebg_config
        self.device = device
        self.action_dim = action_dim
        self.sigma = sigma

        self.actor = Actor(hebg_config, action_dim, hidden_dim=hidden_dim_fc).to(device)
        self.actor_target = Actor(hebg_config, action_dim, hidden_dim=hidden_dim_fc).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(hebg_config, action_dim, hidden_dim=hidden_dim_fc).to(device)
        self.critic_target = Critic(hebg_config, action_dim, hidden_dim=hidden_dim_fc).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.tau = tau

    def select_action(self, state_data: Data, sigma: float = None, training: bool = True) -> torch.Tensor:
        with torch.no_grad():
            action = self.actor(state_data)
            if training:
                sigma = sigma or self.sigma
                action = action + sigma * torch.randn_like(action)
        return action

    def update(self, batch: Dict[str, Any]):
        state = batch["state"]
        action = batch["action"]
        reward = batch["reward"]
        next_state = batch["next_state"]
        done = batch["done"]

        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, next_action)
            target_q = reward + (1 - done) * self.gamma * target_q

        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

    def _soft_update(self, target: nn.Module, source: nn.Module):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def save_model(self, path: str):
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "hebg_config": self.hebg_config,  # Save HEBG config used for this agent
            },
            path,
        )

    def load_model(self, path: str):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.hebg_config = checkpoint["hebg_config"]


class A2CAgent(nn.Module):
    """
    A2C Agent
    """

    def __init__(
        self,
        hebg_config: Dict[str, int],
        action_dim: int,
        actor_lr=1e-6,
        critic_lr=1e-4,
        gamma=0.99,
        tau: float = 0.001,
        sigma: float = 0.01,
        hidden_dim_fc: int = 256,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()

        self.device = device
        self.gamma = gamma

        self.actor = Actor(hebg_config, action_dim, hidden_dim=hidden_dim_fc).to(device)
        self.critic = Critic(hebg_config, action_dim, hidden_dim=hidden_dim_fc).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.tau = tau
        self.sigma = sigma

    def select_action(self, state_data: Data, sigma: float = None, training: bool = True) -> torch.Tensor:
        with torch.no_grad():
            action = self.actor(state_data)
            if training:
                sigma = sigma or self.sigma
                action = action + sigma * torch.randn_like(action)
        return action

    def update(self, batch: Dict[str, Any]):
        state = batch["state"]
        action = batch["action"]
        reward = batch["reward"]
        next_state = batch["next_state"]
        done = batch["done"]

        with torch.no_grad():
            next_state_value = self.critic(next_state).squeeze()
            td_target = reward + (1 - done) * self.gamma * next_state_value

        current_value = self.critic(state).squeeze()
        td_delta = td_target - current_value

        action_batch = action.long()  # 转换为长整型用于索引

        wall_logits = self.actor(state)  # [batch_size, num_walls]
        wall_probs = F.softmax(wall_logits, dim=1)

        selected_probs = wall_probs.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        log_probs = torch.log(selected_probs + 1e-10)  # 添加小值防止log(0)

        value_loss = F.mse_loss(current_value, td_target)

        policy_loss = -(log_probs * td_delta.detach()).mean()

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        return {
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
        }

    def save(self, path: str):
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
