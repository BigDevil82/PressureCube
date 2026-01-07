import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool


class StructuralGNN(nn.Module):
    """
    GNN 特征提取器 + Actor-Critic Heads
    """

    def __init__(self, node_dim, edge_dim, action_dim, hidden_dim=128):
        super(StructuralGNN, self).__init__()

        # --- 1. Graph Encoder (Shared Backbone) ---
        # 第一层 GAT: 输入节点特征 + 边特征
        self.conv1 = GATv2Conv(node_dim, hidden_dim, heads=4, edge_dim=edge_dim, concat=False)
        # 第二层 GAT
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=4, edge_dim=edge_dim, concat=False)
        # 第三层 GAT
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim, heads=4, edge_dim=edge_dim, concat=False)

        # --- 2. Actor Head (Policy) ---
        # 我们有 4 个决策维度，每个维度有 5 个选项，所以总输出节点是 4 * 5 = 20
        self.action_dim = action_dim  # 4
        # self.action_options = action_options  # 5

        self.actor_frame = nn.Linear(hidden_dim * 2, 20)
        self.actor_main = nn.Linear(hidden_dim * 2, 20)
        self.actor_sec = nn.Linear(hidden_dim * 2, 20)
        self.actor_brace = nn.Linear(hidden_dim * 2, 20)

        # --- 3. Critic Head (Value) ---
        self.critic_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.critic_value = nn.Linear(hidden_dim, 1)

    def forward(self, data: Data):
        """
        前向传播
        Args:
            data: PyG Batch Data 对象
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # --- GNN Backbone ---
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = F.elu(self.conv3(x, edge_index, edge_attr))

        # --- Global Pooling (Readout) ---
        # 将节点特征聚合为图特征。结合 Mean 和 Max 可以同时捕捉整体趋势和局部极值（如最大应力）
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_graph = torch.cat([x_mean, x_max], dim=1)  # Shape: [batch_size, hidden_dim * 2]

        return x_graph

    def get_action(self, data, eval_mode=False):
        x_graph = self.forward(data)

        logits = {
            "frame": self.actor_frame(x_graph),
            "main": self.actor_main(x_graph),
            "sec": self.actor_sec(x_graph),
            "brace": self.actor_brace(x_graph),
        }

        dists = {k: Categorical(logits=v) for k, v in logits.items()}

        if eval_mode:
            actions = {k: torch.argmax(v.logits, dim=-1) for k, v in dists.items()}
        else:
            actions = {k: v.sample() for k, v in dists.items()}

        # 拼成一个 tensor: [batch, 4]
        action_tensor = torch.stack(
            [actions["frame"], actions["main"], actions["sec"], actions["brace"]], dim=-1
        )

        log_prob = sum(dists[k].log_prob(actions[k]) for k in dists)

        return action_tensor, log_prob

    def evaluate(self, data, actions):
        """Actor + Critic: 评估动作 (用于更新)"""
        x_graph = self.forward(data)

        logits = {
            "frame": self.actor_frame(x_graph),
            "main": self.actor_main(x_graph),
            "sec": self.actor_sec(x_graph),
            "brace": self.actor_brace(x_graph),
        }

        dists = {k: Categorical(logits=v) for k, v in logits.items()}

        log_probs = (
            dists["frame"].log_prob(actions[:, 0])
            + dists["main"].log_prob(actions[:, 1])
            + dists["sec"].log_prob(actions[:, 2])
            + dists["brace"].log_prob(actions[:, 3])
        )

        entropy = sum(d.entropy() for d in dists.values())

        value = self.critic_value(F.elu(self.critic_fc(x_graph)))

        return log_probs, value.squeeze(-1), entropy
