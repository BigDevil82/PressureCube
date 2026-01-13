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

        self.actor = nn.Linear(hidden_dim * 2, action_dim)  # 选择缩哪一类梁

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

        logits = self.actor(x_graph)

        dists = Categorical(logits=logits)

        if eval_mode:
            actions = torch.argmax(dists.logits, dim=-1)
        else:
            actions = dists.sample()

        log_prob = dists.log_prob(actions)

        return actions, log_prob

    def evaluate(self, data, actions):
        """Actor + Critic: 评估动作 (用于更新)"""
        x_graph = self.forward(data)

        logits = self.actor(x_graph)
        dists = Categorical(logits=logits)

        log_probs = dists.log_prob(actions)

        entropy = dists.entropy()

        value = self.critic_value(F.elu(self.critic_fc(x_graph)))

        return log_probs, value.squeeze(-1), entropy
