import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool

from src.graph.gnn_encoder import GraphSAGEEncoder


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.action_masks = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]
        del self.values[:]
        del self.action_masks[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, global_feature_dim):
        super().__init__()
        self.encoder = GraphSAGEEncoder(
            in_channels=state_dim, hidden_channels=64, out_channels=32, num_layers=3
        )
        self.global_feature_encoder = nn.Sequential(
            nn.Linear(global_feature_dim, 64), nn.ReLU(), nn.Linear(64, 32)
        )

        # Actor head
        self.actor = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, action_dim))

        # Critic head
        self.critic = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, data: Data, action_mask=None):
        # Get node embeddings
        node_embeddings = self.encoder(data)
        global_features: torch.Tensor = self.global_feature_encoder(data.global_features)

        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch
            glb_feature_batch = []
            node_nums_per_graph = torch.bincount(batch).tolist()
            for i, node_num in enumerate(node_nums_per_graph):
                glb_feature_batch.append(global_features[i : i + 1].repeat(node_num, 1))
            global_features = torch.cat(glb_feature_batch, dim=0)

            x = torch.cat([node_embeddings, global_features], dim=1)
            x = global_mean_pool(x, batch)
        else:
            # Single graph
            batch = torch.zeros(node_embeddings.size(0), dtype=torch.long, device=node_embeddings.device)
            global_features = global_features.repeat(node_embeddings.size(0), 1)
            x = torch.cat([node_embeddings, global_features], dim=1)
            x = global_mean_pool(x, batch)

        action_logits = self.actor(x)

        # 应用动作掩码
        if action_mask is not None:
            # action_mask: [batch_size, num_actions] 布尔张量
            mask_value = -1e9
            action_logits = action_logits.masked_fill(~action_mask, mask_value)

        state_values = self.critic(x)

        return action_logits, state_values

    def act(self, state, action_mask=None):
        action_logits, state_value = self.forward(state, action_mask)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.item(), action_logprob, state_value

    def evaluate(self, state, action, action_mask=None):
        action_logits, state_value = self.forward(state, action_mask)
        dist = Categorical(logits=action_logits)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, state_value, dist_entropy


class PPOAgent:
    def __init__(
        self,
        state_dim,
        glb_feat_dim,
        action_dim,
        device,
        lr=5e-5,  # 降低学习率
        gamma=0.99,
        k_epochs=4,
        eps_clip=0.2,
        gae_lambda=0.95,
        ent_coef: float = 0.05,  # 提高初始熵系数
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        clip_range_decay: float = 0.999,  # eps_clip衰减率
        ent_coef_decay: float = 0.9995,  # 熵系数衰减率
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.initial_eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.device = device
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.initial_ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.clip_range_decay = clip_range_decay
        self.ent_coef_decay = ent_coef_decay

        self.policy = ActorCritic(state_dim, action_dim, glb_feat_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, betas=(0.9, 0.999))

        self.policy_old = ActorCritic(state_dim, action_dim, glb_feat_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.mse_loss = nn.MSELoss()

    def select_action(self, state, memory: Memory, action_mask=None):
        with torch.no_grad():
            # 将action_mask转换为torch张量
            if action_mask is not None:
                action_mask_tensor = torch.tensor(
                    action_mask, dtype=torch.bool, device=self.device
                ).unsqueeze(0)
            else:
                action_mask_tensor = None

            action, log_prob, value = self.policy_old.act(state, action_mask_tensor)

        memory.states.append(state)
        memory.actions.append(torch.tensor(action, device=self.device))
        memory.logprobs.append(log_prob.detach())
        memory.values.append(value.detach())
        if action_mask is not None:
            memory.action_masks.append(torch.tensor(action_mask, dtype=torch.bool, device=self.device))
        else:
            memory.action_masks.append(None)

        return action

    def update(self, memory: Memory, last_value, done):
        # GAE: General Advantage Estimation
        old_values = torch.cat(memory.values).detach()  # Shape: [num_steps, 1]
        rewards = torch.tensor(memory.rewards, dtype=torch.float32, device=self.device).unsqueeze(
            1
        )  # Shape: [num_steps, 1]
        dones = torch.tensor(memory.dones, dtype=torch.float32, device=self.device).unsqueeze(
            1
        )  # Shape: [num_steps, 1]

        advantages = torch.zeros_like(rewards)  # Shape: [num_steps, 1]
        last_gae_lam = 0
        num_steps = len(rewards)

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_val = last_value  # Shape: [1, 1]
            else:
                next_val = old_values[t + 1]

            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_val * mask - old_values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * last_gae_lam * mask
            advantages[t] = last_gae_lam

        # Compute returns
        returns = advantages + old_values  # Shape: [num_steps, 1]

        # Normalizing advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # convert list to tensor
        old_states = Batch.from_data_list(memory.states).to(self.device)
        old_actions = torch.stack(memory.actions, dim=0).detach()  # Shape: [num_steps]
        old_logprobs = torch.stack(memory.logprobs, dim=0).detach()  # Shape: [num_steps]

        # 处理action masks
        old_action_masks = None
        if any(mask is not None for mask in memory.action_masks):
            valid_masks = [
                mask if mask is not None else torch.ones(8, dtype=torch.bool, device=self.device)
                for mask in memory.action_masks
            ]
            old_action_masks = torch.stack(valid_masks, dim=0)

        # Optimize policy for K epochs
        for i in range(self.k_epochs):
            # Evaluate old actions and values with action masks
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions, old_action_masks
            )
            # Ensure shapes are consistent: [num_steps]
            logprobs = logprobs.view(-1)
            old_logprobs_view = old_logprobs.view(-1)
            advantages_view = advantages.view(-1)
            returns_view = returns.view(-1, 1)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs_view)  # Shape: [num_steps]

            # Finding Surrogate Loss
            surr1 = ratios * advantages_view
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_view

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.mse_loss(state_values, returns)
            entropy_loss = -dist_entropy.mean()

            loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

            # take gradient step
            self.optimizer.zero_grad()

            # The retain_graph=True is crucial for the inner loop of PPO
            # as the graph is needed for the subsequent updates in the same batch.
            is_last_epoch = i == self.k_epochs - 1
            loss.backward(retain_graph=not is_last_epoch)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

        # 衰减超参数
        self.eps_clip = max(0.1, self.eps_clip * self.clip_range_decay)
        self.ent_coef = max(0.005, self.ent_coef * self.ent_coef_decay)

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save_model(self, path: str):
        torch.save(self.policy.state_dict(), path)

    def load_model(self, path: str):
        self.policy.load_state_dict(torch.load(path))
        self.policy_old.load_state_dict(self.policy.state_dict())

    def set_mode(self, eval=False):
        if eval:
            self.policy.eval()
            self.policy_old.eval()
        else:
            self.policy.train()
            self.policy_old.train()
