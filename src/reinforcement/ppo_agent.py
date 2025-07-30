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

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]
        del self.values[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, global_feature_dim):
        super().__init__()
        self.encoder = GraphSAGEEncoder(
            in_channels=state_dim, hidden_channels=64, out_channels=32, num_layers=3
        )
        self.global_feature_encoder = nn.Sequential(
            nn.Linear(global_feature_dim, 64), nn.ReLU(), nn.Linear(64, 32)
        )

        # Actor head
        self.actor = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 8))

        # Critic head
        self.critic = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, data: Data):
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
        state_values = self.critic(x)

        return action_logits, state_values

    def act(self, state):
        action_logits, state_value = self.forward(state)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.item(), action_logprob, state_value

    def evaluate(self, state, action):
        action_logits, state_value = self.forward(state)
        dist = Categorical(logits=action_logits)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, state_value, dist_entropy


class PPOAgent:
    def __init__(
        self,
        state_dim,
        glb_feat_dim,
        device,
        lr=1e-4,
        gamma=0.99,
        k_epochs=4,
        eps_clip=0.2,
        gae_lambda=0.95,
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.device = device
        self.gae_lambda = gae_lambda

        self.policy = ActorCritic(state_dim, glb_feat_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.policy_old = ActorCritic(state_dim, glb_feat_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.mse_loss = nn.MSELoss()

    def select_action(self, state, memory: Memory):
        with torch.no_grad():
            action, log_prob, value = self.policy_old.act(state)

        memory.states.append(state)
        memory.actions.append(torch.tensor(action, device=self.device))
        memory.logprobs.append(log_prob)
        memory.values.append(value)

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
        old_logprobs = torch.stack(memory.logprobs, dim=0).detach()  # Shape: [num_steps, 1]

        # Optimize policy for K epochs
        for i in range(self.k_epochs):
            # Evaluate old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # logprobs: [num_steps], state_values: [num_steps, 1], dist_entropy: [num_steps]

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs.unsqueeze(1) - old_logprobs)  # Shape: [num_steps, 1]

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = (
                -torch.min(surr1, surr2).mean()
                + 0.5 * self.mse_loss(state_values, returns)
                - 0.01 * dist_entropy.mean()
            )

            # take gradient step
            self.optimizer.zero_grad()

            # The retain_graph=True is crucial for the inner loop of PPO
            # as the graph is needed for the subsequent updates in the same batch.
            is_last_epoch = i == self.k_epochs - 1
            loss.mean().backward(retain_graph=not is_last_epoch)

            self.optimizer.step()

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
