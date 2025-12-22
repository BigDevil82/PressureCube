import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from src.reinforcement.agent_model import StructuralGNN


class PPOAgent:
    def __init__(self, node_dim, edge_dim, action_dim, lr, gamma, K_epochs, eps_clip, device):
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = StructuralGNN(node_dim, edge_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # 经验缓冲区
        self.buffer = []

    def put_data(self, item):
        """存储单步数据: (data, action, log_prob, reward, done)"""
        self.buffer.append(item)

    def update(self):
        """使用 PPO 更新网络"""
        if len(self.buffer) == 0:
            return

        # 1. 整理数据
        # PyG 的 Data 对象不能直接 stack，需要用列表保存
        batched_data = []
        actions = []
        log_probs = []
        rewards = []
        dones = []

        for item in self.buffer:
            d, a, lp, r, done = item
            batched_data.append(d)
            actions.append(a)
            log_probs.append(lp)
            rewards.append(r)
            dones.append(done)

        # 转换 Tensor
        # 注意：Data 对象在 DataLoader 中会自动处理 batch
        actions = torch.stack(actions).detach().to(self.device).squeeze(1)
        old_log_probs = torch.stack(log_probs).detach().to(self.device).squeeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # 2. 计算 Monte Carlo 奖励 (简单的 Reward-to-Go, 也可以换成 GAE)
        discounted_rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(self.device)
        # 归一化奖励，加速收敛
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
            discounted_rewards.std() + 1e-7
        )

        # 3. PPO 更新循环
        # 使用 DataLoader 处理 Graph Batching
        # batch_size 设为 len(buffer) 即全批量更新，也可以设小一点做 mini-batch
        loader = DataLoader(batched_data, batch_size=len(batched_data), shuffle=False)

        for _ in range(self.K_epochs):
            # 因为只有一个全批量，这里循环其实只跑一次 forward，但为了兼容 mini-batch 逻辑保留结构
            for i, batch_data in enumerate(loader):
                batch_data = batch_data.to(self.device)

                # 评估旧动作在新策略下的概率
                logprobs, state_values, dist_entropy = self.policy.evaluate(batch_data, actions)

                # 匹配维度
                state_values = torch.squeeze(state_values)

                # 计算比率 ratio = pi_new / pi_old
                ratios = torch.exp(logprobs - old_log_probs)

                # 计算优势函数 Advantages
                advantages = discounted_rewards - state_values.detach()

                # Surrogate Loss
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                # 总 Loss = Actor Loss - Critic Loss * 0.5 - Entropy * 0.01
                loss = (
                    -torch.min(surr1, surr2)
                    + 0.5 * self.loss_fn(state_values, discounted_rewards)
                    - 0.03 * dist_entropy
                )

                # 反向传播
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

        # 清空缓冲区
        self.buffer = []

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path))
