import torch
import torch.nn as nn
from torch_geometric.data import Batch

from src.reinforcement.agent_model import StructuralGNN


class PPOAgent:
    def __init__(self, node_dim, edge_dim, action_dim, lr, gamma, K_epochs, eps_clip, batch_size, device):
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.batch_size = batch_size
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

        # --- 1. 整理数据 (这部分保持不变) ---
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

        # 转换为 Tensor
        # 【修改点 1】Action 处理
        # 以前是 .long()，现在连续动作必须是 .float()
        actions = torch.stack(actions).to(self.device).detach()

        # 在 Continuous 场景下，action shape 应该是 [batch_size, 4]
        # 如果 stack 后多了一维 (例如 [batch, 1, 4])，需要 squeeze
        if actions.dim() > 2:
            actions = actions.squeeze(1)

        # 关键：转为 float32
        actions = actions.float()

        old_log_probs = torch.stack(log_probs).detach().to(self.device).squeeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # --- 2. 计算 Monte Carlo 奖励 (保持不变) ---
        discounted_rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(self.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
            discounted_rewards.std() + 1e-7
        )

        # --- 3. PPO 更新循环 (关键修改) ---

        # 获取总样本数
        buffer_size = len(batched_data)
        # 确保 batch_size 合理
        batch_size = self.batch_size if self.batch_size < buffer_size else buffer_size

        for _ in range(self.K_epochs):
            # 【关键】每个 Epoch 都生成随机索引来实现 Shuffle
            indices = torch.randperm(buffer_size)

            # Mini-batch 循环
            for start in range(0, buffer_size, batch_size):
                end = start + batch_size
                idx = indices[start:end]  # 获取当前 batch 的索引

                # A. 切片张量数据 (对应当前 batch)
                mini_actions = actions[idx]
                mini_old_log_probs = old_log_probs[idx]
                mini_advantages = discounted_rewards[idx]  # 这里暂时用 Reward 当 Advantage

                # B. 切片图数据并手动 Batch 化
                # 取出对应的 Data 对象列表
                mini_batch_graph_list = [batched_data[i] for i in idx]
                # 使用 PyG 的工具将列表合并为一个 Batch Data 对象
                mini_batch_data = Batch.from_data_list(mini_batch_graph_list).to(self.device)

                # --- 下面是标准的 PPO 计算，注意变量名换成了 mini_ ---

                # 评估旧动作在新策略下的概率
                # 传入的是 mini_batch_data 和 mini_actions
                logprobs, state_values, dist_entropy = self.policy.evaluate(mini_batch_data, mini_actions)

                state_values = torch.squeeze(state_values)

                # 计算比率 ratio = pi_new / pi_old
                ratios = torch.exp(logprobs - mini_old_log_probs)

                # 计算优势函数 (这里用的是 returns - values)
                advantages = mini_advantages - state_values.detach()

                # Surrogate Loss
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                # Loss 计算
                loss = (
                    -torch.min(surr1, surr2)
                    + 0.5 * self.loss_fn(state_values, mini_advantages)
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
        print("Loading PPO Agent from:", path)
        self.policy.load_state_dict(torch.load(path, weights_only=True))
