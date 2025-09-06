import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class Memory:
    """存储PPO训练过程中的经验数据"""

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


class VectorActorCritic(nn.Module):
    """基于向量状态的Actor-Critic网络"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # 共享特征提取层，添加LayerNorm和Dropout提高稳定性
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )

        # Actor网络 - 输出动作概率分布，添加LayerNorm
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim),
        )

        # Critic网络 - 输出状态价值，添加LayerNorm
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用更保守的初始化
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)

        # Actor输出层使用非常小的权重初始化
        with torch.no_grad():
            self.actor[-1].weight.data *= 0.01
            self.actor[-1].bias.data.zero_()

        # Critic输出层也使用较小的初始化
        with torch.no_grad():
            self.critic[-1].weight.data *= 0.01
            self.critic[-1].bias.data.zero_()

    def forward(self, state, action_mask=None):
        """
        前向传播
        Args:
            state: 状态向量 [batch_size, state_dim]
            action_mask: 动作掩码 [batch_size, action_dim]
        Returns:
            action_logits: 动作logits [batch_size, action_dim]
            state_values: 状态价值 [batch_size, 1]
        """
        # 确保输入是正确的形状
        if state.dim() == 1:
            state = state.unsqueeze(0)  # [1, state_dim]

        # 特征提取
        # 确保状态张量是float32类型，与网络参数类型一致
        state = state.float()

        # 检查输入数据的有效性
        if torch.isnan(state).any() or torch.isinf(state).any():
            print("Warning: Input state contains NaN/Inf values!")
            state = torch.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

        # 对输入进行软裁剪，防止极端值
        state = torch.clamp(state, -10.0, 10.0)

        features = self.feature_extractor(state)

        # 获取动作logits
        action_logits = self.actor(features)  # [batch_size, action_dim]

        # 应用动作掩码
        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            # 使用更保守的掩码值，避免数值不稳定
            mask_value = -1e8
            action_logits = action_logits.masked_fill(~action_mask, mask_value)

        # 检查并处理NaN/Inf值
        if torch.isnan(action_logits).any() or torch.isinf(action_logits).any():
            print("Warning: action_logits contains NaN/Inf values!")
            action_logits = torch.nan_to_num(action_logits, nan=0.0, posinf=1e8, neginf=-1e8)

        # 获取状态价值
        state_values = self.critic(features)  # [batch_size, 1]

        # 检查并处理critic输出的NaN/Inf值
        if torch.isnan(state_values).any() or torch.isinf(state_values).any():
            print("Warning: state_values contains NaN/Inf values!")
            state_values = torch.nan_to_num(state_values, nan=0.0, posinf=1e8, neginf=-1e8)

        return action_logits, state_values

    def act(self, state, action_mask=None):
        """
        根据当前状态选择动作
        Args:
            state: 状态向量
            action_mask: 动作掩码
        Returns:
            action: 选择的动作
            action_logprob: 动作的对数概率
            state_value: 状态价值
        """
        action_logits, state_value = self.forward(state, action_mask)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.item(), action_logprob, state_value.squeeze()

    def evaluate(self, state, action, action_mask=None):
        """
        评估给定状态和动作
        Args:
            state: 状态向量批次
            action: 动作批次
            action_mask: 动作掩码批次
        Returns:
            action_logprobs: 动作对数概率
            state_values: 状态价值
            dist_entropy: 策略熵
        """
        action_logits, state_values = self.forward(state, action_mask)
        dist = Categorical(logits=action_logits)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        # 确保state_values是正确的维度
        if state_values.dim() > 1:
            state_values = state_values.squeeze(-1)
        return action_logprobs, state_values, dist_entropy


class VectorPPOAgent:
    """基于向量状态的PPO智能体"""

    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        hidden_dim=256,
        lr=5e-5,  # 更低的学习率
        gamma=0.99,
        k_epochs=4,
        eps_clip=0.1,  # 更小的裁剪范围
        gae_lambda=0.95,
        ent_coef=0.02,
        vf_coef=0.25,  # 降低价值损失权重
        max_grad_norm=0.2,  # 更严格的梯度裁剪
        clip_range_decay=0.999,
        ent_coef_decay=0.995,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.initial_eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.initial_ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.clip_range_decay = clip_range_decay
        self.ent_coef_decay = ent_coef_decay

        # 创建Actor-Critic网络
        self.policy = VectorActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # 创建旧策略网络用于PPO更新
        self.policy_old = VectorActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.mse_loss = nn.MSELoss()

    def select_action(self, state, memory: Memory = None, action_mask=None):
        """
        选择动作
        Args:
            state: 状态向量 (numpy array or list)
            memory: 经验存储器
            action_mask: 动作掩码 (numpy array or list)
        Returns:
            action: 选择的动作
        """
        with torch.no_grad():
            # 转换状态为tensor
            if isinstance(state, (list, np.ndarray)):
                state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
            else:
                state_tensor = state.to(self.device).float()

            # 转换动作掩码为tensor
            if action_mask is not None:
                if isinstance(action_mask, (list, np.ndarray)):
                    action_mask_tensor = torch.BoolTensor(action_mask).to(self.device)
                else:
                    action_mask_tensor = action_mask.to(self.device)
                if action_mask_tensor.dim() == 1:
                    action_mask_tensor = action_mask_tensor.unsqueeze(0)
            else:
                action_mask_tensor = None

            action, log_prob, value = self.policy_old.act(state_tensor, action_mask_tensor)

        # 存储到memory中
        if memory is not None:
            memory.states.append(state_tensor.cpu())
            memory.actions.append(torch.tensor(action, device=self.device))
            memory.logprobs.append(log_prob.cpu())
            # 确保value是标量
            if value.dim() > 0:
                memory.values.append(value.squeeze().cpu())
            else:
                memory.values.append(value.cpu())
            if action_mask is not None:
                memory.action_masks.append(torch.BoolTensor(action_mask))
            else:
                memory.action_masks.append(None)

        return action

    def update(self, memory, last_value, done):
        """
        使用PPO算法更新策略
        Args:
            memory: 存储的经验数据
            last_value: 最后一步的状态价值
            done: 是否结束
        """
        # 计算GAE优势估计
        old_values = torch.stack(memory.values).to(self.device)  # [num_steps]
        rewards = torch.tensor(memory.rewards, dtype=torch.float32, device=self.device)  # [num_steps]
        dones = torch.tensor(memory.dones, dtype=torch.float32, device=self.device)  # [num_steps]

        if isinstance(last_value, torch.Tensor):
            last_value = last_value.cpu().item()

        # GAE计算
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        num_steps = len(rewards)

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_val = last_value if not done else 0.0
            else:
                next_val = old_values[t + 1].item()

            mask = 1.0 - dones[t]
            # 确保old_values[t]是标量
            old_val = old_values[t].item()
            delta = rewards[t] + self.gamma * next_val * mask - old_val
            last_gae_lam = delta + self.gamma * self.gae_lambda * last_gae_lam * mask
            advantages[t] = last_gae_lam

        # 计算回报
        returns = advantages + old_values

        # 标准化优势
        std = advantages.std(unbiased=False)
        if std > 0:
            advantages = (advantages - advantages.mean()) / (std + 1e-8)
        if any(torch.isnan(advantage) for advantage in advantages):
            raise ValueError("Warning: advantages contains NaN!")

        # 准备训练数据
        old_states = torch.stack(memory.states).to(self.device)  # [num_steps, state_dim]
        old_actions = torch.stack(memory.actions).to(self.device)  # [num_steps]
        old_logprobs = torch.stack(memory.logprobs).to(self.device).view(-1)  # [num_steps]

        # 处理动作掩码
        old_action_masks = None
        if any(mask is not None for mask in memory.action_masks):
            valid_masks = []
            for mask in memory.action_masks:
                if mask is not None:
                    valid_masks.append(mask)
                else:
                    # 如果没有掩码，创建全True掩码
                    valid_masks.append(torch.ones(self.action_dim, dtype=torch.bool))
            old_action_masks = torch.stack(valid_masks).to(self.device)

        # PPO更新循环
        for epoch in range(self.k_epochs):
            # 评估旧动作和价值
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions, old_action_masks
            )

            # 计算比率
            ratios = torch.exp(logprobs - old_logprobs)

            # 计算代理损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = self.mse_loss(state_values, returns)

            # 熵损失
            entropy_loss = -dist_entropy.mean()

            # 总损失
            total_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

            # 梯度更新
            self.optimizer.zero_grad()
            total_loss.backward()

            # 检查梯度
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

            self.optimizer.step()

        # 衰减超参数
        self.eps_clip = max(0.05, self.eps_clip * self.clip_range_decay)
        self.ent_coef = max(0.001, self.ent_coef * self.ent_coef_decay)

        # 更新旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 返回训练统计信息
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": total_loss.item(),
            "eps_clip": self.eps_clip,
            "ent_coef": self.ent_coef,
        }

    def save_model(self, path: str):
        """保存模型"""
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "eps_clip": self.eps_clip,
                "ent_coef": self.ent_coef,
            },
            path,
        )

    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.policy_old.load_state_dict(self.policy.state_dict())

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "eps_clip" in checkpoint:
            self.eps_clip = checkpoint["eps_clip"]
        if "ent_coef" in checkpoint:
            self.ent_coef = checkpoint["ent_coef"]

    def set_mode(self, eval=False):
        """设置网络模式"""
        if eval:
            self.policy.eval()
            self.policy_old.eval()
        else:
            self.policy.train()
            self.policy_old.train()

    def get_action_probabilities(self, state, action_mask=None):
        """
        获取所有动作的概率分布（用于分析）
        Args:
            state: 状态向量
            action_mask: 动作掩码
        Returns:
            action_probs: 动作概率分布
        """
        with torch.no_grad():
            if isinstance(state, (list, np.ndarray)):
                state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
            else:
                state_tensor = state.to(self.device).float()

            if action_mask is not None:
                if isinstance(action_mask, (list, np.ndarray)):
                    action_mask_tensor = torch.BoolTensor(action_mask).to(self.device)
                else:
                    action_mask_tensor = action_mask.to(self.device)
                if action_mask_tensor.dim() == 1:
                    action_mask_tensor = action_mask_tensor.unsqueeze(0)
            else:
                action_mask_tensor = None

            action_logits, _ = self.policy.forward(state_tensor, action_mask_tensor)
            action_probs = torch.softmax(action_logits, dim=-1)

            return action_probs.cpu().numpy()
