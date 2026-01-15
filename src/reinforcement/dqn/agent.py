import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        # dueling
        self.V = nn.Linear(256, 1)
        self.A = nn.Linear(256, n_actions)

    def forward(self, x):
        h = self.net(x)
        v = self.V(h)
        a = self.A(h)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q


class Replay:
    def __init__(self, cap=200000):
        self.buf = deque(maxlen=cap)

    def add(self, *item):
        self.buf.append(item)

    def sample(self, batch):
        batch = random.sample(self.buf, batch)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buf)


def masked_argmax(q, mask):
    # q: (n_actions,), mask: (n_actions,) bool
    qq = q.copy()
    qq[~mask] = -1e30
    return int(qq.argmax())


class DQNAgent:
    def __init__(self, obs_dim, n_actions, lr=2e-4, gamma=0.95, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.q = QNet(obs_dim, n_actions).to(self.device)
        self.tgt = QNet(obs_dim, n_actions).to(self.device)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.n_actions = n_actions

    @torch.no_grad()
    def act(self, obs, mask, eps=0.1):
        if np.random.rand() < eps:
            valid = np.flatnonzero(mask)
            return int(np.random.choice(valid))
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.q(x).squeeze(0).cpu().numpy()
        return masked_argmax(q, mask)

    def update(self, replay, batch=256, target_update_tau=0.005):
        if len(replay) < batch:
            return

        obs, act, rew, nobs, done, mask, nmask = replay.sample(batch)

        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        nobs = torch.tensor(nobs, dtype=torch.float32, device=self.device)
        act = torch.tensor(act, dtype=torch.int64, device=self.device)
        rew = torch.tensor(rew, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)
        nmask = torch.tensor(nmask.astype(np.float32), dtype=torch.float32, device=self.device)  # 1 for valid

        q = self.q(obs).gather(1, act.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: action from online net, value from target net
            q_next_online = self.q(nobs)  # (B, A)
            q_next_online = q_next_online + (nmask - 1.0) * 1e30  # invalid -> -inf
            a_star = q_next_online.argmax(dim=1)

            q_next_tgt = self.tgt(nobs).gather(1, a_star.view(-1, 1)).squeeze(1)
            y = rew + self.gamma * (1.0 - done) * q_next_tgt

        loss = (q - y).pow(2).mean()
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()

        # soft update target
        with torch.no_grad():
            for p, tp in zip(self.q.parameters(), self.tgt.parameters()):
                tp.data.mul_(1 - target_update_tau).add_(target_update_tau * p.data)
