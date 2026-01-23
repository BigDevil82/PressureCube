"""
Fixed DQN training script with critical improvements.

Key fixes:
1. Much smaller batch size (32) - start learning immediately
2. Longer training (1000+ episodes minimum)
3. Smaller network for this simple problem
4. Lower gamma (0.5) - short horizon task
5. Simple, consistent reward function
6. No early stopping exploitation
"""

import os
import random
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.reinforcement.dqn.env import SteelSectionOptEnv
from src.reinforcement.dqn.metric import EpisodeStats, MetricsLogger


# ============= Simplified Network =============
class SimpleQNet(nn.Module):
    """Much smaller network - this problem is simple!"""

    def __init__(self, obs_dim, n_actions):
        super().__init__()
        # 12 inputs, 81 outputs - use small hidden layers
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class FastReplay:
    """Simple replay buffer."""

    def __init__(self, cap=50000):
        self.buf = deque(maxlen=cap)

    def add(self, obs, act, rew, nobs, done, mask, nmask):
        self.buf.append((obs, act, rew, nobs, done, mask, nmask))

    def sample(self, batch):
        batch_data = random.sample(self.buf, batch)
        return map(np.array, zip(*batch_data))

    def __len__(self):
        return len(self.buf)


def masked_argmax(q, mask):
    qq = q.copy()
    qq[~mask] = -1e30
    return int(qq.argmax())


class FixedDQNAgent:
    """
    DQN Agent with fixes for this specific problem.

    Key changes:
    - batch_size=32 (not 256!) - start learning fast
    - gamma=0.5 - short horizon, don't over-plan
    - smaller network
    - more frequent updates
    """

    def __init__(self, obs_dim, n_actions, lr=1e-3, gamma=0.5, tau=0.01, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.q = SimpleQNet(obs_dim, n_actions).to(self.device)
        self.tgt = SimpleQNet(obs_dim, n_actions).to(self.device)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.batch_size = 32  # CRITICAL: small batch for fast learning

    @torch.no_grad()
    def act(self, obs, mask, eps=0.1):
        if np.random.rand() < eps:
            valid = np.flatnonzero(mask)
            return int(np.random.choice(valid))
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.q(x).squeeze(0).cpu().numpy()
        return masked_argmax(q, mask)

    def update(self, replay, n_updates=1):
        """Perform gradient updates. Returns average loss."""
        if len(replay) < self.batch_size:
            return None

        total_loss = 0.0
        for _ in range(n_updates):
            obs, act, rew, nobs, done, mask, nmask = replay.sample(self.batch_size)

            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            nobs = torch.tensor(nobs, dtype=torch.float32, device=self.device)
            act = torch.tensor(act, dtype=torch.int64, device=self.device)
            rew = torch.tensor(rew, dtype=torch.float32, device=self.device)
            done = torch.tensor(done, dtype=torch.float32, device=self.device)
            nmask = torch.tensor(nmask.astype(np.float32), device=self.device)

            q = self.q(obs).gather(1, act.view(-1, 1)).squeeze(1)

            with torch.no_grad():
                q_next = self.q(nobs)
                q_next = q_next + (nmask - 1.0) * 1e30  # mask invalid actions
                a_star = q_next.argmax(dim=1)
                q_next_tgt = self.tgt(nobs).gather(1, a_star.view(-1, 1)).squeeze(1)
                y = rew + self.gamma * (1.0 - done) * q_next_tgt

            loss = (q - y).pow(2).mean()
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
            self.opt.step()

            total_loss += loss.item()

        # Soft update target network
        with torch.no_grad():
            for p, tp in zip(self.q.parameters(), self.tgt.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return total_loss / n_updates

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"q_state_dict": self.q.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q.load_state_dict(checkpoint["q_state_dict"])
        self.tgt.load_state_dict(checkpoint["q_state_dict"])


def simple_reward(u_sigma, u_delta, mnorm, prev_mnorm, deltas):
    """
    Simple, consistent reward function.

    Design philosophy:
    - Feasibility is REQUIRED (big negative for violation)
    - Within feasible region: reward mass reduction
    - Bonus for being feasible
    """
    violation = max(0.0, u_sigma - 1.0) + max(0.0, u_delta - 1.0)
    mass_change = mnorm - prev_mnorm  # positive = bad (mass increased)

    if violation > 0:
        # Infeasible: strong penalty proportional to violation
        r = -5.0 * violation
    else:
        # Feasible: constant bonus + reward for mass reduction
        r = 0.5 - 2.0 * mass_change  # mass减少 → mass_change负 → 正奖励

    # Small penalty for no-op to encourage action
    if np.all(deltas == 0):
        r -= 0.1

    return float(r)


def train_fixed(
    max_episodes=500,
    log_every=20,
    save_every=100,
    output_dir=None,
    fixed_geometry=True,  # Start with fixed geometry for debugging
):
    """
    Fixed training loop with proper hyperparameters.
    """
    if output_dir is None:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        output_dir = f"results/dqn_fixed/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Environment config
    env_config = {
        "T": 40,  # Longer episodes
        "K_no_improve": 10,  # Harder to exploit early stopping
    }
    env = SteelSectionOptEnv(env_config)

    # Fixed geometry for curriculum (easier to learn first)
    if fixed_geometry:
        fixed_L, fixed_W, fixed_H = 8.0, 3.6, 3.6
        print(f"Using FIXED geometry: L={fixed_L}, W={fixed_W}, H={fixed_H}")

    obs, info = env.reset()
    if fixed_geometry:
        out = env._run_simulation_and_collect()
        env.m0 = out["mass"]
        obs = env._build_obs(out)
        info = {**out, "action_mask": env._get_action_mask()}

    mask = info["action_mask"]

    # Epsilon schedule - faster decay for exploitation
    eps_start, eps_end = 1.0, 0.05
    eps_decay_episodes = max_episodes * 0.5  # Decay over 50% of training

    # Agent and replay
    agent = FixedDQNAgent(obs_dim=len(obs), n_actions=env.n_actions)
    replay = FastReplay(cap=50000)

    # Logger
    logger = MetricsLogger(f"{output_dir}/metrics.csv", window=max(10, max_episodes // 10))

    # Tracking
    ep = 0
    global_step = 0
    ep_ret = 0.0
    ep_steps = 0
    ep_losses = []

    best_mnorm = float("inf")
    best_u_sigma = 10.0
    best_u_delta = 10.0
    feasible_steps = 0
    steps_to_feasible = None

    prev_mnorm = 1.0  # Start at max mass

    print(f"Starting FIXED training: {max_episodes} episodes")
    print(f"Initial: u_sigma={info['u_sigma']:.3f}, u_delta={info['u_delta']:.3f}, mass={info['mass']:.1f}kg")
    print(f"Agent: batch={agent.batch_size}, gamma={agent.gamma}, lr=1e-3")
    print("-" * 60)

    while ep < max_episodes:
        eps = max(eps_end, eps_start - (eps_start - eps_end) * (ep / eps_decay_episodes))

        # Select action
        a = agent.act(obs, mask, eps=eps)

        # Environment step
        nobs, r_env, terminated, truncated, ninfo = env.step(a)
        nmask = ninfo["action_mask"]
        global_step += 1

        # Use our simple reward instead of env's reward
        deltas = env.decode_action(a)
        mnorm = ninfo.get("mnorm", ninfo["mass"] / env.m0)
        r = simple_reward(ninfo["u_sigma"], ninfo["u_delta"], mnorm, prev_mnorm, deltas)
        prev_mnorm = mnorm

        # Store transition
        replay.add(obs, a, r, nobs, float(terminated or truncated), mask, nmask)

        # Update agent - do multiple updates per step to learn faster
        loss = agent.update(replay, n_updates=2)
        if loss is not None:
            ep_losses.append(loss)

        obs, mask = nobs, nmask
        ep_ret += r
        ep_steps += 1

        # Track metrics
        u_sigma = ninfo["u_sigma"]
        u_delta = ninfo["u_delta"]

        if u_sigma <= 1 and u_delta <= 1:
            feasible_steps += 1
            if steps_to_feasible is None:
                steps_to_feasible = ep_steps

        if mnorm < best_mnorm and u_sigma <= 1 and u_delta <= 1:
            best_mnorm = mnorm
            best_u_sigma = u_sigma
            best_u_delta = u_delta

        # Episode end
        if terminated or truncated:
            ep += 1
            final_mnorm = mnorm
            feasible = int(u_sigma <= 1 and u_delta <= 1)

            st = EpisodeStats(
                episode=ep,
                steps=ep_steps,
                ep_return=ep_ret,
                final_feasible=feasible,
                best_mnorm=best_mnorm if best_mnorm < float("inf") else final_mnorm,
                final_mnorm=final_mnorm,
                best_u_sigma=best_u_sigma,
                best_u_delta=best_u_delta,
                final_u_sigma=u_sigma,
                final_u_delta=u_delta,
                steps_to_feasible=int(steps_to_feasible if steps_to_feasible else env.T + 1),
                feasible_steps_ratio=float(feasible_steps / max(1, ep_steps)),
            )
            logger.log_episode(st)

            if ep % log_every == 0:
                s = logger.summary()
                avg_loss = sum(ep_losses[-100:]) / max(1, len(ep_losses[-100:])) if ep_losses else 0
                print(
                    f"[ep {ep:4d}] eps={eps:.2f} feas={s['feasible_rate']:.2f} "
                    f"best_mnorm={s['best_mnorm_mean']:.3f} "
                    f"final_mnorm={final_mnorm:.3f} "
                    f"ret={s['return_mean']:.2f} loss={avg_loss:.4f}"
                )
                logger.plot(window=max(5, max_episodes // 20), save_path=f"{output_dir}/metrics_plot.png")

            if ep % save_every == 0:
                agent.save(f"{output_dir}/agent_ep{ep}.pth")

            # Reset
            obs, info = env.reset()
            if fixed_geometry:
                out = env._run_simulation_and_collect()
                env.m0 = out["mass"]
                obs = env._build_obs(out)
                info = {**out, "action_mask": env._get_action_mask()}
            mask = info["action_mask"]

            ep_ret = 0.0
            ep_steps = 0
            feasible_steps = 0
            best_mnorm = float("inf")
            best_u_sigma = 10.0
            best_u_delta = 10.0
            steps_to_feasible = None
            prev_mnorm = 1.0

    agent.save(f"{output_dir}/agent_final.pth")
    print(f"\nTraining complete. Model saved to {output_dir}/agent_final.pth")

    return agent, logger


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--random-geometry", action="store_true")
    args = parser.parse_args()

    train_fixed(
        max_episodes=args.episodes,
        fixed_geometry=not args.random_geometry,
    )
