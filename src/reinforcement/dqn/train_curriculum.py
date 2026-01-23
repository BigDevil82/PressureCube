"""
Curriculum Learning DQN for Generalized Steel Section Optimization.

This training script implements a multi-phase curriculum to enable the agent
to generalize across different geometries and loads:

Phase 1: Fixed geometry, fixed load (easiest - verify learning works)
Phase 2: Fixed geometry, random load (learn load adaptation)
Phase 3: Random geometry from small set (learn geometry adaptation)
Phase 4: Full random geometry + load (full generalization)

Key techniques:
1. Curriculum Learning: Gradually increase task difficulty
2. Larger network: More capacity for generalization
3. Better state representation: Include geometry-aware features
4. Longer training: Generalization needs more samples
5. Experience replay with geometric diversity
"""

import os
import random
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.reinforcement.dqn.env import SteelSectionOptEnv
from src.reinforcement.dqn.metric import EpisodeStats, MetricsLogger


class CurriculumPhase(Enum):
    """Training curriculum phases."""

    PHASE1_FIXED = 1  # Fixed geometry, fixed load
    PHASE2_RANDOM_LOAD = 2  # Fixed geometry, random load
    PHASE3_FEW_GEOM = 3  # Few geometries, random load
    PHASE4_FULL_RANDOM = 4  # Full random


@dataclass
class PhaseConfig:
    """Configuration for a curriculum phase."""

    name: str
    geometries: list[tuple[float, float, float]]  # List of (L, W, H)
    pressure_range: tuple[float, float]  # (min, max) in Pa
    episodes: int
    eps_start: float
    eps_end: float


# Default curriculum configuration
DEFAULT_CURRICULUM = [
    PhaseConfig(
        name="Phase 1: Fixed Geometry",
        geometries=[(8.0, 3.7, 3.3)],
        pressure_range=(30000.0, 30000.0),
        episodes=300,
        eps_start=1.0,
        eps_end=0.1,
    ),
    PhaseConfig(
        name="Phase 2: Random Load",
        geometries=[(8.0, 3.7, 3.3)],
        pressure_range=(20000.0, 40000.0),
        episodes=300,
        eps_start=0.3,
        eps_end=0.1,
    ),
    PhaseConfig(
        name="Phase 3: Few Geometries",
        geometries=[
            (6.0, 3.0, 3.0),
            (8.0, 3.7, 3.3),
            (10.0, 4.2, 4.2),
        ],
        pressure_range=(20000.0, 40000.0),
        episodes=500,
        eps_start=0.3,
        eps_end=0.05,
    ),
    PhaseConfig(
        name="Phase 4: Full Random",
        geometries=None,  # None means sample from full range
        pressure_range=(20000.0, 40000.0),
        episodes=1000,
        eps_start=0.2,
        eps_end=0.05,
    ),
]


class GeneralizationQNet(nn.Module):
    """
    Larger network with geometry-aware architecture.

    Input features:
    - Geometry: L, W, H (normalized)
    - Section indices: 4 values (normalized)
    - Constraints: u_sigma, u_delta
    - Mass: normalized mass
    - Control info: argmax_sigma, argmax_delta

    Architecture:
    - Wider hidden layers for more capacity
    - Separate geometry encoder branch (optional)
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()

        # Main network with residual-like structure
        self.input_layer = nn.Linear(obs_dim, hidden_dim)
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)

        # Dueling architecture
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )

        self.relu = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input projection
        h = self.relu(self.input_layer(x))

        # Hidden layers with residual connections
        h1 = self.relu(self.hidden1(h))
        h1 = self.layer_norm1(h1 + h)  # Residual

        h2 = self.relu(self.hidden2(h1))
        h2 = self.layer_norm2(h2 + h1)  # Residual

        # Dueling output
        value = self.value_stream(h2)
        advantage = self.advantage_stream(h2)

        # Q = V + (A - mean(A))
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q


class PrioritizedReplay:
    """
    Prioritized Experience Replay with geometric diversity.

    Stores transitions with priority based on TD error.
    Also tracks geometry to ensure diverse sampling.
    """

    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta  # Importance sampling exponent
        self.beta_increment = 0.001

        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(self, obs, act, rew, nobs, done, mask, nmask, priority: float = 1.0):
        """Add transition with given priority."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (obs, act, rew, nobs, done, mask, nmask)
        self.priorities[self.position] = priority**self.alpha

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """Sample batch with prioritized sampling."""
        if self.size < batch_size:
            return None

        # Calculate sampling probabilities
        priorities = self.priorities[: self.size]
        probs = priorities / priorities.sum()

        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)

        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Get samples
        samples = [self.buffer[i] for i in indices]
        obs, act, rew, nobs, done, mask, nmask = zip(*samples)

        return (
            np.array(obs),
            np.array(act),
            np.array(rew),
            np.array(nobs),
            np.array(done),
            np.array(mask),
            np.array(nmask),
            indices,
            weights.astype(np.float32),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = (abs(td_error) + 1e-6) ** self.alpha

    def __len__(self):
        return self.size


class CurriculumDQNAgent:
    """
    DQN Agent with curriculum learning support.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 5e-4,
        gamma: float = 0.9,
        tau: float = 0.005,
        hidden_dim: int = 128,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.q = GeneralizationQNet(obs_dim, n_actions, hidden_dim).to(self.device)
        self.target = GeneralizationQNet(obs_dim, n_actions, hidden_dim).to(self.device)
        self.target.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.9)

        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.batch_size = 64

    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: np.ndarray, eps: float = 0.1) -> int:
        if np.random.rand() < eps:
            valid = np.flatnonzero(mask)
            return int(np.random.choice(valid))

        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.q(x).squeeze(0).cpu().numpy()

        # Mask invalid actions
        q[~mask] = -1e30
        return int(q.argmax())

    def update(self, replay: PrioritizedReplay, n_updates: int = 1) -> Optional[float]:
        """Perform gradient updates with prioritized replay."""
        sample = replay.sample(self.batch_size)
        if sample is None:
            return None

        obs, act, rew, nobs, done, mask, nmask, indices, weights = sample

        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        nobs = torch.tensor(nobs, dtype=torch.float32, device=self.device)
        act = torch.tensor(act, dtype=torch.int64, device=self.device)
        rew = torch.tensor(rew, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)
        nmask = torch.tensor(nmask.astype(np.float32), device=self.device)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        # Current Q values
        q_values = self.q(obs).gather(1, act.view(-1, 1)).squeeze(1)

        # Target Q values (Double DQN)
        with torch.no_grad():
            q_next = self.q(nobs)
            q_next[nmask == 0] = -1e30
            next_actions = q_next.argmax(dim=1)

            q_target_next = self.target(nobs).gather(1, next_actions.view(-1, 1)).squeeze(1)
            targets = rew + self.gamma * (1 - done) * q_target_next

        # TD errors for priority update
        td_errors = (q_values - targets).detach().cpu().numpy()
        replay.update_priorities(indices, td_errors)

        # Weighted loss
        loss = (weights * (q_values - targets) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.optimizer.step()

        # Soft update target network
        with torch.no_grad():
            for p, tp in zip(self.q.parameters(), self.target.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return float(loss.item())

    def step_scheduler(self):
        """Step the learning rate scheduler."""
        self.scheduler.step()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "q_state_dict": self.q.state_dict(),
                "target_state_dict": self.target.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q.load_state_dict(checkpoint["q_state_dict"])
        self.target.load_state_dict(checkpoint["target_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def generalization_reward(
    u_sigma: float,
    u_delta: float,
    mnorm: float,
    prev_mnorm: float,
    deltas: np.ndarray,
    feasible_prev: bool,
    feasible_now: bool,
) -> float:
    """
    Reward function designed for generalization.

    Key design choices:
    1. Strong feasibility signal (agents must learn constraints matter)
    2. Relative mass improvement (works across different geometries)
    3. Bonus for staying feasible while reducing mass
    """
    violation = max(0, u_sigma - 1.0) + max(0, u_delta - 1.0)
    mass_change = mnorm - prev_mnorm  # positive = mass increased

    if violation > 0:
        # Infeasible: penalty proportional to violation
        r = -5.0 * violation - 0.1  # Small step penalty
    else:
        # Feasible: reward for mass reduction
        r = 0.5 - 3.0 * mass_change  # Higher coefficient for mass

        # Bonus for significant mass reduction
        if mass_change < -0.02:
            r += 0.3

    # Penalty for leaving feasible region
    if feasible_prev and not feasible_now:
        r -= 1.0

    # Small penalty for no-op
    if np.all(deltas == 0):
        r -= 0.1

    return float(r)


def sample_geometry(
    rng: np.random.Generator,
    geometries: Optional[list[tuple[float, float, float]]] = None,
) -> tuple[float, float, float]:
    """Sample geometry from list or full range."""
    if geometries is not None:
        idx = rng.integers(len(geometries))
        return geometries[idx]
    else:
        # Full random from range
        L = float(round(rng.choice(np.arange(6.0, 12.2, 0.2)), 2))
        W = float(round(rng.choice(np.arange(3.0, 4.6, 0.3)), 2))
        H = float(round(rng.choice(np.arange(3.0, 4.6, 0.3)), 2))
        return L, W, H


def sample_pressure(
    rng: np.random.Generator,
    pressure_range: tuple[float, float],
) -> float:
    """Sample pressure from range."""
    p_min, p_max = pressure_range
    if p_min == p_max:
        return p_min
    return float(rng.uniform(p_min, p_max))


def sample_secs(
    rng: np.random.Generator,
    sec_counts: List[int],
) -> np.ndarray:
    """Sample section indices."""
    return np.array(
        [rng.choice(np.arange(0, sec_counts[i])) for i in range(len(sec_counts))],
        dtype=np.int64,
    )


def train_with_curriculum(
    curriculum: list[PhaseConfig] = None,
    output_dir: str = None,
    resume_from: str = None,
    start_phase: int = 1,
    seed: int = 42,
):
    """
    Train DQN with curriculum learning.

    Args:
        curriculum: List of PhaseConfig for each phase
        output_dir: Directory for outputs
        resume_from: Path to checkpoint to resume from
        start_phase: Which phase to start from (1-indexed)
    """
    if curriculum is None:
        curriculum = DEFAULT_CURRICULUM

    if output_dir is None:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        output_dir = f"results/dqn_curriculum/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize environment
    env = SteelSectionOptEnv({"T": 40, "K_no_improve": 15})

    # Get observation dimension (need to reset once to get obs)
    obs, info = env.reset()
    obs_dim = len(obs)

    # Initialize agent
    agent = CurriculumDQNAgent(
        obs_dim=obs_dim,
        n_actions=env.n_actions,
        lr=5e-4,
        gamma=0.9,
        hidden_dim=128,
    )

    # Resume from checkpoint if specified
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from {resume_from}")
        agent.load(resume_from)

    # Initialize replay buffer
    replay = PrioritizedReplay(capacity=100000)

    # Logger
    logger = MetricsLogger(f"{output_dir}/metrics.csv", window=50)

    print("=" * 70)
    print("CURRICULUM LEARNING DQN TRAINING")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print(f"Phases: {len(curriculum)}")
    for i, phase in enumerate(curriculum):
        print(f"  Phase {i+1}: {phase.name} ({phase.episodes} episodes)")
    print("=" * 70)

    global_episode = 0
    rng = np.random.default_rng(seed)

    # Run each phase
    for phase_idx, phase in enumerate(curriculum):
        if phase_idx + 1 < start_phase:
            global_episode += phase.episodes
            continue

        print(f"\n{'='*70}")
        print(f"PHASE {phase_idx + 1}: {phase.name}")
        print(f"{'='*70}")
        print(f"Geometries: {phase.geometries if phase.geometries else 'Full random'}")
        print(f"Pressure: {phase.pressure_range}")
        print(f"Episodes: {phase.episodes}")
        print(f"Epsilon: {phase.eps_start} -> {phase.eps_end}")

        for ep_in_phase in range(phase.episodes):
            global_episode += 1

            # Epsilon schedule for this phase
            progress = ep_in_phase / max(1, phase.episodes - 1)
            eps = phase.eps_start + (phase.eps_end - phase.eps_start) * progress

            # Sample geometry and pressure
            L, W, H = sample_geometry(rng, phase.geometries)
            pressure = sample_pressure(rng, phase.pressure_range)
            init_secs = sample_secs(rng, env.sec_counts)

            # Reset environment with sampled parameters
            obs, info = env.reset(
                seed=seed, options={"L": L, "W": W, "H": H, "pressure": pressure, "init_sections": init_secs}
            )

            mask = info["action_mask"]
            init_secs = env.idx.copy()
            prev_mnorm = env.prev_mnorm
            feasible_prev = env.prev_v <= 0

            ep_return = 0.0
            ep_steps = 0
            ep_losses = []
            best_mnorm = float("inf")
            best_u_sigma = 10.0
            best_u_delta = 10.0
            feasible_steps = 0
            steps_to_feasible = None
            best_secs = init_secs.copy()

            done = False
            while not done and ep_steps < env.T:
                # Select action
                action = agent.act(obs, mask, eps)
                deltas = env.decode_action(action)

                # Step environment
                nobs, _, terminated, truncated, ninfo = env.step(action)
                nmask = ninfo["action_mask"]
                done = terminated or truncated

                # Calculate reward
                mnorm = ninfo.get("mnorm", ninfo["mass"] / env.m0)
                u_sigma = ninfo["u_sigma"]
                u_delta = ninfo["u_delta"]
                violation = ninfo["violation"]
                feasible_now = violation <= 0

                reward = generalization_reward(
                    u_sigma, u_delta, mnorm, prev_mnorm, deltas, feasible_prev, feasible_now
                )

                # Store transition with high initial priority
                replay.add(obs, action, reward, nobs, float(done), mask, nmask, priority=1.0)

                # Update agent
                loss = agent.update(replay)
                if loss is not None:
                    ep_losses.append(loss)

                # Track metrics
                ep_return += reward
                ep_steps += 1

                if feasible_now:
                    feasible_steps += 1
                    if steps_to_feasible is None:
                        steps_to_feasible = ep_steps

                if mnorm < best_mnorm and feasible_now:
                    best_mnorm = mnorm
                    best_u_sigma = u_sigma
                    best_u_delta = u_delta
                    best_secs = env.idx.copy()

                # Update state
                obs = nobs
                mask = nmask
                prev_mnorm = mnorm
                feasible_prev = feasible_now

            # Log episode
            final_feasible = int(feasible_now)
            st = EpisodeStats(
                episode=global_episode,
                steps=ep_steps,
                ep_return=ep_return,
                final_feasible=final_feasible,
                best_mnorm=best_mnorm if best_mnorm < float("inf") else mnorm,
                final_mnorm=mnorm,
                best_u_sigma=best_u_sigma,
                best_u_delta=best_u_delta,
                final_u_sigma=u_sigma,
                final_u_delta=u_delta,
                steps_to_feasible=int(steps_to_feasible if steps_to_feasible else env.T + 1),
                feasible_steps_ratio=float(feasible_steps / max(1, ep_steps)),
                pressure=pressure,
                size=f"(L={L:.1f},W={W:.1f},H={H:.1f})",
                init_secs=str(init_secs.tolist()),
                best_secs=str(best_secs.tolist()),
            )
            logger.log_episode(st)

            # Print progress
            if (ep_in_phase + 1) % 20 == 0:
                s = logger.summary()
                avg_loss = np.mean(ep_losses[-100:]) if ep_losses else 0
                print(
                    f"[Phase {phase_idx+1} Ep {ep_in_phase+1:4d}/{phase.episodes}] "
                    f"eps={eps:.2f} feas={s['feasible_rate']:.2f} "
                    f"best_mnorm={s['best_mnorm_mean']:.3f} "
                    f"ret={s['return_mean']:.2f} loss={avg_loss:.4f} "
                    f"(L={L:.1f},W={W:.1f},H={H:.1f})"
                    f" P={pressure/1000:.1f}kPa"
                )
                logger.plot(window=max(5, phase.episodes // 20), save_path=f"{output_dir}/metrics_plot.png")

        # Save checkpoint at end of each phase
        agent.save(f"{output_dir}/agent_phase{phase_idx+1}.pth")
        logger.plot(window=20, save_path=f"{output_dir}/metrics_phase{phase_idx+1}.png")
        print(f"\nPhase {phase_idx+1} complete. Checkpoint saved.")

        # Step learning rate scheduler
        agent.step_scheduler()

    # Save final model
    agent.save(f"{output_dir}/agent_final.pth")
    print(f"\n{'='*70}")
    print(f"Training complete! Final model: {output_dir}/agent_final.pth")
    print(f"{'='*70}")

    return agent, logger


def evaluate_generalization(
    agent_path: str,
    n_test_structures: int = 20,
    seed: int = 123,
):
    """
    Evaluate trained agent on unseen geometries.

    Args:
        agent_path: Path to trained agent checkpoint
        n_test_structures: Number of test structures
        seed: Random seed for reproducibility
    """
    print("=" * 70)
    print("GENERALIZATION EVALUATION")
    print("=" * 70)

    # Create test geometries
    rng = np.random.default_rng(seed)
    test_geometries = []
    for _ in range(n_test_structures):
        L = float(round(rng.choice(np.arange(6.0, 12.2, 0.4)), 2))
        W = float(round(rng.choice(np.arange(3.0, 4.6, 0.3)), 2))
        H = float(round(rng.choice(np.arange(3.0, 4.6, 0.3)), 2))
        P = float(rng.uniform(20000, 40000))
        test_geometries.append({"L": L, "W": W, "H": H, "P": P})

    # Load agent
    env = SteelSectionOptEnv({"T": 40})
    obs, _ = env.reset(seed=seed)

    agent = CurriculumDQNAgent(
        obs_dim=len(obs),
        n_actions=env.n_actions,
        hidden_dim=128,
    )
    agent.load(agent_path)

    results = []

    for i, geom in enumerate(test_geometries):
        L, W, H, P = geom["L"], geom["W"], geom["H"], geom["P"]

        # Reset with test geometry
        obs, info = env.reset(seed=seed, options={"L": L, "W": W, "H": H, "pressure": P})
        mask = info["action_mask"]

        # Run episode with greedy policy (eps=0)
        done = False
        steps = 0
        best_mnorm = 1.0
        best_sections = env.idx.copy()

        while not done and steps < env.T:
            action = agent.act(obs, mask, eps=0.0)
            obs, _, terminated, truncated, info = env.step(action)
            mask = info["action_mask"]
            done = terminated or truncated
            steps += 1

            mnorm = info.get("mnorm", info["mass"] / env.m0)
            feasible = info["u_sigma"] <= 1 and info["u_delta"] <= 1

            if feasible and mnorm < best_mnorm:
                best_mnorm = mnorm
                best_sections = env.idx.copy()

        # Record result
        final_mass = best_mnorm * env.m0
        reduction = (1 - best_mnorm) * 100

        results.append(
            {
                "geometry": f"L={L:.1f},W={W:.1f},H={H:.1f}",
                "pressure": f"{P/1000:.0f}kPa",
                "max_mass": env.m0,
                "final_mass": final_mass,
                "reduction": reduction,
                "sections": best_sections.tolist(),
                "steps": steps,
            }
        )

        print(
            f"[{i+1:2d}/{n_test_structures}] {results[-1]['geometry']} P={results[-1]['pressure']}: "
            f"{reduction:.1f}% reduction in {steps} steps"
        )

    # Summary
    reductions = [r["reduction"] for r in results]
    print(f"\n{'='*70}")
    print(f"SUMMARY: {n_test_structures} test structures")
    print(f"{'='*70}")
    print(f"Mass reduction: {np.mean(reductions):.1f}% ± {np.std(reductions):.1f}%")
    print(f"Min/Max: {np.min(reductions):.1f}% / {np.max(reductions):.1f}%")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Curriculum DQN Training")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate", "quick"])
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--start-phase", type=int, default=1, help="Phase to start training from")
    parser.add_argument("--agent", type=str, default=None, help="Agent path for evaluation")

    args = parser.parse_args()

    if args.mode == "train":
        train_with_curriculum(
            resume_from=args.resume,
            start_phase=args.start_phase,
        )
    elif args.mode == "quick":
        # Quick test with shorter curriculum
        quick_curriculum = [
            PhaseConfig("Quick Phase 1", [(8.0, 3.6, 3.6)], (30000, 30000), 100, 1.0, 0.1),
            PhaseConfig("Quick Phase 2", [(8.0, 3.6, 3.6)], (20000, 40000), 100, 0.2, 0.05),
            PhaseConfig(
                "Quick Phase 3",
                [(6.0, 3.0, 3.0), (8.0, 3.6, 3.6), (10.0, 4.2, 4.2)],
                (20000, 40000),
                200,
                0.2,
                0.05,
            ),
        ]
        train_with_curriculum(curriculum=quick_curriculum)
    elif args.mode == "evaluate":
        if args.agent is None:
            print("Error: --agent path required for evaluation")
        else:
            evaluate_generalization(args.agent)
