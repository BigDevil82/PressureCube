"""
Training script for Steel Section Optimization using Stable-Baselines3.

Uses MaskablePPO from sb3-contrib for action masking support.
"""

import argparse
import os
from datetime import datetime
from typing import Callable, Dict, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from src.reinforcement.sb3.env import SteelSectionOptEnvSB3


def mask_fn(env: SteelSectionOptEnvSB3) -> np.ndarray:
    """Get action mask from environment."""
    return env.action_masks()


def make_env(config: Dict, rank: int, seed: int = 0) -> Callable:
    """Create environment factory function."""

    def _init():
        env = SteelSectionOptEnvSB3(config)
        env = ActionMasker(env, mask_fn)
        env.reset(seed=seed + rank)
        return env

    return _init


class MetricsCallback(BaseCallback):
    """
    Custom callback to log episode-level metrics.

    Logs to TensorBoard:
    - custom/ep_feasible_rate: Rate of feasible final states
    - custom/ep_final_mnorm_mean: Mean normalized mass at episode end
    - custom/ep_final_mnorm_min: Best (min) normalized mass achieved
    - custom/ep_final_u_sigma_mean: Mean stress utilization
    - custom/ep_final_u_delta_mean: Mean displacement utilization
    - custom/ep_best_feasible_mnorm: Best mnorm among feasible episodes
    """

    def __init__(self, verbose: int = 0, log_freq: int = 10):
        super().__init__(verbose)
        self.log_freq = log_freq

        # Episode-level tracking (reset after logging)
        self.ep_final_feasible = []
        self.ep_final_mnorm = []
        self.ep_final_u_sigma = []
        self.ep_final_u_delta = []

        # Global best tracking
        self.best_feasible_mnorm = float("inf")
        self.total_episodes = 0

    def _on_step(self) -> bool:
        # Check for episode termination via "terminal_observation" key
        # This is set by VecEnv when episode ends
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                info = self.locals["infos"][i]
                self.total_episodes += 1

                # Get final state metrics
                feasible = info.get("feasible", False)
                mnorm = info.get("mnorm", 1.0)
                u_sigma = info.get("u_sigma", 1.0)
                u_delta = info.get("u_delta", 1.0)

                self.ep_final_feasible.append(int(feasible))
                self.ep_final_mnorm.append(mnorm)
                self.ep_final_u_sigma.append(u_sigma)
                self.ep_final_u_delta.append(u_delta)

                # Track best feasible
                if feasible and mnorm < self.best_feasible_mnorm:
                    self.best_feasible_mnorm = mnorm

        return True

    def _on_rollout_end(self) -> None:
        # Log aggregated metrics every rollout
        if not self.ep_final_feasible:
            return

        n_eps = len(self.ep_final_feasible)

        # Feasibility rate
        feas_rate = np.mean(self.ep_final_feasible)
        self.logger.record("custom/ep_feasible_rate", feas_rate)
        self.logger.record("custom/ep_count", n_eps)
        self.logger.record("custom/total_episodes", self.total_episodes)

        # Mass metrics
        self.logger.record("custom/ep_final_mnorm_mean", np.mean(self.ep_final_mnorm))
        self.logger.record("custom/ep_final_mnorm_min", np.min(self.ep_final_mnorm))

        # Utilization metrics
        self.logger.record("custom/ep_final_u_sigma_mean", np.mean(self.ep_final_u_sigma))
        self.logger.record("custom/ep_final_u_delta_mean", np.mean(self.ep_final_u_delta))

        # Best feasible
        if self.best_feasible_mnorm < float("inf"):
            self.logger.record("custom/best_feasible_mnorm", self.best_feasible_mnorm)

        # Feasible-only mnorm stats
        feasible_mnorms = [m for m, f in zip(self.ep_final_mnorm, self.ep_final_feasible) if f]
        if feasible_mnorms:
            self.logger.record("custom/ep_feasible_mnorm_mean", np.mean(feasible_mnorms))
            self.logger.record("custom/ep_feasible_mnorm_min", np.min(feasible_mnorms))

        # Clear episode buffers
        self.ep_final_feasible.clear()
        self.ep_final_mnorm.clear()
        self.ep_final_u_sigma.clear()
        self.ep_final_u_delta.clear()


def train(
    total_timesteps: int = 500_000,
    n_envs: int = 4,
    eval_mode: str = "surrogate",
    surrogate_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    learning_rate: float = 3e-4,
    n_steps: int = 256,  # Reduced for faster logging (was 2048)
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    seed: int = 42,
    fixed_geometry: Optional[tuple] = None,
    eval_freq: int = 10_000,
    save_freq: int = 50_000,
    verbose: int = 1,
):
    """
    Train MaskablePPO agent for steel section optimization.

    Args:
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        eval_mode: "fea" or "surrogate"
        surrogate_path: Path to surrogate model (required for surrogate mode)
        output_dir: Output directory for logs and checkpoints
        learning_rate: Learning rate
        n_steps: Steps per rollout
        batch_size: Minibatch size
        n_epochs: Number of epochs per update
        gamma: Discount factor
        seed: Random seed
        fixed_geometry: (L, W, H) tuple or None for random
        eval_freq: Evaluation frequency
        save_freq: Checkpoint save frequency
        verbose: Verbosity level
    """
    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        output_dir = f"results/sb3/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Eval mode: {eval_mode}, n_envs: {n_envs}")

    # Environment config
    env_config = {
        "eval_mode": eval_mode,
        "surrogate_path": surrogate_path,
        "fixed_geometry": fixed_geometry,
        "T": 30,
        "K_no_improve": 5,
        "alpha": 20.0,
        "beta": 1.0,
    }

    # Create vectorized environments
    if n_envs == 1:
        train_env = DummyVecEnv([make_env(env_config, 0, seed)])
    else:
        train_env = SubprocVecEnv(
            [make_env(env_config, i, seed) for i in range(n_envs)],
            start_method="spawn",
        )

    train_env = VecMonitor(train_env)

    # Create eval environment (single env, deterministic)
    eval_env = DummyVecEnv([make_env(env_config, 0, seed + 1000)])
    eval_env = VecMonitor(eval_env)

    # Initialize model
    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        verbose=verbose,
        tensorboard_log=f"{output_dir}/logs",
        seed=seed,
        policy_kwargs={
            "net_arch": dict(pi=[128, 128], vf=[128, 128]),
        },
    )

    print(f"Model initialized: {model.policy}")

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{output_dir}/checkpoints",
        log_path=f"{output_dir}/eval_logs",
        eval_freq=eval_freq // n_envs,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    metrics_callback = MetricsCallback(verbose=verbose)

    # Train
    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, metrics_callback],
        progress_bar=True,
    )

    # Save final model
    model.save(f"{output_dir}/final_model")
    print(f"Training complete. Model saved to {output_dir}/final_model.zip")

    # Cleanup
    train_env.close()
    eval_env.close()

    return model


def main():
    parser = argparse.ArgumentParser(description="Train SB3 MaskablePPO for steel section optimization")

    parser.add_argument("--total_timesteps", type=int, default=500_000, help="Total training timesteps")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--eval_mode", type=str, default="surrogate", choices=["fea", "surrogate"])
    parser.add_argument("--surrogate_path", type=str, default=None, help="Path to surrogate model")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n_steps", type=int, default=2048, help="Steps per rollout")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fixed_L", type=float, default=None, help="Fixed length (optional)")
    parser.add_argument("--fixed_W", type=float, default=None, help="Fixed width (optional)")
    parser.add_argument("--fixed_H", type=float, default=None, help="Fixed height (optional)")
    parser.add_argument("--eval_freq", type=int, default=10_000, help="Evaluation frequency")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")

    args = parser.parse_args()

    # Handle fixed geometry
    fixed_geometry = None
    if args.fixed_L is not None and args.fixed_W is not None and args.fixed_H is not None:
        fixed_geometry = (args.fixed_L, args.fixed_W, args.fixed_H)

    # Validate surrogate mode
    if args.eval_mode == "surrogate" and args.surrogate_path is None:
        print("Warning: surrogate mode requires --surrogate_path. Using default path.")
        args.surrogate_path = "results/surrogate_data/model_advanced.pkl"

    train(
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        eval_mode=args.eval_mode,
        surrogate_path=args.surrogate_path,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        seed=args.seed,
        fixed_geometry=fixed_geometry,
        eval_freq=args.eval_freq,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
