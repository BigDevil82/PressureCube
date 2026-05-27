"""
Training script for Steel Section Optimization using Stable-Baselines3.

Uses MaskablePPO from sb3-contrib for action masking support.
Extended version: optimizes both section indices and topology parameters.
"""

import argparse
import os
from datetime import datetime
from typing import Callable, Dict, Optional

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from src.reinforcement.sb3.env import SteelSectionOptEnvSB3


def mask_fn(env: SteelSectionOptEnvSB3) -> np.ndarray:
    """Get action mask from environment for MultiDiscrete action space."""
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
    - custom/ep_final_feas_rate: Rate of feasible final states
    - custom/ep_final_mnorm_mean: Mean normalized mass at episode end
    - custom/ep_final_mnorm_min: Best (min) normalized mass achieved
    - custom/ep_final_u_sigma_mean: Mean stress utilization
    - custom/ep_final_u_delta_mean: Mean displacement utilization
    - custom/ep_best_feasible_mnorm: Best mnorm among feasible episodes
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

        # 1. 临时缓存：用于记录每个并行环境正在进行的回合中的历史最佳
        # 格式：self.current_best[env_idx] = {'mnorm': inf, 'u_sigma': ..., 'u_delta': ...}
        self.current_best = []

        # 2. 结果仓库：用于存放到 TensorBoard 的数据 (每个 Rollout 结算一次)
        self.rollout_best_mnorms = []
        self.rollout_best_u_sigmas = []
        self.rollout_best_u_deltas = []

        # 3. 逐步可行性统计
        self.step_feasible_count = 0
        self.step_total_count = 0

    def _init_callback(self) -> None:
        # 初始化并行环境的缓存，数量等于 n_envs
        # 这里的 mnorm 初始设为无穷大，表示还没找到可行解
        num_envs = self.training_env.num_envs
        self.current_best = [{"mnorm": float("inf"), "u_sigma": 0.0, "u_delta": 0.0} for _ in range(num_envs)]

    def _on_step(self) -> bool:
        # 获取当前这一步所有环境的信息
        infos = self.locals["infos"]
        dones = self.locals["dones"]
        num_envs = len(infos)

        for i in range(num_envs):
            info = infos[i]
            done = dones[i]

            # --- A. 统计逐步可行率 (Step-wise Feasibility) ---
            is_feasible = info.get("feasible", False)
            if is_feasible:
                self.step_feasible_count += 1
            self.step_total_count += 1

            # --- B. 更新当前回合的最佳纪录 (Best Feasible Tracking) ---
            # 只有当当前步是 feasible 的，且质量比之前记录的更小，才更新
            current_mnorm = info.get("mnorm", float("inf"))

            if is_feasible and current_mnorm < self.current_best[i]["mnorm"]:
                self.current_best[i] = {
                    "mnorm": current_mnorm,
                    "u_sigma": info.get("u_sigma", 0.0),
                    "u_delta": info.get("u_delta", 0.0),
                }

            # --- C. 回合结束处理 (Episode Termination) ---
            if done:
                # 1. 只有当这个回合里至少出现过一次可行解时，才记录数据
                best_rec = self.current_best[i]
                if best_rec["mnorm"] != float("inf"):
                    self.rollout_best_mnorms.append(best_rec["mnorm"])
                    self.rollout_best_u_sigmas.append(best_rec["u_sigma"])
                    self.rollout_best_u_deltas.append(best_rec["u_delta"])

                # 2. 重置该环境的缓存，准备迎接下一个 Episode
                self.current_best[i] = {"mnorm": float("inf"), "u_sigma": 0.0, "u_delta": 0.0}

        return True

    def _on_rollout_end(self) -> None:
        """
        每训练一轮（默认为 512 步），计算一次统计数据写入 TensorBoard
        """
        # 1. 记录逐步可行率 (Global Step Feasibility Rate)
        if self.step_total_count > 0:
            feas_rate = self.step_feasible_count / self.step_total_count
            self.logger.record("custom/step_feasible_rate", feas_rate)

        # 重置步数计数器
        self.step_feasible_count = 0
        self.step_total_count = 0

        # 2. 记录“回合最佳”的统计值
        if self.rollout_best_mnorms:
            self.logger.record("custom/ep_best_mnorm_mean", np.mean(self.rollout_best_mnorms))
            self.logger.record("custom/ep_best_mnorm_min", np.min(self.rollout_best_mnorms))
            # 记录对应的应力和位移（这是对应那个最轻时刻的，不是平均值）
            self.logger.record("custom/ep_best_u_sigma_mean", np.mean(self.rollout_best_u_sigmas))
            self.logger.record("custom/ep_best_u_delta_mean", np.mean(self.rollout_best_u_deltas))

        # 清空列表
        self.rollout_best_mnorms.clear()
        self.rollout_best_u_sigmas.clear()
        self.rollout_best_u_deltas.clear()


def train(
    total_timesteps: int = 500_000,
    n_envs: int = 4,
    eval_mode: str = "surrogate",
    surrogate_path: Optional[str] = None,
    layout_strategy: str = "research",
    resume_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    learning_rate: float = 3e-4,
    n_steps: int = 256,  # Reduced for faster logging (was 2048)
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    seed: int = 42,
    fixed_geometry: Optional[tuple] = None,
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
        layout_strategy: "research" or "engineering"
        output_dir: Output directory for logs and checkpoints
        resume_path: Path to resume model
        learning_rate: Learning rate
        n_steps: Steps per rollout
        batch_size: Minibatch size
        n_epochs: Number of epochs per update
        gamma: Discount factor
        seed: Random seed
        fixed_geometry: (L, W, H) tuple or None for random
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
    print(f"Eval mode: {eval_mode}, layout: {layout_strategy}, n_envs: {n_envs}")

    # Environment config
    env_config = {
        "eval_mode": eval_mode,
        "surrogate_path": surrogate_path,
        "layout_strategy": layout_strategy,
        "fixed_geometry": fixed_geometry,
        "T": 30,
        "K_no_improve": 5,
        "alpha": 20.0,
        "beta": 1.0,
    }

    # Create vectorized environments
    # Use SubprocVecEnv for parallel sampling (faster with surrogate mode)
    if n_envs == 1:
        train_env = DummyVecEnv([make_env(env_config, 0, seed)])
    else:
        train_env = SubprocVecEnv(
            [make_env(env_config, i, seed) for i in range(n_envs)],
            start_method="spawn",
        )

    train_env = VecMonitor(train_env)

    # Initialize model
    # Larger network for extended observation space (26-dim)
    if resume_path is None:
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
                "net_arch": dict(pi=[256, 256], vf=[256, 256]),
            },
        )
    else:
        # Resume from checkpoint if provided
        print(f"Resuming training from: {resume_path}")
        model = MaskablePPO.load(resume_path, env=train_env)

    print(f"Model initialized: {model.policy}")

    # Callbacks
    # Note: EvalCallback doesn't work with MaskablePPO (action_masks issue)
    # Use CheckpointCallback for periodic saves, MetricsCallback for logging
    checkpoint_callback = CheckpointCallback(
        save_freq=max(save_freq // n_envs, 1),
        save_path=f"{output_dir}/checkpoints",
        name_prefix="model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    metrics_callback = MetricsCallback(verbose=verbose)

    # Train
    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, metrics_callback],
        progress_bar=True,
    )

    # Save final model
    model.save(f"{output_dir}/final_model")
    print(f"Training complete. Model saved to {output_dir}/final_model.zip")

    # Cleanup
    train_env.close()

    return model


def main():
    parser = argparse.ArgumentParser(description="Train SB3 MaskablePPO for steel section optimization")

    parser.add_argument("--total_timesteps", type=int, default=500_000, help="Total training timesteps")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--eval_mode", type=str, default="surrogate", choices=["fea", "surrogate"])
    parser.add_argument("--surrogate_path", type=str, default=None, help="Path to surrogate model")
    parser.add_argument(
        "--layout_strategy",
        type=str,
        default="research",
        choices=["research", "engineering"],
        help="Layout strategy used by the RL environment.",
    )
    parser.add_argument("--resume_path", type=str, default=None, help="Path to resume model")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n_steps", type=int, default=128, help="Steps per rollout")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fixed_L", type=float, default=None, help="Fixed length (optional)")
    parser.add_argument("--fixed_W", type=float, default=None, help="Fixed width (optional)")
    parser.add_argument("--fixed_H", type=float, default=None, help="Fixed height (optional)")
    parser.add_argument("--save_freq", type=int, default=50_000, help="Checkpoint save frequency")
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
        layout_strategy=args.layout_strategy,
        output_dir=args.output_dir,
        resume_path=args.resume_path,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        seed=args.seed,
        fixed_geometry=fixed_geometry,
        save_freq=args.save_freq,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
