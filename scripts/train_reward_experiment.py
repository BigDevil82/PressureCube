"""
Run DQN training with different reward functions.

Usage:
    python scripts/train_reward_experiment.py --reward balanced --episodes 200
    python scripts/train_reward_experiment.py --reward mass_centric --episodes 200
    python scripts/train_reward_experiment.py --reward potential --episodes 200
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import os
from datetime import datetime

from src.reinforcement.dqn.agent import DQNAgent, Replay
from src.reinforcement.dqn.env import SteelSectionOptEnv
from src.reinforcement.dqn.metric import EpisodeStats, MetricsLogger


def main():
    parser = argparse.ArgumentParser(description="Train DQN with reward function variant")
    parser.add_argument(
        "--reward",
        type=str,
        required=True,
        choices=["original", "balanced", "balanced_v2", "mass_centric", "potential", "potential_v2"],
        help="Reward function to use",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Number of training episodes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/dqn/reward_{reward}_{timestamp})",
    )

    args = parser.parse_args()

    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        output_dir = f"results/dqn/reward_{args.reward}_{timestamp}"
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*80}")
    print(f"DQN Training - Reward Experiment: {args.reward.upper()}")
    print(f"{'='*80}")
    print(f"Reward function: {args.reward}")
    print(f"Episodes: {args.episodes}")
    print(f"Random seed: {args.seed}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")

    # Environment config with reward function
    env_config = {
        "reward_type": args.reward,  # Key parameter!
        "T": 30,
        "K_no_improve": 999,
        "eps_improve": 1e-3,
    }

    # Training hyperparameters
    max_episodes = args.episodes
    replay_cap = 50000
    batch_size = 32
    gamma = 0.95
    lr = 2e-4
    tau = 0.005
    eps_start = 1.0
    eps_end = 0.05
    eps_decay_episodes = int(args.episodes * 0.8)  # Decay over 80%
    log_every = 10
    save_every = 20

    # Initialize environment
    env = SteelSectionOptEnv(env_config)
    obs, info = env.reset()
    mask = info["action_mask"]

    # Initialize agent and replay buffer
    agent = DQNAgent(obs_dim=len(obs), n_actions=env.n_actions, lr=lr, gamma=gamma, tau=tau)
    replay = Replay(cap=replay_cap)

    # Logger
    logger = MetricsLogger(f"{output_dir}/metrics.csv", window=max(10, max_episodes // 10))

    # Training loop variables
    ep = 0
    global_step = 0
    ep_ret = 0.0
    ep_steps = 0
    ep_losses = []

    best_mnorm = float("inf")
    best_u_sigma = 10.0
    best_u_delta = 10.0
    steps_to_feasible = None
    feasible_steps = 0

    print(f"Starting training: {max_episodes} episodes")
    print(
        f"Initial state: u_sigma={info['u_sigma']:.3f}, u_delta={info['u_delta']:.3f}, mass={info['mass']:.1f}kg\n"
    )

    while ep < max_episodes:
        # Epsilon decay
        eps = max(eps_end, eps_start - (eps_start - eps_end) * (ep / eps_decay_episodes))

        # Select action
        a = agent.act(obs, mask, eps=eps)

        # Environment step
        nobs, r, terminated, truncated, ninfo = env.step(a)
        nmask = ninfo["action_mask"]
        global_step += 1

        # Store transition
        replay.add(obs, a, r, nobs, float(terminated or truncated), mask, nmask)

        # Update agent
        if len(replay) >= batch_size:
            loss = agent.update(replay, batch=batch_size)
            if loss is not None:
                ep_losses.append(loss)

        # Move to next state
        obs, mask = nobs, nmask

        ep_ret += float(r)
        ep_steps += 1

        # Track best feasible state
        mnorm = ninfo.get("mnorm", ninfo["mass"] / env.m0)
        u_sigma = ninfo["u_sigma"]
        u_delta = ninfo["u_delta"]

        if u_sigma <= 1 and u_delta <= 1:
            feasible_steps += 1
            if steps_to_feasible is None:
                steps_to_feasible = ep_steps

        if mnorm < best_mnorm and (u_sigma <= 1 and u_delta <= 1):
            best_mnorm = mnorm
            best_u_sigma = u_sigma
            best_u_delta = u_delta

        # Episode end
        if terminated or truncated:
            ep += 1
            final_mnorm = mnorm
            final_u_sigma = u_sigma
            final_u_delta = u_delta
            final_feasible = int(final_u_sigma <= 1 and final_u_delta <= 1)

            # Log episode stats
            st = EpisodeStats(
                episode=ep,
                steps=ep_steps,
                ep_return=ep_ret,
                final_feasible=final_feasible,
                best_mnorm=best_mnorm if best_mnorm < float("inf") else final_mnorm,
                final_mnorm=final_mnorm,
                best_u_sigma=best_u_sigma,
                best_u_delta=best_u_delta,
                final_u_sigma=final_u_sigma,
                final_u_delta=final_u_delta,
                steps_to_feasible=int(steps_to_feasible if steps_to_feasible is not None else env.T + 1),
                feasible_steps_ratio=float(feasible_steps / max(1, ep_steps)),
            )
            logger.log_episode(st)

            # Print progress
            if ep % log_every == 0:
                s = logger.summary()
                avg_loss = sum(ep_losses[-100:]) / max(1, len(ep_losses[-100:])) if ep_losses else 0
                print(
                    f"[ep {ep:4d}] eps={eps:.2f} feas={s['feasible_rate']:.2f} "
                    f"best_mnorm={s['best_mnorm_mean']:.3f} "
                    f"steps2feas={s['steps_to_feasible_mean']:.1f} "
                    f"ret={s['return_mean']:.2f} loss={avg_loss:.4f}"
                )
                logger.plot(window=max(5, max_episodes // 20), save_path=f"{output_dir}/metrics_plot.png")

            # Save checkpoint
            if ep % save_every == 0:
                agent.save(f"{output_dir}/agent_ep{ep}.pth")

            # Reset for next episode
            obs, info = env.reset()
            mask = info["action_mask"]

            ep_ret = 0.0
            ep_steps = 0
            feasible_steps = 0
            best_mnorm = float("inf")
            best_u_sigma = 10.0
            best_u_delta = 10.0
            steps_to_feasible = None

    # Save final model
    agent.save(f"{output_dir}/agent_final.pth")

    print(f"\n{'='*80}")
    print(f"Training complete! Results saved to: {output_dir}")
    print(f"{'='*80}")
    print(f"\nNext steps:")
    print(f"1. Evaluate on test set:")
    print(f'   ".venv\\Scripts\\python.exe" scripts/run_evaluation.py \\')
    print(f"     --agent-path {output_dir}/agent_final.pth \\")
    print(f"     --test-set results/test_structures.json \\")
    print(f"     --output {output_dir}/evaluation_results.csv \\")
    print(f"     --compare-baseline results/baseline/greedy_shrink_results.csv")
    print(f"\n2. View training metrics:")
    print(f"   type {output_dir}\\metrics.csv")


if __name__ == "__main__":
    main()
