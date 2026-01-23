import os
from datetime import datetime

from src.reinforcement.dqn.agent import DQNAgent, Replay
from src.reinforcement.dqn.env import SteelSectionOptEnv
from src.reinforcement.dqn.metric import EpisodeStats, MetricsLogger


def train(
    max_episodes=100,
    log_every_episodes=10,
    save_every_episodes=10,
    output_dir=None,
):
    """
    Train DQN agent for steel section optimization.

    Args:
        max_episodes: Maximum number of episodes to train
        log_every_episodes: Print summary every N episodes
        save_every_episodes: Save model checkpoint every N episodes
        output_dir: Directory for logs and checkpoints (default: auto-generated)
    """
    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        output_dir = f"results/dqn/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize environment
    env = SteelSectionOptEnv({})
    obs, info = env.reset()
    mask = info["action_mask"]

    # Epsilon schedule
    eps_start, eps_end = 1.0, 0.05
    eps_decay_episodes = 80  # Decay over 80% of training

    # Initialize agent and replay buffer
    agent = DQNAgent(obs_dim=len(obs), n_actions=env.n_actions)
    replay = Replay(cap=50000)

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

    print(f"Starting training: {max_episodes} episodes, output: {output_dir}")
    print(
        f"Initial state: u_sigma={info['u_sigma']:.3f}, u_delta={info['u_delta']:.3f}, mass={info['mass']:.1f}kg"
    )

    while ep < max_episodes:
        # Epsilon decay based on episodes
        eps = max(eps_end, eps_start - (eps_start - eps_end) * (ep / eps_decay_episodes))

        # Select action
        a = agent.act(obs, mask, eps=eps)

        # Environment step
        nobs, r, terminated, truncated, ninfo = env.step(a)
        nmask = ninfo["action_mask"]
        global_step += 1

        # Store transition in replay buffer
        replay.add(obs, a, r, nobs, float(terminated or truncated), mask, nmask)

        # CRITICAL: Update the agent's neural network
        loss = agent.update(replay)
        if loss is not None:
            ep_losses.append(loss)

        # Move to next state
        obs, mask = nobs, nmask

        ep_ret += float(r)
        ep_steps += 1

        # Track best feasible state in episode
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
            feasible = int(final_u_sigma <= 1 and final_u_delta <= 1)

            # Log episode stats
            st = EpisodeStats(
                episode=ep,
                steps=ep_steps,
                ep_return=ep_ret,
                final_feasible=feasible,
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
            if ep % log_every_episodes == 0:
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
            if ep % save_every_episodes == 0:
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
    print(f"Training complete. Final model saved to {output_dir}/agent_final.pth")

    return agent, logger


if __name__ == "__main__":
    train(max_episodes=100, log_every_episodes=5, save_every_episodes=20)
