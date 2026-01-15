from src.reinforcement.dqn.agent import DQNAgent, Replay
from src.reinforcement.dqn.env import SteelSectionOptEnv
from src.reinforcement.dqn.metric import EpisodeStats, MetricsLogger


def train(steps=30000, log_every_episodes=10):
    env = SteelSectionOptEnv({})
    obs, info = env.reset()
    mask = info["action_mask"]

    eps_start, eps_end = 1.0, 0.05
    eps_decay_steps = 50000

    agent = DQNAgent(obs_dim=len(obs), n_actions=env.n_actions)
    replay = Replay()

    logger = MetricsLogger("results/dqn/0116/metrics.csv", window=1000)

    obs, info = env.reset()
    mask = info["action_mask"]

    ep = 0
    ep_ret = 0.0
    ep_steps = 0

    best_mnorm = float("inf")
    best_u_sigma = 10.0
    best_u_delta = 10.0
    steps_to_feasible = None
    feasible_steps = 0

    for global_step in range(1, steps + 1):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * (global_step / eps_decay_steps))
        a = agent.act(obs, mask, eps=eps)

        nobs, r, terminated, truncated, ninfo = env.step(a)
        nmask = ninfo["action_mask"]

        # ---- replay add/update（略） ----
        replay.add(obs, a, r, nobs, float(terminated or truncated), mask, nmask)
        obs, mask = nobs, nmask

        ep_ret += float(r)
        ep_steps += 1

        # ---- 追踪 best ----
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

        if terminated or truncated:
            ep += 1
            final_mnorm = mnorm
            final_u_sigma = u_sigma
            final_u_delta = u_delta
            feasible = int(final_u_sigma <= 1 and final_u_delta <= 1)

            st = EpisodeStats(
                episode=ep,
                steps=ep_steps,
                ep_return=ep_ret,
                feasible=feasible,
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

            if ep % log_every_episodes == 0:
                s = logger.summary()
                print(
                    f"[ep {ep:5d}] feas={s['feasible_rate']:.2f} "
                    f"best_mnorm={s['best_mnorm_mean']:.3f} "
                    f"p10={s['best_mnorm_p10']:.3f} p90={s['best_mnorm_p90']:.3f} "
                    f"steps2feas={s['steps_to_feasible_mean']:.1f} "
                    f"ret={s['return_mean']:.2f}"
                )
                logger.plot(window=5, save_path="results/dqn/0116/metrics_plot.png")

            # ---- reset episode accumulators ----
            obs, info = env.reset()
            mask = info["action_mask"]

            ep_ret = 0.0
            ep_steps = 0
            feasible_steps = 0
            best_mnorm = float("inf")
            best_u_sigma = 10.0
            best_u_delta = 10.0
            steps_to_feasible = None


if __name__ == "__main__":
    train(log_every_episodes=1)
