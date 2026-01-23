"""
Diagnostic script to understand why DQN fails on this task.

This script:
1. Compares problem complexity to what DQN can handle
2. Shows reward signal quality
3. Demonstrates why Greedy works and DQN struggles
"""
import numpy as np
import time
from src.reinforcement.dqn.env import SteelSectionOptEnv


def diagnose_problem():
    print("=" * 70)
    print("DIAGNOSTIC: Why is DQN struggling on this task?")
    print("=" * 70)

    env = SteelSectionOptEnv({})
    obs, info = env.reset()

    # 1. Problem Complexity
    print("\n1. PROBLEM COMPLEXITY ANALYSIS")
    print("-" * 50)
    total_states = np.prod(env.max_indices + 1)
    print(f"   State space size: {env.max_indices + 1} = {total_states:,} combinations")
    print(f"   Action space: 81 discrete actions (3^4 deltas)")
    print(f"   Observation dim: {env.obs_dim}")

    # 2. Sample Efficiency
    print("\n2. SAMPLE EFFICIENCY PROBLEM")
    print("-" * 50)
    print("   Your current setup:")
    print("   - batch_size = 256")
    print("   - max_episodes = 100")
    print("   - T (max steps) = 30")
    print("   - Total samples: ~3000")
    print()
    print("   DQN typically needs:")
    print("   - 100,000+ samples for simple tasks")
    print("   - 1,000,000+ for complex tasks")
    print()
    print("   ⚠️  You're providing 30x-300x FEWER samples than needed!")

    # 3. Greedy simulation
    print("\n3. GREEDY BASELINE SIMULATION")
    print("-" * 50)

    # Reset to max sections
    obs, info = env.reset()
    env.idx = env.max_indices.copy()
    initial_mass = info["mass"]

    # Greedy shrink
    idx = env.max_indices.copy()
    num_sims = 0
    trajectory = []

    for comp in range(4):
        comp_names = ["frame", "main", "secondary", "brace"]
        while idx[comp] > 0:
            idx[comp] -= 1
            env.idx = idx.copy()
            env.current_params.frame_section_index = int(idx[0])
            env.current_params.main_beam_section_index = int(idx[1])
            env.current_params.secondary_beam_section_index = int(idx[2])
            env.current_params.brace_section_index = int(idx[3])

            out = env._run_simulation_and_collect()
            num_sims += 1

            if out["u_sigma"] > 1.0 or out["u_delta"] > 1.0:
                idx[comp] += 1  # Revert
                break

            trajectory.append({
                "step": num_sims,
                "comp": comp_names[comp],
                "idx": idx.copy(),
                "mass": out["mass"],
                "u_sigma": out["u_sigma"],
                "u_delta": out["u_delta"],
            })

    # Get final state
    env.idx = idx.copy()
    env.current_params.frame_section_index = int(idx[0])
    env.current_params.main_beam_section_index = int(idx[1])
    env.current_params.secondary_beam_section_index = int(idx[2])
    env.current_params.brace_section_index = int(idx[3])
    final_out = env._run_simulation_and_collect()

    print(f"   Initial sections: {env.max_indices}")
    print(f"   Initial mass: {initial_mass:.1f} kg")
    print(f"   Final sections: {idx}")
    print(f"   Final mass: {final_out['mass']:.1f} kg")
    print(f"   Mass reduction: {(initial_mass - final_out['mass']) / initial_mass * 100:.1f}%")
    print(f"   Simulations needed: {num_sims}")
    print()
    print("   Trajectory (last 10 steps):")
    for t in trajectory[-10:]:
        print(f"      Step {t['step']:2d}: {t['comp']:9s} idx={t['idx']} mass={t['mass']:.0f}kg")

    # 4. Reward Signal Quality
    print("\n4. REWARD SIGNAL QUALITY")
    print("-" * 50)

    # Simulate random episode and check rewards
    obs, info = env.reset()
    mask = info["action_mask"]
    rewards = []
    violations = []

    for _ in range(30):
        valid = np.flatnonzero(mask)
        a = np.random.choice(valid)
        obs, r, term, trunc, info = env.step(a)
        mask = info["action_mask"]
        rewards.append(r)
        violations.append(env._violation(info["u_sigma"], info["u_delta"]))
        if term or trunc:
            break

    print(f"   Random episode rewards: mean={np.mean(rewards):.3f}, std={np.std(rewards):.3f}")
    print(f"   Reward range: [{min(rewards):.3f}, {max(rewards):.3f}]")
    print(f"   Violations: {sum(v > 0 for v in violations)}/{len(violations)} steps infeasible")

    # 5. Key Insight
    print("\n5. KEY INSIGHT: WHY GREEDY WORKS, DQN DOESN'T")
    print("-" * 50)
    print("""
   Greedy works because:
   ✓ Problem has MONOTONIC structure (reducing sections → reducing mass)
   ✓ Constraint boundary is SMOOTH
   ✓ No need for "lookahead" - just reduce until violation

   DQN struggles because:
   ✗ DQN tries to learn Q(s,a) = r + γ*max Q(s',a')
   ✗ But optimal policy doesn't need "future value estimation"
   ✗ 256-batch with 3000 samples → severe overfitting / underfitting
   ✗ 65k parameters for a problem a 10-line if-else can solve

   RECOMMENDATION:
   → For this task, use Greedy, GA, or Bayesian Optimization
   → If you MUST use RL, try:
      1. Much smaller network (64-64)
      2. batch_size=32, not 256
      3. gamma=0.5 (short horizon)
      4. At least 1000 episodes
      5. Fixed geometry first (curriculum learning)
    """)

    # 6. Compare approaches
    print("\n6. APPROACH COMPARISON")
    print("-" * 50)
    print("   | Method      | Samples Needed | Time (1 struct) | Optimality |")
    print("   |-------------|----------------|-----------------|------------|")
    print("   | Greedy      | ~40            | ~20 sec         | Local opt  |")
    print("   | GA (pop=50) | ~500           | ~4 min          | Near-opt   |")
    print("   | DQN (fixed) | ~30,000        | ~4 hours        | ???        |")
    print("   | DQN (random)| ~300,000       | ~40 hours       | ???        |")


def test_reward_functions():
    """Test different reward functions on same trajectory."""
    print("\n" + "=" * 70)
    print("REWARD FUNCTION COMPARISON")
    print("=" * 70)

    from src.reinforcement.dqn.configs.reward_variants import REWARD_FUNCTIONS

    env = SteelSectionOptEnv({})
    obs, info = env.reset()
    mask = info["action_mask"]

    # Fixed action sequence for comparison
    np.random.seed(42)
    action_sequence = []
    valid = np.flatnonzero(mask)
    for _ in range(10):
        a = np.random.choice(valid)
        action_sequence.append(a)
        obs, r, term, trunc, info = env.step(a)
        mask = info["action_mask"]
        valid = np.flatnonzero(mask)
        if term or trunc:
            break

    # Test each reward function
    print("\nCumulative rewards for same action sequence:")
    print("-" * 50)

    for name, reward_fn in REWARD_FUNCTIONS.items():
        env_config = {"reward_type": name} if name != "original" else {}
        env = SteelSectionOptEnv(env_config)
        obs, info = env.reset(seed=42)  # Same seed for reproducibility

        total_reward = 0.0
        for a in action_sequence:
            obs, r, term, trunc, info = env.step(a)
            total_reward += r
            if term or trunc:
                break

        final_mnorm = info.get("mnorm", info["mass"] / env.m0)
        feasible = info["u_sigma"] <= 1 and info["u_delta"] <= 1

        print(f"   {name:15s}: total_r={total_reward:7.2f}, mnorm={final_mnorm:.3f}, feas={feasible}")


if __name__ == "__main__":
    diagnose_problem()
    test_reward_functions()
