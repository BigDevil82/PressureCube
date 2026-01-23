"""
Sanity check script for DQN Steel Section Optimization environment.
Run this before training to verify the environment is correctly configured.
"""

import numpy as np

from src.reinforcement.dqn.env import SteelSectionOptEnv


def test_action_encoding():
    """Test action encode/decode symmetry."""
    print("=" * 50)
    print("Test 1: Action encoding/decoding")
    print("=" * 50)

    passed = True
    for a in range(81):
        deltas = SteelSectionOptEnv.decode_action(a)
        a_back = SteelSectionOptEnv.encode_action(deltas)
        if a != a_back:
            print(f"  FAIL: action {a} -> {deltas} -> {a_back}")
            passed = False

    # Check specific cases
    test_cases = [
        (0, [-1, -1, -1, -1]),  # All decrease
        (40, [0, 0, 0, 0]),  # No change
        (80, [1, 1, 1, 1]),  # All increase
    ]
    for expected_a, expected_deltas in test_cases:
        deltas = SteelSectionOptEnv.decode_action(expected_a)
        if not np.array_equal(deltas, expected_deltas):
            print(f"  FAIL: decode({expected_a}) = {deltas}, expected {expected_deltas}")
            passed = False

    if passed:
        print("  PASS: All 81 actions encode/decode correctly")
    return passed


def test_initial_state_feasibility():
    """Test that initial state (max sections) is feasible."""
    print("\n" + "=" * 50)
    print("Test 2: Initial state feasibility")
    print("=" * 50)

    env = SteelSectionOptEnv({})
    obs, info = env.reset()

    print(f"  Initial section indices: {env.idx}")
    print(f"  Max indices: {env.max_indices}")
    print(f"  Section counts: {env.sec_counts}")
    print(f"  u_sigma: {info['u_sigma']:.4f}")
    print(f"  u_delta: {info['u_delta']:.4f}")
    print(f"  mass (m0): {info['mass']:.2f} kg")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Observation: {obs}")

    feasible = info["u_sigma"] <= 1.0 and info["u_delta"] <= 1.0
    if feasible:
        print("  PASS: Initial state is FEASIBLE")
    else:
        print("  WARN: Initial state is INFEASIBLE - this may make training harder")
        print("        Consider using larger sections or reducing load")

    return feasible


def test_action_mask():
    """Test action mask at boundaries."""
    print("\n" + "=" * 50)
    print("Test 3: Action mask at boundaries")
    print("=" * 50)

    env = SteelSectionOptEnv({})
    env.reset()

    # At max indices, should not allow +1 actions
    env.idx = env.max_indices.copy()
    mask_at_max = env._get_action_mask()

    # At min indices (0,0,0,0), should not allow -1 actions
    env.idx = np.zeros(4, dtype=np.int64)
    mask_at_min = env._get_action_mask()

    # Check that action 80 (all +1) is blocked at max
    action_all_plus = 80
    if mask_at_max[action_all_plus]:
        print(f"  FAIL: Action {action_all_plus} (all +1) should be masked at max indices")
        return False

    # Check that action 0 (all -1) is blocked at min
    action_all_minus = 0
    if mask_at_min[action_all_minus]:
        print(f"  FAIL: Action {action_all_minus} (all -1) should be masked at min indices")
        return False

    # Check that action 40 (no change) is always allowed
    action_noop = 40
    if not mask_at_max[action_noop] or not mask_at_min[action_noop]:
        print(f"  FAIL: Action {action_noop} (no-op) should always be allowed")
        return False

    print(f"  Valid actions at max indices: {mask_at_max.sum()} / 81")
    print(f"  Valid actions at min indices: {mask_at_min.sum()} / 81")
    print("  PASS: Action mask correctly prevents boundary violations")
    return True


def test_random_episode():
    """Run a random episode to check reward ranges."""
    print("\n" + "=" * 50)
    print("Test 4: Random episode simulation")
    print("=" * 50)

    env = SteelSectionOptEnv({})
    obs, info = env.reset()
    mask = info["action_mask"]

    rewards = []
    u_sigmas = []
    u_deltas = []
    masses = []

    done = False
    step = 0
    while not done and step < 30:
        # Random valid action
        valid_actions = np.flatnonzero(mask)
        action = np.random.choice(valid_actions)

        obs, reward, terminated, truncated, info = env.step(action)
        mask = info["action_mask"]
        done = terminated or truncated

        rewards.append(reward)
        u_sigmas.append(info["u_sigma"])
        u_deltas.append(info["u_delta"])
        masses.append(info["mass"])
        step += 1

    print(f"  Episode length: {step} steps")
    print(f"  Reward range: [{min(rewards):.3f}, {max(rewards):.3f}]")
    print(f"  Reward mean: {np.mean(rewards):.3f}")
    print(f"  u_sigma range: [{min(u_sigmas):.3f}, {max(u_sigmas):.3f}]")
    print(f"  u_delta range: [{min(u_deltas):.3f}, {max(u_deltas):.3f}]")
    print(f"  mass range: [{min(masses):.1f}, {max(masses):.1f}] kg")

    # Check for reasonable reward magnitudes
    if max(abs(r) for r in rewards) > 100:
        print("  WARN: Rewards seem very large, consider scaling")
    else:
        print("  PASS: Reward magnitudes appear reasonable")

    return True


def test_step_timing():
    """Measure time per step."""
    import time

    print("\n" + "=" * 50)
    print("Test 5: Step timing")
    print("=" * 50)

    env = SteelSectionOptEnv({})

    # Time reset
    start = time.time()
    obs, info = env.reset()
    reset_time = time.time() - start
    print(f"  Reset time: {reset_time:.2f} s")

    # Time a few steps
    mask = info["action_mask"]
    step_times = []
    for _ in range(3):
        valid_actions = np.flatnonzero(mask)
        action = np.random.choice(valid_actions)
        start = time.time()
        obs, reward, terminated, truncated, info = env.step(action)
        step_times.append(time.time() - start)
        mask = info["action_mask"]
        if terminated or truncated:
            break

    avg_step_time = np.mean(step_times)
    print(f"  Average step time: {avg_step_time:.2f} s")
    print(f"  Estimated time for 100 episodes (30 steps each): {100 * 30 * avg_step_time / 3600:.1f} hours")

    return True


def main():
    print("\n" + "#" * 60)
    print("  DQN Environment Sanity Check")
    print("#" * 60)

    results = []

    results.append(("Action encoding", test_action_encoding()))
    results.append(("Initial feasibility", test_initial_state_feasibility()))
    results.append(("Action mask", test_action_mask()))
    results.append(("Random episode", test_random_episode()))
    results.append(("Step timing", test_step_timing()))

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL/WARN"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll checks passed! Ready to train.")
    else:
        print("\nSome checks failed or have warnings. Review above for details.")

    return all_passed


if __name__ == "__main__":
    main()
