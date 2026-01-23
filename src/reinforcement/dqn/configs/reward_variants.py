"""
Reward function variants for DQN steel section optimization experiments.

Each reward function takes environment state and returns a scalar reward.
"""

import numpy as np


def reward_original(env_state: dict) -> float:
    """
    Original reward function (baseline for comparison).

    Two-stage incremental reward:
    - Infeasible: prioritize constraint satisfaction (alpha=20)
    - Feasible: constant reward + mass reduction (beta=1)

    Problems:
    - Scale mismatch (alpha >> beta)
    - Sparse mass signal in feasible region
    - No absolute mass gradient
    """
    # Unpack state
    v = env_state["violation"]
    prev_v = env_state["prev_violation"]
    mnorm = env_state["mnorm"]
    prev_mnorm = env_state["prev_mnorm"]
    feasible_prev = env_state["feasible_prev"]
    feasible_now = env_state["feasible_now"]
    deltas = env_state["deltas"]

    # Hyperparameters (original)
    alpha = 20.0
    beta = 1.0
    c_feas = 1.0
    penalty_feas_to_infeas = 2.0
    penalty_noop = 0.05

    # Two-stage reward
    if v > 0.0:  # Infeasible
        r = alpha * (prev_v - v) - beta * (mnorm - prev_mnorm)
    else:  # Feasible
        r = c_feas - beta * (mnorm - prev_mnorm)

    # Penalties
    if feasible_prev and (not feasible_now):
        r -= penalty_feas_to_infeas

    if np.all(deltas == 0):
        r -= penalty_noop

    return float(r)


def reward_balanced(env_state: dict) -> float:
    """
    Variant A: Balanced two-stage reward.

    Key changes from original:
    1. Reduced alpha from 20 to 5 (better balance with mass term)
    2. In feasible region: use absolute mass instead of constant
       - r = 2.0 - 5.0 * mnorm (lower mass → higher reward)
    3. Keep penalties for safety

    Expected behavior:
    - Still prioritizes feasibility, but not excessively
    - Stronger pull toward low-mass regions in feasible zone
    - Smooth gradient throughout state space
    """
    # Unpack state
    v = env_state["violation"]
    prev_v = env_state["prev_violation"]
    mnorm = env_state["mnorm"]
    prev_mnorm = env_state["prev_mnorm"]
    feasible_prev = env_state["feasible_prev"]
    feasible_now = env_state["feasible_now"]
    deltas = env_state["deltas"]

    # Hyperparameters (balanced)
    alpha = 5.0  # Reduced from 20
    beta = 1.0
    penalty_feas_to_infeas = 2.0
    penalty_noop = 0.05

    # Two-stage reward with better balance
    if v > 0.0:  # Infeasible
        # Constraint improvement + mass improvement
        r_constraint = alpha * (prev_v - v)
        r_mass = -beta * (mnorm - prev_mnorm)
        r = r_constraint + r_mass
    else:  # Feasible - absolute mass reward
        # Lower mass → higher reward (potential-based)
        r = 2.0 - 5.0 * mnorm

    # Penalties
    if feasible_prev and (not feasible_now):
        r -= penalty_feas_to_infeas

    if np.all(deltas == 0):
        r -= penalty_noop

    return float(r)


def reward_balanced_v2(env_state: dict) -> float:
    """
    Variant A-v2: Fixed Balanced reward (addressing negative reward issue).

    CRITICAL FIX:
    - Original balanced had r = 2.0 - 5.0*mnorm in feasible region
    - This gives NEGATIVE reward when mnorm > 0.4 (which is always true!)
    - Agent learned to prefer infeasible states (positive reward) over feasible (negative)

    New design:
    1. Infeasible region: Large negative penalty (-10 * violation)
    2. Feasible region: Positive reward based on mass reduction (1.0 - mnorm)
       - mnorm=0.6 → r=0.4 (good)
       - mnorm=0.9 → r=0.1 (okay)
       - Always positive, incentivizes staying feasible

    Expected behavior:
    - Agent strongly prefers feasible region (positive vs negative reward)
    - Within feasible region, lower mass → higher reward
    - Clear gradient toward optimal (low mass + feasible)
    """
    # Unpack state
    v = env_state["violation"]
    prev_v = env_state["prev_violation"]
    mnorm = env_state["mnorm"]
    prev_mnorm = env_state["prev_mnorm"]
    feasible_prev = env_state["feasible_prev"]
    feasible_now = env_state["feasible_now"]
    deltas = env_state["deltas"]

    # Hyperparameters
    beta = 1.0
    penalty_feas_to_infeas = 5.0  # Increased from 2.0
    penalty_noop = 0.1

    # Two-stage reward with CORRECTED feasible region
    if v > 0.0:  # Infeasible - strong negative penalty
        r = -10.0 * v - beta * (mnorm - prev_mnorm)
    else:  # Feasible - positive reward, lower mass better
        r = 1.0 * (1.0 - mnorm)

    # Penalties
    if feasible_prev and (not feasible_now):
        r -= penalty_feas_to_infeas

    if np.all(deltas == 0):
        r -= penalty_noop

    return float(r)


def reward_mass_centric(env_state: dict) -> float:
    """
    Variant B: Mass-centric with constraint penalty.

    Philosophy: Treat mass reduction as primary goal, constraints as penalty.

    Key changes:
    1. Strong mass improvement signal (scale=10)
    2. Constraint violation as penalty (not primary reward)
    3. No special case for feasible/infeasible

    Expected behavior:
    - Agent focuses on mass reduction
    - Avoids violations due to penalty
    - May converge faster if violations are rare

    Risk: May violate constraints more often during training
    """
    # Unpack state
    v = env_state["violation"]
    mnorm = env_state["mnorm"]
    prev_mnorm = env_state["prev_mnorm"]
    u_sigma = env_state["u_sigma"]
    u_delta = env_state["u_delta"]
    feasible_prev = env_state["feasible_prev"]
    feasible_now = env_state["feasible_now"]
    deltas = env_state["deltas"]

    # Hyperparameters
    mass_scale = 10.0  # Strong mass signal
    violation_penalty = 5.0
    penalty_feas_to_infeas = 2.0
    penalty_noop = 0.05

    # Mass improvement (negative delta = reduction = positive reward)
    r_mass = -mass_scale * (mnorm - prev_mnorm)

    # Constraint penalty
    if u_sigma > 1.0 or u_delta > 1.0:
        r_violation = -violation_penalty * (max(u_sigma - 1.0, 0) + max(u_delta - 1.0, 0))
    else:
        r_violation = 0.0

    r = r_mass + r_violation

    # Additional penalties
    if feasible_prev and (not feasible_now):
        r -= penalty_feas_to_infeas

    if np.all(deltas == 0):
        r -= penalty_noop

    return float(r)


def reward_potential(env_state: dict) -> float:
    """
    Variant C: Potential-based reward shaping.

    Based on Ng et al. 1999: "Policy Invariance Under Reward Shaping"
    Reward = Φ(s') - Φ(s), where Φ is a potential function

    Potential: Φ(s) = -mass - λ * violation

    Properties:
    - Guaranteed to preserve optimal policy
    - Smooth gradient toward low-mass + feasible region
    - Theoretically sound

    Expected behavior:
    - Consistent guidance throughout state space
    - No special cases or heuristics
    - Should be most stable
    """
    # Unpack state
    v = env_state["violation"]
    prev_v = env_state["prev_violation"]
    mnorm = env_state["mnorm"]
    prev_mnorm = env_state["prev_mnorm"]
    feasible_prev = env_state["feasible_prev"]
    feasible_now = env_state["feasible_now"]
    deltas = env_state["deltas"]

    # Hyperparameters
    lambda_constraint = 10.0  # Weight for constraint violation
    penalty_feas_to_infeas = 2.0
    penalty_noop = 0.05

    def potential(mass_norm: float, violation: float) -> float:
        """Potential function: lower mass + lower violation = higher potential"""
        return -mass_norm - lambda_constraint * violation

    # Reward = change in potential
    phi_new = potential(mnorm, v)
    phi_old = potential(prev_mnorm, prev_v)
    r = phi_new - phi_old

    # Additional penalties (optional, can be removed for pure potential-based)
    if feasible_prev and (not feasible_now):
        r -= penalty_feas_to_infeas

    if np.all(deltas == 0):
        r -= penalty_noop

    return float(r)


def reward_potential_v2(env_state: dict) -> float:
    """
    Variant C-v2: Enhanced potential-based reward (fixing balanced_v2 issues).

    DIAGNOSIS OF BALANCED_V2 FAILURE:
    - balanced_v2 gave r = 1.0 - mnorm in feasible region
    - mnorm=0.9 → r=0.1, mnorm=0.7 → r=0.3 (difference too small!)
    - Agent found "lazy strategy": stay at mnorm~0.9, get 30×0.1=+3.0 total reward
    - True optimization (mnorm=0.7) requires exploration with negative rewards, not worth it

    NEW DESIGN:
    - Use potential-based shaping with STRONG constraint penalty (λ=20)
    - Ensures large reward difference between mnorm=0.9 and mnorm=0.7
    - No special cases, pure gradient-based learning

    Potential function: Φ(s) = -mnorm - λ * max(violation, 0)
    Reward = Φ(s') - Φ(s)

    Example rewards:
    - mnorm: 0.9→0.7, v=0→0: r = (-0.7) - (-0.9) = +0.2 ✅ Good!
    - mnorm: 0.9→0.9, v=0→0: r = 0 ✅ No reward for no-op
    - mnorm: 0.9→0.85, v=0→0.1: r = (-0.85-20×0.1) - (-0.9) = -1.95 ❌ Strong penalty

    Properties:
    - Guaranteed optimal policy (Ng et al. 1999)
    - No "lazy strategy" exploitation
    - Clear gradient: lower mass in feasible region → higher cumulative reward
    """
    # Unpack state
    v = env_state["violation"]
    prev_v = env_state["prev_violation"]
    mnorm = env_state["mnorm"]
    prev_mnorm = env_state["prev_mnorm"]
    feasible_prev = env_state["feasible_prev"]
    feasible_now = env_state["feasible_now"]
    deltas = env_state["deltas"]

    # Hyperparameters
    lambda_constraint = 5.0  # CRITICAL: High penalty for violations (same as original alpha)
    penalty_feas_to_infeas = 5.0  # Strong penalty for losing feasibility
    penalty_noop = 0.1

    def potential(mass_norm: float, violation: float) -> float:
        """
        Potential function: Φ(s) = -mnorm - λ * max(v, 0)

        Lower mass + no violation → higher potential → better state
        """
        return -mass_norm - lambda_constraint * max(violation, 0)

    # Reward = change in potential
    phi_now = potential(mnorm, v)
    phi_prev = potential(prev_mnorm, prev_v)
    r = phi_now - phi_prev

    # Additional penalties for safety
    if feasible_prev and (not feasible_now):
        r -= penalty_feas_to_infeas

    if np.all(deltas == 0):
        r -= penalty_noop

    return float(r)


# Registry for easy access
REWARD_FUNCTIONS = {
    "original": reward_original,
    "balanced": reward_balanced,
    "balanced_v2": reward_balanced_v2,  # Fixed version with positive feasible rewards
    "mass_centric": reward_mass_centric,
    "potential": reward_potential,
    "potential_v2": reward_potential_v2,  # Enhanced potential fixing lazy strategy issue
}


def get_reward_function(name: str):
    """
    Get reward function by name.

    Args:
        name: One of "original", "balanced", "mass_centric", "potential"

    Returns:
        Reward function

    Raises:
        ValueError: If name is not recognized
    """
    if name not in REWARD_FUNCTIONS:
        raise ValueError(
            f"Unknown reward function: {name}. " f"Available options: {list(REWARD_FUNCTIONS.keys())}"
        )
    return REWARD_FUNCTIONS[name]
