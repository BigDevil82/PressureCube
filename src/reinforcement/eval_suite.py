import csv
import multiprocessing as mp
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from src.reinforcement.env import StructuralEnv
from src.reinforcement.ppo_agent import PPOAgent

# --- 1. 配置参数 ---
env_config = {
    "reset_section_init": "max",  # 强烈建议先 max
    "target_ratio": 0.95,
    "w_mass": 1.0,
    "w_delta": 0.3,
    "w_slack": 2.0,
    "w_viol": 50.0,
    "feasible_bonus": 0.5,
    "infeasible_base": -1.0,
    "clip_reward": 5.0,
}

# 超参数
MAX_EPISODES = 500  # 总训练回合数
MAX_TIMESTEPS = 20  # 每个回合最大步数 (设计优化通常不需要太长步数)
UPDATE_TIMESTEP = 60  # 每隔多少步更新一次网络
LR = 0.0003  # 学习率
GAMMA = 0.99  # 折扣因子
K_EPOCHS = 5  # 更新时重复训练次数
EPS_CLIP = 0.2  # PPO 裁剪参数
BATCH_SIZE = 64  # 批量大小

# 维度定义 (基于之前的 StructuralGraph)
NODE_DIM = 19  # 6(geom) + 4(type) + 3(phys) + 4(resp)
EDGE_DIM = 3  # 3(coord)
ACTION_DIM = 4  # 8个设计变量

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---- globals for multiprocessing workers ----
_G_ENV = None
_G_AGENT = None
_G_DEVICE = None
_G_MAX_STEPS = None
_G_RUN_BASELINE = True
_G_BASE_CACHE = None


def compute_mass_norm(env: StructuralEnv, mass_kg: float) -> float:
    min_mass = env.min_volume * 7850.0
    max_mass = env.max_volume * 7850.0
    return float((mass_kg - min_mass) / (max_mass - min_mass + 1e-6))


def is_feasible(info) -> bool:
    r = max(float(info["max_stress_ratio"]), float(info["max_disp_ratio"]))
    return r <= 1.0


def max_ratio(info) -> float:
    return max(float(info["max_stress_ratio"]), float(info["max_disp_ratio"]))


def rollout_policy_best_feasible(
    env: StructuralEnv, agent: PPOAgent, seed: int, max_steps: int, device, eval_mode=True
):
    max_idx = np.array([c - 1 for c in env.sec_counts], dtype=np.int64)

    state, _ = env.reset(seed=seed, options={"init_sections": max_idx})
    best = None

    for t in range(1, max_steps + 1):
        state = state.to(device)
        action, _ = agent.policy.get_action(state, eval_mode=eval_mode)
        action_np = action.detach().cpu().numpy().flatten()

        next_state, _, _, _, info = env.step(action_np)
        if is_feasible(info):
            if (best is None) or (float(info["mass"]) < float(best["mass"])):
                best = dict(info)  # 保存最优可行信息
        state = next_state

    return best


def greedy_shrink_baseline(env: StructuralEnv, seed: int):
    max_action = np.array([c - 1 for c in env.sec_counts], dtype=np.int64)

    # 同一结构同一压力
    _, info0 = env.reset(seed=seed, options={"init_sections": max_action})
    if not is_feasible(info0):
        # 极少数情况下最大截面都不可行（一般不会），直接返回 None
        return None

    best_action = max_action.copy()
    best_info = dict(info0)
    improved = True

    while improved:
        improved = False
        # 尝试单维减 1（你也可以改成减多步）
        for d in range(4):
            if best_action[d] <= 0:
                continue
            cand = best_action.copy()
            cand[d] -= 2  # 减少2个截面等级

            _, _, _, _, cand_info = env.step(cand)
            if is_feasible(cand_info) and float(cand_info["mass"]) < float(best_info["mass"]) - 1e-6:
                best_action = cand
                best_info = dict(cand_info)
                improved = True

    return best_info


def _init_worker(
    model_path: str,
    env_config_local: dict,
    max_steps: int,
    device_str: str,
    run_baseline: bool,
    baseline_cache_path: str = None,
):
    """
    每个进程只初始化一次：独立 env + 独立模型，避免每个 seed 重复 load。
    同时切到独立工作目录，避免 OpenSees/后处理写文件时冲突。
    """
    global _G_ENV, _G_AGENT, _G_DEVICE, _G_MAX_STEPS, _G_RUN_BASELINE, _G_BASE_CACHE

    # 避免线程超卖：每个进程只用 1 个 torch 线程（否则 8进程×8线程=爆炸）
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    _G_DEVICE = torch.device(device_str)
    _G_MAX_STEPS = int(max_steps)
    _G_RUN_BASELINE = bool(run_baseline)

    _G_ENV = StructuralEnv(env_config_local)

    _G_AGENT = PPOAgent(NODE_DIM, EDGE_DIM, ACTION_DIM, LR, GAMMA, K_EPOCHS, EPS_CLIP, BATCH_SIZE, _G_DEVICE)
    _G_AGENT.load(model_path)
    _G_AGENT.policy.eval()

    _G_BASE_CACHE = (
        load_baseline_cache(baseline_cache_path) if (baseline_cache_path and run_baseline) else None
    )


def _eval_one_seed(seed: int):
    """
    单个 seed 的评估：返回一行 dict，主进程收集后写 CSV。
    注意：用全局 _G_ENV/_G_AGENT（每进程一个实例），避免频繁初始化。
    """
    global _G_ENV, _G_AGENT, _G_DEVICE, _G_MAX_STEPS, _G_RUN_BASELINE

    env = _G_ENV
    agent = _G_AGENT

    # baseline（可选，最耗时的往往是它）
    base_info = None
    if _G_RUN_BASELINE:
        if _G_BASE_CACHE is None or seed not in _G_BASE_CACHE:
            # 缓存缺失时：你可以选择 None 或者临时补算（补算会慢）
            base_info = None
            raise NotImplementedError(
                "Baseline cache missing for seed {seed}, please precompute the baseline cache."
            )
        else:
            # 构造一个与之前格式兼容的 dict
            b = _G_BASE_CACHE[seed]
            if b["base_feasible"]:
                base_info = {
                    "mass": b["base_mass"],
                    "max_stress_ratio": None,  # 下面不用
                    "max_disp_ratio": None,  # 下面不用
                }
            else:
                base_info = None

    # policy
    pol_info = rollout_policy_best_feasible(env, agent, seed, _G_MAX_STEPS, _G_DEVICE, eval_mode=True)

    pol_ok = pol_info is not None
    base_ok = base_info is not None

    if pol_ok:
        pol_mass = float(pol_info["mass"])
        pol_ratio = max_ratio(pol_info)
        pol_mass_norm = compute_mass_norm(env, pol_mass)
    else:
        pol_mass = pol_ratio = pol_mass_norm = None

    if base_ok:
        # baseline from cache
        base_ok = False
        base_mass = base_mass_norm = base_ratio = None
        if _G_RUN_BASELINE and (_G_BASE_CACHE is not None) and (seed in _G_BASE_CACHE):
            b = _G_BASE_CACHE[seed]
            base_ok = bool(b["base_feasible"])
            base_mass = b["base_mass"]
            base_mass_norm = b["base_mass_norm"]
            base_ratio = b["base_max_ratio"]
    else:
        base_mass = base_ratio = base_mass_norm = None

    regret = None
    if pol_ok and base_ok:
        regret = (pol_mass - base_mass) / (base_mass + 1e-6)

    return {
        "seed": int(seed),
        "policy_feasible": pol_ok,
        "policy_mass": pol_mass,
        "policy_mass_norm": pol_mass_norm,
        "policy_max_ratio": pol_ratio,
        "base_feasible": base_ok,
        "base_mass": base_mass,
        "base_mass_norm": base_mass_norm,
        "base_max_ratio": base_ratio,
        "regret_vs_greedy": regret,
    }


def _init_worker_baseline(env_config_local: dict):
    global _G_ENV
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    _G_ENV = StructuralEnv(env_config_local)


def _baseline_one_seed(seed: int):
    global _G_ENV
    info = greedy_shrink_baseline(_G_ENV, seed)
    ok = info is not None
    if not ok:
        return {
            "seed": int(seed),
            "base_feasible": False,
            "base_mass": None,
            "base_mass_norm": None,
            "base_max_ratio": None,
        }
    base_mass = float(info["mass"])
    return {
        "seed": int(seed),
        "base_feasible": True,
        "base_mass": base_mass,
        "base_mass_norm": compute_mass_norm(_G_ENV, base_mass),
        "base_max_ratio": max_ratio(info),
    }


def build_baseline_cache(
    seeds,
    out_csv="baseline_cache.csv",
    num_workers=None,
):
    import csv
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 2) - 1)

    rows = []
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker_baseline,
        initargs=(env_config,),
    ) as ex:
        futs = {ex.submit(_baseline_one_seed, s): s for s in seeds}
        for fut in as_completed(futs):
            rows.append(fut.result())

    rows.sort(key=lambda r: r["seed"])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for r in rows if r["base_feasible"])
    print(f"[baseline cache] saved: {out_csv}, seeds={len(rows)}, feasible={ok}/{len(rows)}")


def load_baseline_cache(csv_path: str):
    import csv

    cache = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seed = int(row["seed"])

            # 处理 None
            def to_float(x):
                return None if (x is None or x == "" or x.lower() == "none") else float(x)

            cache[seed] = {
                "base_feasible": (row["base_feasible"].lower() == "true"),
                "base_mass": to_float(row["base_mass"]),
                "base_mass_norm": to_float(row["base_mass_norm"]),
                "base_max_ratio": to_float(row["base_max_ratio"]),
            }
    return cache


def evaluate_suite(model_path: str, test_seeds, max_steps=20, out_csv="eval_suite.csv"):
    env = StructuralEnv(env_config)
    agent = PPOAgent(NODE_DIM, EDGE_DIM, ACTION_DIM, LR, GAMMA, K_EPOCHS, EPS_CLIP, BATCH_SIZE, device)
    agent.load(model_path)

    rows = []
    for s in test_seeds:
        # baseline（同结构）
        base_info = greedy_shrink_baseline(env, s)

        # policy（同结构）
        pol_info = rollout_policy_best_feasible(env, agent, s, max_steps, device, eval_mode=True)

        # 统计
        pol_ok = pol_info is not None
        base_ok = base_info is not None

        if pol_ok:
            pol_mass = float(pol_info["mass"])
            pol_ratio = max_ratio(pol_info)
            pol_mass_norm = compute_mass_norm(env, pol_mass)
        else:
            pol_mass = None
            pol_ratio = None
            pol_mass_norm = None

        if base_ok:
            base_mass = float(base_info["mass"])
            base_ratio = max_ratio(base_info)
            base_mass_norm = compute_mass_norm(env, base_mass)
        else:
            base_mass = None
            base_ratio = None
            base_mass_norm = None

        regret = None
        if pol_ok and base_ok:
            regret = (pol_mass - base_mass) / (base_mass + 1e-6)

        rows.append(
            {
                "seed": s,
                "policy_feasible": pol_ok,
                "policy_mass": pol_mass,
                "policy_mass_norm": pol_mass_norm,
                "policy_max_ratio": pol_ratio,
                "base_feasible": base_ok,
                "base_mass": base_mass,
                "base_mass_norm": base_mass_norm,
                "base_max_ratio": base_ratio,
                "regret_vs_greedy": regret,
            }
        )

    # 写 CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 汇总
    pol_feasible = [r for r in rows if r["policy_feasible"]]
    feasible_rate = len(pol_feasible) / len(rows)

    def safe_mean(vals):
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    mean_mass_norm = safe_mean([r["policy_mass_norm"] for r in pol_feasible])
    mean_util = safe_mean([r["policy_max_ratio"] for r in pol_feasible])

    regrets = [r["regret_vs_greedy"] for r in rows if r["regret_vs_greedy"] is not None]
    mean_regret = safe_mean(regrets)
    p90_regret = float(np.percentile(regrets, 90)) if regrets else None

    summary = "===== EVAL SUITE SUMMARY =====\n" + "\n".join(
        [
            (f"Test cases: {len(rows)}"),
            (f"Feasible rate: {feasible_rate*100:.1f}%"),
            (f"Mean mass_norm (feasible): {mean_mass_norm}"),
            (f"Mean utilization max_ratio (feasible): {mean_util}"),
            (f"Mean regret vs greedy (where both feasible): {mean_regret}"),
            (f"P90  regret vs greedy (where both feasible): {p90_regret}"),
            (f"Saved CSV: {out_csv}"),
        ]
    )

    print(summary)
    # write to file
    summary_file = out_csv.replace(".csv", "_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary)


def evaluate_suite_parallel(
    model_path: str,
    test_seeds,
    max_steps=20,
    out_csv="eval_suite.csv",
    num_workers=None,
    device_str="cpu",
    run_baseline=True,
    baseline_cache_path=None,
):
    """
    多进程并行评估：按 seed 分发任务。建议 device_str="cpu"，
    因为多进程共享单卡 GPU 反而容易更慢/更麻烦。
    """
    if num_workers is None:
        # 留一个核给系统/IO
        num_workers = max(1, (os.cpu_count() or 2) - 1)

    # Windows 下必须放在 __main__ 保护里调用；这里不强制 set_start_method
    # 但如果你遇到奇怪的 fork 问题，可以改成 spawn
    # mp.set_start_method("spawn", force=True)

    rows = []
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(model_path, env_config, max_steps, device_str, run_baseline, baseline_cache_path),
    ) as ex:
        futures = {ex.submit(_eval_one_seed, s): s for s in test_seeds}
        for fut in as_completed(futures):
            rows.append(fut.result())

    # 排序保证输出稳定
    rows.sort(key=lambda r: r["seed"])

    # 写 CSV（复用你原逻辑）
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 汇总（复用你原逻辑）
    pol_feasible = [r for r in rows if r["policy_feasible"]]
    feasible_rate = len(pol_feasible) / len(rows)

    def safe_mean(vals):
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    mean_mass_norm = safe_mean([r["policy_mass_norm"] for r in pol_feasible])
    mean_util = safe_mean([r["policy_max_ratio"] for r in pol_feasible])

    regrets = [r["regret_vs_greedy"] for r in rows if r["regret_vs_greedy"] is not None]
    mean_regret = safe_mean(regrets)
    p90_regret = float(np.percentile(regrets, 90)) if regrets else None

    summary = "===== EVAL SUITE SUMMARY (PARALLEL) =====\n" + "\n".join(
        [
            (f"Workers: {num_workers}"),
            (f"Device: {device_str}"),
            (f"Baseline: {run_baseline}"),
            (f"Test cases: {len(rows)}"),
            (f"Feasible rate: {feasible_rate*100:.1f}%"),
            (f"Mean mass_norm (feasible): {mean_mass_norm}"),
            (f"Mean utilization max_ratio (feasible): {mean_util}"),
            (f"Mean regret vs greedy (where both feasible): {mean_regret}"),
            (f"P90  regret vs greedy (where both feasible): {p90_regret}"),
            (f"Saved CSV: {out_csv}"),
        ]
    )

    print(summary)
    summary_file = out_csv.replace(".csv", "_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary)


if __name__ == "__main__":
    TEST_SEEDS = list(range(10000, 10100))  # 100 个固定结构
    # evaluate_suite(
    #     model_path="results/ppo/0105_step20_multiDiscrete_opt/ppo_500.pth",
    #     test_seeds=TEST_SEEDS,
    #     max_steps=MAX_TIMESTEPS,
    #     out_csv="results/ppo/0105_step20_multiDiscrete_opt/eval_suite_ppo_500.csv",
    # )

    # 1) 只跑一次：生成 baseline 缓存
    build_baseline_cache(TEST_SEEDS, out_csv="results/ppo/baseline_cache_100.csv", num_workers=8)

    # # 2) 多次复用：评估不同模型时直接读缓存
    # evaluate_suite_parallel(
    #     model_path=".../ppo_500.pth",
    #     test_seeds=TEST_SEEDS,
    #     max_steps=MAX_TIMESTEPS,
    #     out_csv=".../eval_ppo_500.csv",
    #     num_workers=8,
    #     device_str="cpu",
    #     run_baseline=True,
    #     baseline_cache_path="baseline_cache_100.csv",
    # )
