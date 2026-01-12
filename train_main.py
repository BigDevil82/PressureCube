import argparse
import os
import random

import numpy as np
import torch

from src.reinforcement.env import StructuralEnv
from src.reinforcement.ppo_agent import PPOAgent
from src.utils.visualize import TrainingVisualizer

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
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

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


def train():

    print(f"Training on {device}")

    # --- 2. 初始化 ---
    env = StructuralEnv(env_config)
    agent = PPOAgent(NODE_DIM, EDGE_DIM, ACTION_DIM, LR, GAMMA, K_EPOCHS, EPS_CLIP, BATCH_SIZE, device)
    # agent.load("results/ppo/1227_step20_multiDiscrete/ppo_300.pth")  # 可选：加载预训练模型

    log_dir = "results/ppo/0105_step20_multiDiscrete_opt"
    os.makedirs(log_dir, exist_ok=True)
    visualizer = TrainingVisualizer(log_dir)

    time_step = 0

    # --- 3. 训练循环 ---
    for i_episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        current_ep_reward = 0
        ep_best_feasible_mass = float("inf")
        # 用于统计本回合的物理指标
        ep_masses = []
        ep_stress_ratios = []
        ep_disp_ratios = []

        for t in range(1, MAX_TIMESTEPS + 1):
            time_step += 1

            # 1. 智能体根据当前图状态选择动作
            # state 是 PyG Data 对象，需要先转到 device
            state = state.to(device)
            action, action_log_prob = agent.policy.get_action(state)

            # 转为 numpy 传给环境
            action_np = action.detach().cpu().numpy().flatten()

            # 2. 环境执行动作
            next_state, reward, terminated, truncated, info = env.step(action_np)
            terminated = terminated or t == MAX_TIMESTEPS
            done = terminated or truncated

            # --- 收集物理数据 ---
            # info 字典是在 env.step() 中返回的
            if "mass" in info:
                ep_masses.append(info["mass"])
                ep_stress_ratios.append(info["max_stress_ratio"])
                ep_disp_ratios.append(info["max_disp_ratio"])

                if max(ep_stress_ratios[-1], ep_disp_ratios[-1]) < 1.0:
                    if info["mass"] < ep_best_feasible_mass:
                        ep_best_feasible_mass = info["mass"]

            # 3. 存储数据
            # 注意：存入 buffer 的 state 最好转回 CPU 以节省显存，或者保持原样取决于内存
            agent.put_data((state.cpu(), action.cpu(), action_log_prob.cpu(), reward, done))

            state = next_state
            current_ep_reward += reward

            # 打印单步信息
            if t % 5 == 0 and "mass" in info:
                print(
                    f"Step {t}: Reward={reward:.4f}, "
                    f"Mass={info['mass']:.0f}, "
                    f"StressR={info['max_stress_ratio']:.2f}, "
                    f"DispR={info['max_disp_ratio']:.2f}"
                )

            # 4. 更新网络
            if time_step % UPDATE_TIMESTEP == 0:
                print(f"--- Updating PPO Agent at step {time_step} ---")
                agent.update()

            if done:
                break

        if ep_masses:
            if ep_best_feasible_mass == float("inf"):
                ep_best_feasible_mass = 20000  # 或者一个明显的高值
            # 取本回合中最后一步的应力和挠度比
            last_stress = ep_stress_ratios[-1]
            last_disp = ep_disp_ratios[-1]

            # 记录并绘图
            visualizer.log_episode(
                episode=i_episode,
                reward=current_ep_reward,
                mass=ep_best_feasible_mass,
                stress_ratio=last_stress,
                disp_ratio=last_disp,
            )

            # 保存图像 (覆盖上一张，保持最新)
            visualizer.save_plot("training_curves.png")

        print(f"Episode {i_episode} finished. Total Reward: {current_ep_reward:.2f}")

        # 保存模型
        if i_episode % 50 == 0:
            agent.save(f"{log_dir}/ppo_{i_episode}.pth")


def evaluate(model_path, num_episodes=20):
    # Setup
    env = StructuralEnv(env_config)
    # Load dimensions from your config
    agent = PPOAgent(NODE_DIM, EDGE_DIM, ACTION_DIM, LR, GAMMA, K_EPOCHS, EPS_CLIP, BATCH_SIZE, device)
    agent.load(model_path)

    success_count = 0
    feasible_masses = []

    print(f"{'Episode':<10} | {'Status':<10} | {'Final Mass':<15} | {'Steps':<10}")
    print("-" * 55)

    for ep in range(num_episodes):
        state, _ = env.reset()
        env.log_state()
        done = False
        min_mass_ep = float("inf")
        is_success_ep = False

        for step in range(20):  # Max steps
            state = state.to(device)
            action, _ = agent.policy.get_action(state, eval_mode=True)
            action_np = action.detach().cpu().numpy().flatten()
            next_state, reward, terminated, _, info = env.step(action_np)
            terminated = terminated or (step == 19)

            ratio = max(info["max_stress_ratio"], info["max_disp_ratio"])
            mass = info["mass"]

            if ratio <= 1.0:
                is_success_ep = True
                min_mass_ep = min(min_mass_ep, mass)

            state = next_state
            if terminated:
                break

        if is_success_ep:
            success_count += 1
            feasible_masses.append(min_mass_ep)
            status = "Success"
        else:
            status = "Failed"
            min_mass_ep = 0.0  # Placeholder

        print(f"{ep:<10} | {status:<10} | {min_mass_ep:<15.1f} | {step:<10}")

    print("-" * 55)
    print(f"Success Rate: {success_count/num_episodes*100:.1f}%")
    if feasible_masses:
        print(f"Average Best Mass: {np.mean(feasible_masses):.1f} kg")
        print(f"Absolute Best Mass: {np.min(feasible_masses):.1f} kg")
    else:
        print("No feasible solutions found.")


if __name__ == "__main__":
    # argparse can be used for more complex argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, choices=["train", "eval"], default="train", help="Mode: train or eval"
    )
    args = parser.parse_args()
    if args.mode == "train":
        train()
    elif args.mode == "eval":
        evaluate("results/ppo/0105_step20_multiDiscrete_opt/ppo_500.pth")  # Adjust filename
