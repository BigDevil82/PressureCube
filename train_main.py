import os

import numpy as np
import torch

from src.reinforcement.env import StructuralEnv
from src.reinforcement.ppo_agent import PPOAgent
from src.utils.visualize import TrainingVisualizer


def train():
    # --- 1. 配置参数 ---
    env_config = {}  # 可以在此传入默认参数

    # 超参数
    MAX_EPISODES = 1000  # 总训练回合数
    MAX_TIMESTEPS = 20  # 每个回合最大步数 (设计优化通常不需要太长步数)
    UPDATE_TIMESTEP = 10  # 每隔多少步更新一次网络
    LR = 0.0003  # 学习率
    GAMMA = 0.99  # 折扣因子
    K_EPOCHS = 4  # 更新时重复训练次数
    EPS_CLIP = 0.2  # PPO 裁剪参数

    # 维度定义 (基于之前的 StructuralGraph)
    NODE_DIM = 17  # 6(geom) + 4(type) + 3(phys) + 4(resp)
    EDGE_DIM = 3  # 3(coord)
    ACTION_DIM = 8  # 8个设计变量

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # --- 2. 初始化 ---
    env = StructuralEnv(env_config)
    agent = PPOAgent(NODE_DIM, EDGE_DIM, ACTION_DIM, LR, GAMMA, K_EPOCHS, EPS_CLIP, device)

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    visualizer = TrainingVisualizer(log_dir)

    time_step = 0

    # --- 3. 训练循环 ---
    for i_episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        current_ep_reward = 0
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
            done = terminated or truncated

            # --- 收集物理数据 ---
            # info 字典是在 env.step() 中返回的
            if "mass" in info:
                ep_masses.append(info["mass"])
                ep_stress_ratios.append(info["max_stress_ratio"])
                ep_disp_ratios.append(info["max_disp_ratio"])

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
            # 这里我们取本回合的平均值作为记录（也可以取最后一步的值）
            avg_mass = np.mean(ep_masses)
            # 取本回合中出现过的最大风险值（保守估计）
            max_stress = np.max(ep_stress_ratios)
            max_disp = np.max(ep_disp_ratios)

            # 记录并绘图
            visualizer.log_episode(
                episode=i_episode,
                reward=current_ep_reward,
                mass=avg_mass,
                stress_ratio=max_stress,
                disp_ratio=max_disp,
            )

            # 保存图像 (覆盖上一张，保持最新)
            visualizer.save_plot("training_curves.png")

        print(f"Episode {i_episode} finished. Total Reward: {current_ep_reward:.2f}")

        # 保存模型
        if i_episode % 20 == 0:
            agent.save(f"{log_dir}/ppo_{i_episode}.pth")


if __name__ == "__main__":
    train()
