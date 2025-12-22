import os

import matplotlib.pyplot as plt
import numpy as np


class TrainingVisualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 数据容器
        self.episodes = []
        self.rewards = []
        self.masses = []
        self.stress_ratios = []
        self.disp_ratios = []

    def log_episode(self, episode, reward, mass, stress_ratio, disp_ratio):
        """记录单回合数据"""
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.masses.append(mass)
        self.stress_ratios.append(stress_ratio)
        self.disp_ratios.append(disp_ratio)

    def save_plot(self, filename="training_progress.png"):
        """绘制并保存 2x2 的监控面板"""
        if len(self.episodes) < 2:
            return

        plt.figure(figsize=(12, 10))
        # plt.style.use("ggplot")  # 使用更好看的样式

        # 1. Cumulative Reward
        plt.subplot(2, 2, 1)
        plt.plot(self.episodes, self.rewards, label="Raw", alpha=1, color="blue")
        # plt.plot(
        #     self.episodes, self._moving_average(self.rewards), label="Avg (10)", color="blue", linewidth=2
        # )
        plt.title("Episode Cumulative Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()

        # 2. Structure Mass
        plt.subplot(2, 2, 2)
        plt.plot(self.episodes, self.masses, label="Raw", alpha=1, color="green")
        # plt.plot(
        #     self.episodes, self._moving_average(self.masses), label="Avg (10)", color="green", linewidth=2
        # )
        plt.title("Average Structure Mass (kg)")
        plt.xlabel("Episode")
        plt.ylabel("Mass (kg)")
        plt.legend()

        # 3. Max Stress Ratio
        plt.subplot(2, 2, 3)
        plt.plot(self.episodes, self.stress_ratios, color="red", alpha=0.6)
        plt.axhline(y=1.0, color="black", linestyle="--", label="Limit (1.0)")
        plt.title("Max Stress Ratio")
        plt.xlabel("Episode")
        plt.ylabel("Ratio (>1 means fail)")
        plt.legend()

        # 4. Max Displacement Ratio
        plt.subplot(2, 2, 4)
        plt.plot(self.episodes, self.disp_ratios, color="orange", alpha=0.6)
        plt.axhline(y=1.0, color="black", linestyle="--", label="Limit (1.0)")
        plt.title("Max Displacement Ratio")
        plt.xlabel("Episode")
        plt.ylabel("Ratio (>1 means fail)")
        plt.legend()

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path)
        plt.close()  # 关闭图表释放内存

    def _moving_average(self, data, window_size=5):
        """计算移动平均线以平滑曲线"""
        if len(data) < window_size:
            window_size = len(data)
        return np.convolve(data, np.ones(window_size) / window_size, mode="valid")
