import csv
import os
from collections import deque
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class EpisodeStats:
    episode: int
    steps: int
    ep_return: float
    feasible: int
    best_mnorm: float
    final_mnorm: float
    best_u_sigma: float
    best_u_delta: float
    final_u_sigma: float
    final_u_delta: float
    steps_to_feasible: int
    feasible_steps_ratio: float


class MetricsLogger:
    def __init__(self, csv_path: str, window: int = 100):
        self.csv_path = csv_path
        if os.path.exists(csv_path):
            # remove existing file to avoid appending to old logs
            os.remove(csv_path)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self.window = window
        self.recent = deque(maxlen=window)
        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=list(asdict(EpisodeStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).keys())
                )
                writer.writeheader()

    def log_episode(self, st: EpisodeStats):
        self.recent.append(st)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(st).keys()))
            writer.writerow(asdict(st))

    def summary(self):
        if not self.recent:
            return {}
        arr = lambda name: np.array([getattr(x, name) for x in self.recent], dtype=float)
        return {
            "N": len(self.recent),
            "feasible_rate": arr("feasible").mean(),
            "best_mnorm_mean": arr("best_mnorm").mean(),
            "best_mnorm_p10": np.percentile(arr("best_mnorm"), 10),
            "best_mnorm_p90": np.percentile(arr("best_mnorm"), 90),
            "steps_to_feasible_mean": arr("steps_to_feasible").mean(),
            "return_mean": arr("ep_return").mean(),
        }

    def plot(self, window=50, save_path=None):
        df = pd.read_csv(self.csv_path)

        def smooth(x: pd.Series):
            return x.rolling(window, min_periods=1).mean()

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].plot(df["episode"], smooth(df["feasible"]))
        axes[0, 0].set_xlabel("episode")
        axes[0, 0].set_ylabel("feasible (rolling mean)")

        axes[0, 1].plot(df["episode"], smooth(df["best_mnorm"]))
        axes[0, 1].set_xlabel("episode")
        axes[0, 1].set_ylabel("best m/m0 (rolling mean)")

        axes[1, 0].plot(df["episode"], smooth(df["steps_to_feasible"]))
        axes[1, 0].set_xlabel("episode")
        axes[1, 0].set_ylabel("steps_to_feasible (rolling mean)")

        axes[1, 1].plot(df["episode"], df["ep_return"])
        axes[1, 1].plot(df["episode"], smooth(df["ep_return"]))
        axes[1, 1].set_xlabel("episode")
        axes[1, 1].set_ylabel("return (rolling mean)")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()
