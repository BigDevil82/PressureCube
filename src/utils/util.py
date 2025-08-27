import os.path
import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def is_close(a, b, abs_tol=1e-6):
    return abs(a - b) < abs_tol


class Timer:
    """
    Timer context manager to measure the time elapsed in a block of code

    Example:
    ```
    with Timer("Execution time:"):
        # block of code
    ```

    Args:
        - prefix: str, prefix to print before the elapsed time
        - postfix: str, postfix to print after the elapsed time
        - print_func: callable, print function to use, default is `print`

    """

    def __init__(self, prefix="", postfix="", print_func=print) -> None:
        self.prefix = prefix
        self.postfix = postfix
        self.print_func = print_func

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, *args):
        elapsed_time = time.time() - self.start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        time_str = f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"
        self.print_func(f"{self.prefix} {time_str} {self.postfix}")


def calculate_rectangle_iou(
    rect1: Tuple[Tuple[float, float], Tuple[float, float]],
    rect2: Tuple[Tuple[float, float], Tuple[float, float]],
) -> float:
    """
    计算两个矩形的IoU (Intersection over Union)

    Args:
        rect1: 第一个矩形的坐标，格式为((x1, y1), (x2, y2))，其中(x1, y1)是左下角坐标，(x2, y2)是右上角坐标
        rect2: 第二个矩形的坐标，格式同上

    Returns:
        float: IoU值，范围[0, 1]
    """
    # 解包矩形坐标
    (x1_1, y1_1), (x2_1, y2_1) = rect1
    (x1_2, y1_2), (x2_2, y2_2) = rect2

    # 计算交集矩形的坐标
    x_left = max(x1_1, x1_2)
    y_bottom = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_top = min(y2_1, y2_2)

    # 如果没有交集，返回0
    if x_right < x_left or y_top < y_bottom:
        return 0.0

    # 计算交集面积
    intersection_area = (x_right - x_left) * (y_top - y_bottom)

    # 计算两个矩形的面积
    rect1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    rect2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    # 计算并集面积
    union_area = rect1_area + rect2_area - intersection_area

    # 计算IoU
    iou = intersection_area / union_area

    return iou


def plot_progress(
    metrics_history,
    save_dir,
    filename,
    hline=355,
    x_axis_label="Episode",
    main_title_prefix="Training Progress",
):
    """Plots and saves the progress for various metrics."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Create x-axis values (episodes or steps)
    x_values = list(range(1, len(metrics_history["rewards"]) + 1))
    if not x_values:
        print("Warning: No metrics to plot.")
        return

    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle(main_title_prefix, fontsize=16)

    # Plot Rewards
    axs[0, 0].plot(
        x_values,
        metrics_history["rewards"],
        label=f"{x_axis_label} Reward",
        color="blue",
        marker="o",
        linestyle="-",
        markersize=2,
    )
    axs[0, 0].set_ylabel("Reward")
    axs[0, 0].set_xlabel(x_axis_label)
    axs[0, 0].set_title(f"Rewards per {x_axis_label}")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot Max Displacement
    axs[0, 1].plot(
        x_values,
        metrics_history["max_displacements"],
        label="Max Displacement",
        color="red",
        marker="o",
        linestyle="-",
        markersize=2,
    )
    axs[0, 1].set_ylabel("Max Displacement (mm)")
    axs[0, 1].set_xlabel(x_axis_label)
    axs[0, 1].set_title(f"Maximum Displacement per {x_axis_label}")
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    axs[0, 1].axhline(y=14.4, color="r", linestyle="--", label="Constraint")

    # Plot Max Stress
    axs[1, 0].plot(
        x_values,
        metrics_history["max_stresses"],
        label="Max Stress",
        color="green",
        marker="o",
        linestyle="-",
        markersize=2,
    )
    axs[1, 0].set_ylabel("Max Stress (MPa)")
    axs[1, 0].set_xlabel(x_axis_label)
    axs[1, 0].set_title(f"Maximum Stress per {x_axis_label}")
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    axs[1, 0].axhline(y=hline, color="r", linestyle="--", label="Constraint")

    # Plot Model Volume
    axs[1, 1].plot(
        x_values,
        metrics_history["model_volumes"],
        label="Model Volume",
        color="purple",
        marker="o",
        linestyle="-",
        markersize=2,
    )
    axs[1, 1].plot(
        x_values,
        metrics_history["volume_efficiencies"],
        label="Volume Efficiency",
        color="orange",
        marker="o",
        linestyle="-",
        markersize=2,
    )
    axs[1, 1].set_ylabel("Model Volume (m³)")
    axs[1, 1].set_xlabel(x_axis_label)
    axs[1, 1].set_title(f"Structure Volume per {x_axis_label}")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout
    plot_filename = os.path.join(save_dir, filename)
    plt.savefig(plot_filename)
    plt.close(fig)
