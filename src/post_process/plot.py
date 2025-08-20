import argparse
import os
import platform

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


def setup_matplotlib_fonts():
    """
    设置matplotlib支持中英文混合字体
    中文使用宋体，英文使用Times New Roman
    """
    # 设置中文字体
    if platform.system() == "Windows":
        # Windows系统
        chinese_font = "SimSun"  # 宋体
        english_font = "Times New Roman"
    elif platform.system() == "Darwin":
        # macOS系统
        chinese_font = "STSong"  # 华文宋体
        english_font = "Times New Roman"
    else:
        # Linux系统
        chinese_font = "DejaVu Sans"  # 备用字体
        english_font = "Times New Roman"

    # 配置matplotlib
    plt.rcParams["font.sans-serif"] = [chinese_font, english_font]
    plt.rcParams["font.serif"] = [english_font, chinese_font]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    # 设置默认字体大小和其他参数
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.labelsize"] = 7
    plt.rcParams["xtick.labelsize"] = 6
    plt.rcParams["ytick.labelsize"] = 6
    plt.rcParams["legend.fontsize"] = 6

    # 设置图形尺寸（调整为7cm宽，高度相应增加）
    plt.rcParams["figure.figsize"] = [7.2 / 2.54, 5.0 / 2.54]  # 转换为英寸
    plt.rcParams["figure.dpi"] = 600  # 提高分辨率到600 DPI

    # 设置坐标轴刻度朝内，减小刻度长度和宽度
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.major.size"] = 2.0  # 减小刻度长度
    plt.rcParams["ytick.major.size"] = 2.0
    plt.rcParams["xtick.minor.size"] = 1.0
    plt.rcParams["ytick.minor.size"] = 1.0

    # 关闭网格线
    plt.rcParams["axes.grid"] = False

    # 设置线宽
    plt.rcParams["axes.linewidth"] = 0.2
    plt.rcParams["xtick.major.width"] = 0.2  # 减小刻度线宽度
    plt.rcParams["ytick.major.width"] = 0.2
    plt.rcParams["xtick.minor.width"] = 0.1
    plt.rcParams["ytick.minor.width"] = 0.1

    # 优化边距设置
    plt.rcParams["figure.subplot.left"] = 0.15
    plt.rcParams["figure.subplot.right"] = 0.95
    plt.rcParams["figure.subplot.bottom"] = 0.15
    plt.rcParams["figure.subplot.top"] = 0.95


def apply_moving_average(values, window_size=10):
    """对数据应用滑动平均滤波"""
    values = np.array(values)
    if window_size <= 1 or len(values) < window_size:
        return values

    # 使用卷积计算滑动平均
    smoothed_values = np.convolve(values, np.ones(window_size), "valid") / window_size

    # 填充开始部分以保持长度一致
    padded_smoothed_values = np.pad(smoothed_values, (window_size - 1, 0), "edge")

    return padded_smoothed_values


def save_figure(fig: plt.Figure, save_dir: str, filename_base: str, file_format: str = "svg") -> str:
    """
    保存图形为指定格式

    Args:
        fig: matplotlib图形对象
        save_dir: 保存目录
        filename_base: 文件名基础部分（不含扩展名）
        file_format: 文件格式 ('eps', 'pdf', 'svg', 'png')
    """
    filename = f"{filename_base}.{file_format}"
    save_path = os.path.join(save_dir, filename)

    if file_format in ["eps", "pdf", "svg"]:
        # 矢量图格式，不需要指定DPI
        fig.savefig(save_path, format=file_format, bbox_inches="tight", pad_inches=0.05)
    else:
        # 位图格式，使用高DPI
        fig.savefig(save_path, format=file_format, dpi=600, bbox_inches="tight", pad_inches=0.05)

    return save_path


def plot_reward_curve(
    npz_file_path: str, save_dir: str = None, window_size: int = 10, file_format: str = "svg"
) -> None:
    """
    绘制累积奖励变化曲线（包含原始数据和滑动平均）

    Args:
        npz_file_path: metrics.npz文件路径
        save_dir: 保存目录
        window_size: 滑动平均窗口大小
        file_format: 输出文件格式 ('eps', 'pdf', 'svg', 'png')
    """
    setup_matplotlib_fonts()

    if not os.path.exists(npz_file_path):
        print(f"Error: Metrics file not found at {npz_file_path}")
        return

    if save_dir is None:
        save_dir = os.path.dirname(npz_file_path)
    os.makedirs(save_dir, exist_ok=True)

    # 加载数据
    data = np.load(npz_file_path)

    if "rewards" not in data:
        print("Error: 'rewards' data not found in npz file")
        return

    rewards = data["rewards"]

    if len(rewards) == 0:
        print("Metrics file is empty. Nothing to plot.")
        return

    # 由于数据已经是每个episode的最终结果，直接使用
    episodes = np.arange(1, len(rewards) + 1)

    # 计算滑动平均
    smoothed_rewards = apply_moving_average(rewards, window_size)

    # 创建图形（使用全局设置的figsize）
    fig, ax = plt.subplots()

    # 绘制原始数据（浅色）
    ax.plot(episodes, rewards, color="lightblue", linewidth=0.5, alpha=0.8, label="原始数据")

    # 绘制滑动平均（深色）
    ax.plot(episodes, smoothed_rewards, color="blue", linewidth=1.0, label=f"滑动平均")

    # 设置坐标轴标签
    ax.set_xlabel("Episode")
    ax.set_ylabel("累积奖励")

    # 添加图例
    ax.legend(frameon=False)

    # 保存图形
    save_path = save_figure(fig, save_dir, "reward_curve", file_format)
    plt.close()

    print(f"Reward curve saved to {save_path}")


def plot_displacement_curve(
    npz_file_path: str,
    save_dir: str = None,
    limit_value: float = 14.4,
    file_format: str = "svg",
    window_size: int = 10,
) -> None:
    """
    绘制结构最大位移变化曲线

    Args:
        npz_file_path: metrics.npz文件路径
        save_dir: 保存目录
        limit_value: 位移限值 (mm)
        file_format: 输出文件格式 ('eps', 'pdf', 'svg', 'png')
        window_size: 滑动平均窗口大小
    """
    setup_matplotlib_fonts()

    if not os.path.exists(npz_file_path):
        print(f"Error: Metrics file not found at {npz_file_path}")
        return

    if save_dir is None:
        save_dir = os.path.dirname(npz_file_path)
    os.makedirs(save_dir, exist_ok=True)

    # 加载数据
    data = np.load(npz_file_path)

    if "max_displacements" not in data:
        print("Error: 'max_displacements' data not found in npz file")
        return

    max_displacements = data["max_displacements"]

    if len(max_displacements) == 0:
        print("Metrics file is empty. Nothing to plot.")
        return

    # 由于数据已经是每个episode的最终结果，直接使用
    episodes = np.arange(1, len(max_displacements) + 1)

    # 计算滑动平均
    smoothed_displacements = apply_moving_average(max_displacements, window_size)

    # 创建图形（使用全局设置的figsize）
    fig, ax = plt.subplots()

    # 绘制原始位移数据（浅色）
    ax.plot(episodes, max_displacements, color="lightcoral", linewidth=0.5, alpha=0.8, label="原始数据")

    # 绘制滑动平均（深色）
    ax.plot(episodes, smoothed_displacements, color="red", linewidth=1.0, label=f"滑动平均")

    # 添加限值线
    ax.axhline(y=limit_value, color="black", linestyle="--", linewidth=0.7, alpha=0.8)

    # 添加箭头标注
    # 计算合适的箭头位置（大约在图的中间位置）
    arrow_x = len(episodes) * 0.7  # 箭头x位置在70%处
    arrow_y = limit_value + (max(max_displacements) - limit_value) * 0.3  # 箭头y位置在限值线上方

    # 添加箭头指向限值线
    ax.annotate(
        "位移限值14.4mm",
        xy=(arrow_x * 0.8, limit_value),  # 箭头指向的点
        xytext=(arrow_x, arrow_y),  # 文字位置
        fontsize=7,
        ha="center",
        va="bottom",
        arrowprops=dict(arrowstyle="->", color="black", lw=0.5, shrinkA=0, shrinkB=2),  # 箭头
    )

    # 设置坐标轴标签
    ax.set_xlabel("Episode")
    ax.set_ylabel("最大位移 (mm)")

    # 添加图例
    ax.legend(frameon=False)

    # 保存图形
    save_path = save_figure(fig, save_dir, "displacement_curve", file_format)
    plt.close()

    print(f"Displacement curve saved to {save_path}")


def plot_stress_curve(
    npz_file_path: str,
    save_dir: str = None,
    limit_value: float = 355,
    file_format: str = "svg",
    window_size: int = 10,
) -> None:
    """
    绘制结构最大应力变化曲线

    Args:
        npz_file_path: metrics.npz文件路径
        save_dir: 保存目录
        limit_value: 应力限值 (MPa)
        file_format: 输出文件格式 ('eps', 'pdf', 'svg', 'png')
        window_size: 滑动平均窗口大小
    """
    setup_matplotlib_fonts()

    if not os.path.exists(npz_file_path):
        print(f"Error: Metrics file not found at {npz_file_path}")
        return

    if save_dir is None:
        save_dir = os.path.dirname(npz_file_path)
    os.makedirs(save_dir, exist_ok=True)

    # 加载数据
    data = np.load(npz_file_path)

    if "max_stresses" not in data:
        print("Error: 'max_stresses' data not found in npz file")
        return

    max_stresses = data["max_stresses"]

    if len(max_stresses) == 0:
        print("Metrics file is empty. Nothing to plot.")
        return

    # 由于数据已经是每个episode的最终结果，直接使用
    episodes = np.arange(1, len(max_stresses) + 1)

    # 计算滑动平均
    smoothed_stresses = apply_moving_average(max_stresses, window_size)

    # 创建图形（使用全局设置的figsize）
    fig, ax = plt.subplots()

    # 绘制原始应力数据（浅色）
    ax.plot(episodes, max_stresses, color="lightgreen", linewidth=0.5, alpha=0.8, label="原始数据")

    # 绘制滑动平均（深色）
    ax.plot(episodes, smoothed_stresses, color="green", linewidth=1.0, label=f"滑动平均")

    # 添加限值线
    ax.axhline(y=limit_value, color="black", linestyle="--", linewidth=0.7, alpha=0.8)

    # 添加箭头标注
    # 计算合适的箭头位置（大约在图的中间位置）
    arrow_x = len(episodes) * 0.7  # 箭头x位置在70%处
    arrow_y = limit_value + (max(max_stresses) - limit_value) * 0.3  # 箭头y位置在限值线上方

    # 添加箭头指向限值线
    ax.annotate(
        "应力限值355MPa",
        xy=(arrow_x * 0.8, limit_value),  # 箭头指向的点
        xytext=(arrow_x, arrow_y),  # 文字位置
        fontsize=7,
        ha="center",
        va="bottom",
        arrowprops=dict(arrowstyle="->", color="black", lw=0.5, shrinkA=0, shrinkB=2),  # 箭头
    )

    # 设置坐标轴标签
    ax.set_xlabel("Episode")
    ax.set_ylabel("最大应力 (MPa)")

    # 添加图例
    ax.legend(frameon=False)

    # 保存图形
    save_path = save_figure(fig, save_dir, "stress_curve", file_format)
    plt.close()

    print(f"Stress curve saved to {save_path}")


def plot_all_curves(
    npz_file_path: str, save_dir: str = None, window_size: int = 10, file_format: str = "svg"
) -> None:
    """
    绘制所有三个图表

    Args:
        npz_file_path: metrics.npz文件路径
        save_dir: 保存目录
        window_size: 滑动平均窗口大小
        file_format: 输出文件格式 ('eps', 'pdf', 'svg', 'png')
    """
    print(f"Generating publication-quality plots in {file_format.upper()} format...")

    plot_reward_curve(npz_file_path, save_dir, window_size, file_format)
    plot_displacement_curve(npz_file_path, save_dir, file_format=file_format, window_size=window_size)
    plot_stress_curve(npz_file_path, save_dir, file_format=file_format, window_size=window_size)

    print("All plots generated successfully!")


def plot_episode_final_results(
    npz_file_path: str, save_dir: str = None, window_size: int = 1, file_format: str = "svg"
) -> None:
    """
    保持向后兼容性的函数（原有功能）
    """
    if window_size > 1:
        # 如果需要滑动平均，使用新的绘图函数
        plot_all_curves(npz_file_path, save_dir, window_size, file_format)
    else:
        # 否则使用原有逻辑，但调用新的绘图函数
        plot_all_curves(npz_file_path, save_dir, 1, file_format)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate publication-quality plots for journal submission.")
    parser.add_argument("--metrics_path", type=str, required=True, help="Path to the metrics.npz file.")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save the plots.")
    parser.add_argument(
        "--window_size",
        type=int,
        default=10,
        help="Window size for moving average in reward plot. Default is 10.",
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        choices=["all", "reward", "displacement", "stress"],
        default="all",
        help="Type of plot to generate. Default is 'all'.",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["eps", "pdf", "svg", "png"],
        default="eps",
        help="Output file format. Default is 'eps' (vector format).",
    )

    args = parser.parse_args()

    if args.plot_type == "all":
        plot_all_curves(args.metrics_path, args.save_dir, args.window_size, args.format)
    elif args.plot_type == "reward":
        plot_reward_curve(args.metrics_path, args.save_dir, args.window_size, args.format)
    elif args.plot_type == "displacement":
        plot_displacement_curve(
            args.metrics_path, args.save_dir, file_format=args.format, window_size=args.window_size
        )
    elif args.plot_type == "stress":
        plot_stress_curve(
            args.metrics_path, args.save_dir, file_format=args.format, window_size=args.window_size
        )
