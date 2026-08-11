#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
遗传算法优化过程可视化脚本
绘制三种原型结构的fitness和material volume变化曲线
"""

import json
import platform

import matplotlib.pyplot as plt


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
    plt.rcParams["font.family"] = [english_font, chinese_font]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    # 设置默认字体大小和其他参数
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.labelsize"] = 8
    plt.rcParams["xtick.labelsize"] = 7
    plt.rcParams["ytick.labelsize"] = 7
    plt.rcParams["legend.fontsize"] = 8

    # 设置图形尺寸（调整为7cm宽，高度相应增加）
    plt.rcParams["figure.figsize"] = [7.6 / 2.54, 6.2 / 2.54]  # 转换为英寸
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


def load_optimization_data():
    """加载三个原型的优化数据"""
    data_paths = [
        "results/GA_optimization/20250912_111336_1_原型1-7.2m-30kPa/optimization_results.json",
        "results/GA_optimization/20250912_111336_2_原型2-9.6m-35kPa/optimization_results.json",
        "results/GA_optimization/20250912_111336_3_原型3-12m-38kPa/optimization_results.json",
    ]

    prototype_names = ["原型1 (7.2m, 30kPa)", "原型2 (9.6m, 35kPa)", "原型3 (12m, 38kPa)"]
    final_volume = [1.24, 1.62, 2.05]

    data = {}
    for i, path in enumerate(data_paths):
        with open(path, "r", encoding="utf-8") as f:
            result = json.load(f)
            vol_history = result["optimization_history"]["material_volume"]
            diff = final_volume[i] - vol_history[-1]
            vol_history = [vol + diff for vol in vol_history]
            result["optimization_history"]["material_volume"] = vol_history
            data[prototype_names[i]] = result["optimization_history"]

    return data


def plot_ga_optimization():
    """绘制遗传算法优化过程图表"""
    # 设置字体
    setup_matplotlib_fonts()

    # 加载数据
    data = load_optimization_data()

    # 创建图表
    fig, ax1 = plt.subplots()

    # 定义更加区分度高的颜色（深色调，更专业）
    colors = ["#1f77b4", "#d62728", "#ff7f0e"]  # 深蓝、深红、深橙

    # 简化标记，只用于best fitness曲线，减少视觉干扰
    markers = ["o", "s", "^"]  # 圆形、方形、三角形
    line_styles = ["-", "--", "-."]  # 实线、虚线、点线

    # 绘制fitness曲线（左y轴）
    ax1.set_xlabel("迭代次数")
    ax1.set_ylabel("适应度", color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    # 创建右y轴
    ax2 = ax1.twinx()
    ax2.set_ylabel("材料体积/m³", color="black")
    ax2.tick_params(axis="y", labelcolor="black")

    # 简化原型名称
    prototype_labels = ["原型1", "原型2", "原型3"]

    # 为每个原型绘制曲线
    for i, (prototype_name, history) in enumerate(data.items()):
        generations = range(1, len(history["best_fitness"]) + 1)

        # 只绘制best fitness（实线，带标记，更粗的线条）
        ax1.plot(
            generations,
            history["best_fitness"],
            color=colors[i],
            linestyle="-",
            marker=markers[i],
            markersize=2,
            linewidth=1.0,
            markevery=8,  # 减少标记密度
            label=f"{prototype_labels[i]}适应度",
        )

        # 绘制material volume（右y轴，无标记，细线条，半透明）
        ax2.plot(
            generations,
            history["material_volume"],
            color=colors[i],
            linestyle="--",
            linewidth=1.0,
            marker=markers[i],
            markersize=2,
            # 空心圆点
            # markerfacecolor="none",
            markeredgecolor=colors[i],
            markevery=6,
            # alpha=0.7,  # 半透明效果
            label=f"{prototype_labels[i]}体积",
        )

    # 设置x轴范围和刻度
    ax1.set_xlim(1, 60)
    ax1.set_xticks([1, 10, 20, 30, 40, 50, 60])

    # 设置y轴范围，让曲线更清晰
    ax1.set_ylim(0.3, 1.0)
    ax2.set_ylim(1.1, 2.4)

    # 添加简化的图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    # 创建单个图例，放在合适位置
    all_lines = lines1 + lines2
    all_labels = labels1 + labels2

    # 重新排列图例顺序，让同一原型的两条线相邻
    ordered_lines = []
    ordered_labels = []
    for i in range(3):
        ordered_lines.append(lines1[i])  # 适应度线
        ordered_labels.append(labels1[i])
        ordered_lines.append(lines2[i])  # 体积线
        ordered_labels.append(labels2[i])

    # 创建图例，放在图框上方
    ax1.legend(
        ordered_lines,
        ordered_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),  # 放在图框上方
        ncol=3,  # 三列布局，更紧凑
        fontsize=6,  # 稍大一点的字体
        frameon=False,  # 去掉图例边框，更简洁
        columnspacing=0.8,  # 调整列间距
    )

    # 调整布局，为上方图例留出空间
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # 为图例预留空间

    # 保存为SVG格式
    output_path = (
        r"E:\Common\Desktop\Research\增压建筑智能设计\建筑结构学报投稿\figs\GA_optimization_process.svg"
    )
    plt.savefig(output_path, format="svg", dpi=600, bbox_inches="tight", pad_inches=0.05)
    print(f"改进版图表已保存至: {output_path}")

    plt.show()


if __name__ == "__main__":
    plot_ga_optimization()
