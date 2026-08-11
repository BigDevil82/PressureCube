#!/usr/bin/env python3
"""Generate a standalone colorbar for finite-element contour figures."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FormatStrFormatter

FEM_COLORS = (
    "#000066",  # deep blue
    "#0000ff",
    "#00bfff",
    "#00ffff",
    "#00cc44",
    "#ffff00",
    "#ff7f00",
    "#ff0000",
    "#660000",  # deep red
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成用于有限元应力云图的独立色带图。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--min", dest="value_min", type=float, default=-49.7, help="色带下限")
    parser.add_argument("--max", dest="value_max", type=float, default=163.72, help="色带上限")
    parser.add_argument("--ticks", type=int, default=8, help="刻度数量（含上下限）")
    parser.add_argument("--label", default="Stress/MPa", help="色带标题，传入空字符串可隐藏")
    parser.add_argument(
        "--orientation", choices=("vertical", "horizontal"), default="vertical", help="色带方向"
    )
    parser.add_argument("--format", dest="number_format", default="%.1f", help="刻度数字格式")
    parser.add_argument(
        "--output", type=Path, default=Path("results/collorbar/colorbar.png"), help="输出文件路径"
    )
    parser.add_argument("--dpi", type=int, default=600, help="位图分辨率")
    parser.add_argument("--transparent", action="store_true", help="使用透明背景")
    return parser.parse_args()


def generate_colorbar(args: argparse.Namespace) -> None:
    if args.value_max <= args.value_min:
        raise ValueError("--max 必须大于 --min")
    if args.ticks < 2:
        raise ValueError("--ticks 必须不小于 2")

    is_vertical = args.orientation == "vertical"
    figsize = (0.85, 5.2) if is_vertical else (6.0, 1.25)
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(
        left=0.18 if is_vertical else 0.08,
        right=0.58 if is_vertical else 0.96,
        bottom=0.08 if is_vertical else 0.42,
        top=0.96 if is_vertical else 0.88,
    )

    cmap = LinearSegmentedColormap.from_list("fem_deep_blue_red", FEM_COLORS, N=256)
    norm = Normalize(vmin=args.value_min, vmax=args.value_max)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])

    colorbar = fig.colorbar(mappable, cax=ax, orientation=args.orientation)
    colorbar.set_ticks(
        [args.value_min + i * (args.value_max - args.value_min) / (args.ticks - 1) for i in range(args.ticks)]
    )
    colorbar.ax.tick_params(labelsize=12, width=0.8, length=3)
    colorbar.outline.set_linewidth(0.8)
    formatter = FormatStrFormatter(args.number_format)
    if is_vertical:
        colorbar.ax.yaxis.set_major_formatter(formatter)
        tick_labels = colorbar.ax.get_yticklabels()
    else:
        colorbar.ax.xaxis.set_major_formatter(formatter)
        tick_labels = colorbar.ax.get_xticklabels()
    for tick_label in tick_labels:
        tick_label.set_fontfamily("Times New Roman")
    if args.label:
        colorbar.set_label(args.label, fontsize=10, labelpad=8, fontfamily="Times New Roman")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.output,
        dpi=args.dpi,
        bbox_inches="tight",
        pad_inches=0.03,
        transparent=args.transparent,
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    generate_colorbar(args)
    print(f"Colorbar saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
