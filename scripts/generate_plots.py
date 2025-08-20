#!/usr/bin/env python3
"""
生成符合《建筑结构学报》要求的论文插图脚本

使用方法:
python scripts/generate_plots.py --metrics_path results/your_experiment/metrics.npz

该脚本将生成三个独立的图表文件：
- reward_curve: 累积奖励变化曲线（含滑动平均）
- displacement_curve: 最大位移变化曲线（含14.4mm限值线）
- stress_curve: 最大应力变化曲线（含355MPa限值线）

支持的输出格式：
- EPS: 矢量图，适合期刊投稿（推荐）
- PDF: 矢量图，适合文档嵌入
- SVG: 矢量图，适合网页显示
- PNG: 位图，适合预览
"""

import argparse
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.post_process.plot import (
    plot_all_curves,
    plot_displacement_curve,
    plot_reward_curve,
    plot_stress_curve,
)


def main():
    parser = argparse.ArgumentParser(
        description="生成符合《建筑结构学报》要求的论文插图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 生成所有图表（EPS矢量格式，推荐用于期刊投稿）
  python scripts/generate_plots.py --metrics_path results/experiment1/metrics.npz
  
  # 生成PDF格式的图表
  python scripts/generate_plots.py --metrics_path results/experiment1/metrics.npz --format pdf
  
  # 只生成奖励曲线，使用20个episode的滑动平均
  python scripts/generate_plots.py --metrics_path results/experiment1/metrics.npz --plot_type reward --window_size 20
  
  # 指定输出目录
  python scripts/generate_plots.py --metrics_path results/experiment1/metrics.npz --output_dir figures/ --format svg

支持的格式：
  - eps: EPS矢量格式（推荐，适合期刊投稿）
  - pdf: PDF矢量格式（适合文档嵌入）
  - svg: SVG矢量格式（适合网页显示）
  - png: PNG位图格式（适合预览）
        """,
    )

    parser.add_argument("--metrics_path", type=str, required=True, help="训练指标文件路径 (metrics.npz)")

    parser.add_argument(
        "--output_dir", type=str, default=None, help="图表输出目录 (默认为metrics.npz所在目录)"
    )

    parser.add_argument(
        "--plot_type",
        type=str,
        choices=["all", "reward", "displacement", "stress"],
        default="all",
        help="要生成的图表类型 (默认: all)",
    )

    parser.add_argument("--window_size", type=int, default=10, help="奖励曲线滑动平均窗口大小 (默认: 10)")

    parser.add_argument(
        "--format",
        type=str,
        choices=["eps", "pdf", "svg", "png"],
        default="svg",
        help="输出文件格式 (默认: eps矢量格式)",
    )

    args = parser.parse_args()

    # 检查输入文件是否存在
    if not os.path.exists(args.metrics_path):
        print(f"错误: 找不到指标文件 {args.metrics_path}")
        sys.exit(1)

    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.metrics_path)

    print(f"正在从 {args.metrics_path} 加载数据...")
    print(f"图表将保存到 {args.output_dir}")
    print(f"图表类型: {args.plot_type}")
    print(f"输出格式: {args.format.upper()}")

    if args.plot_type == "reward":
        print(f"滑动平均窗口大小: {args.window_size}")

    # 生成图表
    try:
        if args.plot_type == "all":
            plot_all_curves(args.metrics_path, args.output_dir, args.window_size, args.format)
        elif args.plot_type == "reward":
            plot_reward_curve(args.metrics_path, args.output_dir, args.window_size, args.format)
        elif args.plot_type == "displacement":
            plot_displacement_curve(args.metrics_path, args.output_dir, file_format=args.format)
        elif args.plot_type == "stress":
            plot_stress_curve(args.metrics_path, args.output_dir, file_format=args.format)

        print("\n✅ 图表生成完成!")

    except Exception as e:
        print(f"❌ 生成图表时发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
