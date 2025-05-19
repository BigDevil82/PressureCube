import argparse
import os
import sys

import numpy as np
import torch
import yaml

from src.analysis.opensees_analysis import OpenSeesAnalyzer
from src.graph.hebg import HEBG
from src.reinforcement.rl_agent import DDPGAgent
from src.structure.structural_model import BeamType, StructuralModel, StructuralParameters, Window
from src.utils.constants import *


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="高海拔增压建筑智能设计")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument(
        "--mode", type=str, choices=["train", "test", "visualize"], default="visualize", help="运行模式"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备: cuda/cpu"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--max_episodes", type=int, default=1000, help="最大训练回合数")
    return parser.parse_args()


def load_config(config_path):
    """加载配置文件"""
    if not os.path.exists(config_path):
        print(f"配置文件 {config_path} 不存在，使用默认配置")
        return {
            "structural_params": {
                "length": 7.2,
                "width": 3.6,
                "height": 3.3,
                "main_beam_spacing": 1.05,
                "secondary_beam_spacing": 0.5,
                "brace_spacing": 0.3,
                "windows": [
                    {"y_position": 3.6, "width": 1.2, "height": 1.0, "bottom_height": 1.0, "on_face": "front"}
                ],
            },
            "rl_params": {
                "actor_lr": 1e-4,
                "critic_lr": 1e-3,
                "gamma": 0.99,
                "tau": 0.005,
                "batch_size": 64,
                "buffer_size": 10000,
                "update_interval": 5,
            },
            "gnn_params": {"in_channels": 10, "hidden_channels": 64, "out_channels": 32},
        }

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def create_structural_model(config):
    """创建结构模型"""
    params_dict = config["structural_params"]

    # 处理窗户参数
    windows = []
    if "windows" in params_dict:
        for window_dict in params_dict["windows"]:
            windows.append(Window(**window_dict))

    # 创建结构参数对象
    params = StructuralParameters(
        length=params_dict["length"],
        width=params_dict["width"],
        height=params_dict["height"],
        main_beam_spacing=params_dict["main_beam_spacing"],
        secondary_beam_spacing=params_dict["secondary_beam_spacing"],
        brace_spacing=params_dict["brace_spacing"],
        windows=windows,
    )

    # 创建结构模型并生成网格
    model = StructuralModel(params)

    # 使用网格化建模方法
    model.generate_grid_mesh()

    return model


def visualize_model(model):
    """可视化结构模型"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 定义不同类型梁的颜色
    colors = {
        BeamType.FRAME: "black",
        BeamType.MAIN: "blue",
        BeamType.SECONDARY: "green",
        BeamType.BRACE: "red",
    }

    # 绘制所有梁
    for beam_id, beam in model.beams.items():
        start_node = model.nodes[beam.start_node]
        end_node = model.nodes[beam.end_node]

        x = [start_node.x, end_node.x]
        y = [start_node.y, end_node.y]
        z = [start_node.z, end_node.z]

        ax.plot(x, y, z, color=colors[beam.beam_type], linewidth=1)

    # 设置图表范围和标签
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    ax.set_xlim([0, model.params.width])
    ax.set_ylim([0, model.params.length])
    ax.set_zlim([0, model.params.height])

    # 添加图例
    for beam_type, color in colors.items():
        ax.plot([], [], [], color=color, label=beam_type.name)
    ax.legend()

    plt.title("增压舱体结构模型")
    plt.show()


def analyze_structure(model):
    """分析结构性能"""
    analyzer = OpenSeesAnalyzer()

    # 转换节点格式
    nodes_dict = {}
    for node_id, node in model.nodes.items():
        nodes_dict[node_id] = np.array([node.x, node.y, node.z])

    # 转换梁格式
    elements_dict = {}
    sections_dict = {}
    for beam_id, beam in model.beams.items():
        elements_dict[beam_id] = (beam.start_node, beam.end_node)
        section = model.sections[beam.section]
        if beam.section not in sections_dict:
            sections_dict[beam.section] = {
                "area": section.area,
                "Ix": section.Ix,
                "Iy": section.Iy,
                "Iz": section.Iz,
            }

    # 创建OpenSees模型
    analyzer.create_model(nodes_dict, elements_dict, sections_dict)

    # 生成荷载
    loads = {}
    internal_pressure = INTERNAL_PRESSURE
    for node_id, node in model.nodes.items():
        # 上表面节点受内压力
        if node.z > model.params.height - 0.01:
            area_tributary = model.params.width * model.params.length / 10  # 简化的受力面积估计
            force = np.array([0, 0, internal_pressure * area_tributary, 0, 0, 0])
            loads[node_id] = force

    # 应用荷载
    analyzer.apply_loads(loads)

    # 执行分析
    results = analyzer.analyze()

    # 检查约束条件
    satisfied, constraints = analyzer.check_constraints(results)

    return satisfied, constraints, results


def train(args, config):
    """训练强化学习智能体"""
    print("开始训练...")
    # TODO: 实现训练逻辑
    pass


def test(args, config):
    """测试训练好的智能体"""
    print("开始测试...")
    # TODO: 实现测试逻辑
    pass


def main():
    # 解析命令行参数
    args = parse_args()

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 加载配置
    config = load_config(args.config)

    # 根据模式执行不同逻辑
    if args.mode == "train":
        train(args, config)
    elif args.mode == "test":
        test(args, config)
    elif args.mode == "visualize":
        # 创建结构模型
        model = create_structural_model(config)

        # 输出结构信息
        print(f"节点数量: {len(model.nodes)}")
        print(f"梁构件数量: {len(model.beams)}")
        print(f"结构总质量: {model.calculate_mass():.2f} kg")

        # 分析各类型梁的数量和总长度
        beam_type_counts = {beam_type: 0 for beam_type in BeamType}
        beam_type_lengths = {beam_type: 0.0 for beam_type in BeamType}

        for beam in model.beams.values():
            beam_type_counts[beam.beam_type] += 1
            beam_type_lengths[beam.beam_type] += beam.length(model.nodes)

        for beam_type in BeamType:
            print(
                f"{beam_type.name} 梁数量: {beam_type_counts[beam_type]}, 总长度: {beam_type_lengths[beam_type]:.2f} m"
            )

        # 可视化模型
        visualize_model(model)

        # 尝试分析结构(如果OpenSees可用)
        try:
            satisfied, constraints, results = analyze_structure(model)
            print(f"结构约束满足: {satisfied}")
            for constraint_name, is_satisfied in constraints.items():
                print(f"  {constraint_name}: {'满足' if is_satisfied else '不满足'}")
        except Exception as e:
            print(f"结构分析失败: {e}")


if __name__ == "__main__":
    main()
