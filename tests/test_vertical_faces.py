import os
import sys

import matplotlib.pyplot as plt

from structure.layout.params import StructuralParameters

# 添加src目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.structure.structural_model import BeamType, StructuralModel, Window


def test_vertical_faces():
    """测试前后立面生成"""
    # 创建结构参数，包含窗户
    params = StructuralParameters(
        length=7.2,  # 舱体长度 (y方向)
        width=3.6,  # 舱体宽度 (x方向)
        height=3.3,  # 舱体高度 (z方向)
        main_beam_spacing=1.2,  # 环向主梁间距
        secondary_beam_spacing=0.5,  # 环向次梁间距
        brace_spacing=0.3,  # 支撑梁间距
        windows=[
            # 前面窗户
            Window(
                y_position=1.8,  # 窗户中心在y轴的位置
                width=1.2,  # 窗户宽度
                height=1.0,  # 窗户高度
                bottom_height=1.0,  # 窗户底部离地高度
                on_face="front",  # 窗户所在面
            ),
            # 后面窗户
            Window(
                y_position=3.0,
                width=1.2,
                height=1.0,
                bottom_height=1.0,
                on_face="back",
            ),
        ],
    )

    # 创建结构模型
    model = StructuralModel(params)

    # 生成结构网格
    model.generate_grid_mesh()

    # 输出结构信息
    print("\n前后立面测试:")
    print(f"生成的节点数量: {len(model.points)}")
    print(f"生成的梁构件数量: {len(model.beams)}")
    print(f"结构总质量: {model.calculate_mass():.2f} kg")

    # 分析各类型梁的数量和总长度
    beam_type_counts = {beam_type: 0 for beam_type in BeamType}
    beam_type_lengths = {beam_type: 0.0 for beam_type in BeamType}

    for beam in model.beams.values():
        beam_type_counts[beam.beam_type] += 1
        beam_type_lengths[beam.beam_type] += beam.length(model.points)

    for beam_type in BeamType:
        print(
            f"{beam_type.name} 梁数量: {beam_type_counts[beam_type]}, "
            f"总长度: {beam_type_lengths[beam_type]:.2f} m"
        )

    # 返回模型供后续可能的可视化
    return model


def visualize_model(model):
    """可视化结构模型"""
    # fig = plt.figure(figsize=(12, 10))
    # ax = fig.add_subplot(111, projection="3d")

    # 定义不同类型梁的颜色
    colors = {
        BeamType.FRAME: "black",
        BeamType.MAIN: "blue",
        BeamType.SECONDARY: "green",
        BeamType.BRACE: "red",
    }

    # # 绘制所有梁
    # for beam_id, beam in model.beams.items():
    #     start_node = model.nodes[beam.start_node]
    #     end_node = model.nodes[beam.end_node]

    #     x = [start_node.x, end_node.x]
    #     y = [start_node.y, end_node.y]
    #     z = [start_node.z, end_node.z]

    #     ax.plot(x, y, z, color=colors[beam.beam_type], linewidth=1)

    # # 设置图表范围和标签
    # ax.set_xlabel("X (m)")
    # ax.set_ylabel("Y (m)")
    # ax.set_zlabel("Z (m)")

    # ax.set_xlim([0, model.params.width])
    # ax.set_ylim([0, model.params.length])
    # ax.set_zlim([0, model.params.height])

    # # 添加图例
    # for beam_type, color in colors.items():
    #     ax.plot([], [], [], color=color, label=beam_type.name)
    # ax.legend()

    # # 设置查看角度
    # ax.view_init(elev=20, azim=45)

    # plt.title("Pressure Vessel Structure (3D)")
    # plt.show()

    # 分别显示前面和后面视图
    # 前面视图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for beam_id, beam in model.beams.items():
        start_node = model.nodes[beam.start_node]
        end_node = model.nodes[beam.end_node]

        # 仅绘制前面的构件
        if abs(start_node.x) < 0.01 and abs(end_node.x) < 0.01:
            x = [start_node.x, end_node.x]
            y = [start_node.y, end_node.y]
            z = [start_node.z, end_node.z]

            ax.plot(x, y, z, color=colors[beam.beam_type], linewidth=1.5)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    ax.view_init(elev=0, azim=0)  # 正面视图

    ax.set_title("Front Face View")

    for beam_type, color in colors.items():
        ax.plot([], [], [], color=color, label=beam_type.name)
    ax.legend()

    plt.show()

    # 后面视图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for beam_id, beam in model.beams.items():
        start_node = model.nodes[beam.start_node]
        end_node = model.nodes[beam.end_node]

        # 仅绘制后面的构件
        if abs(start_node.x - model.params.width) < 0.01 and abs(end_node.x - model.params.width) < 0.01:
            x = [start_node.x, end_node.x]
            y = [start_node.y, end_node.y]
            z = [start_node.z, end_node.z]

            ax.plot(x, y, z, color=colors[beam.beam_type], linewidth=1.5)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    ax.view_init(elev=0, azim=180)  # 后视图

    ax.set_title("Back Face View")

    for beam_type, color in colors.items():
        ax.plot([], [], [], color=color, label=beam_type.name)
    ax.legend()

    plt.show()


if __name__ == "__main__":
    # 测试前后立面
    model = test_vertical_faces()
    visualize_model(model)
