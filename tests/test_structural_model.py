import os
import sys

import matplotlib.pyplot as plt

# 添加src目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.structure.structural_model import BeamType, StructuralModel, StructuralParameters, Window


def test_structural_model():
    """测试结构模型生成"""
    # 创建结构参数
    params = StructuralParameters(
        length=7.2,  # 舱体长度 (y方向)
        width=3.6,  # 舱体宽度 (x方向)
        height=3.3,  # 舱体高度 (z方向)
        main_beam_spacing=1.2,  # 环向主梁间距
        secondary_beam_spacing=0.5,  # 环向次梁间距
        brace_spacing=0.5,  # 支撑梁间距
        windows=[
            # 左侧面窗户
            Window(
                y_position=1.2,  # 窗户中心在x轴的位置(对于侧面)
                width=1.2,  # 窗户宽度
                height=1.0,  # 窗户高度
                bottom_height=1.0,  # 窗户底部离地高度
                on_face="left",  # 窗户所在面
            ),
            # 右侧面门
            Window(
                y_position=1.2,
                width=1.2,
                height=2.0,
                bottom_height=0,  # 门从地面开始
                on_face="right",
                is_door=True,  # 这是一个门
            ),
            # 保留前后面的窗户，确保完整性
            Window(
                y_position=1.8,
                width=1.2,
                height=1.0,
                bottom_height=1.0,
                on_face="front",
            ),
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
    print(f"生成的节点数量: {len(model.points)}")
    print(f"生成的梁构件数量: {len(model.beams)}")
    print(f"结构总质量: {model.calculate_mass():.2f} kg")

    # 分析各类型梁的数量
    beam_type_counts = {beam_type: 0 for beam_type in BeamType}
    for beam in model.beams.values():
        beam_type_counts[beam.beam_type] += 1

    for beam_type, count in beam_type_counts.items():
        print(f"{beam_type.name} 梁数量: {count}")

    # 返回模型供后续可能的可视化
    return model


def visualize_model(model):
    """可视化结构模型"""
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

    # 等轴绘制
    plt.axis("equal")

    # 添加图例
    for beam_type, color in colors.items():
        ax.plot([], [], [], color=color, label=beam_type.name)
    ax.legend()

    plt.title("Pressure Vessel Structure Model")
    plt.show()


def test_model_with_windows():
    """测试带窗户的结构模型"""
    # 创建结构参数
    params = StructuralParameters(
        length=7.2,  # 舱体长度 (y方向)
        width=3.6,  # 舱体宽度 (x方向)
        height=3.3,  # 舱体高度 (z方向)
        main_beam_spacing=1.05,  # 环向主梁间距
        secondary_beam_spacing=0.5,  # 环向次梁间距
        brace_spacing=0.3,  # 支撑梁间距,
        windows=[
            Window(
                y_position=3.6,  # 窗户中心在y轴的位置
                width=1.2,  # 窗户宽度
                height=1.0,  # 窗户高度
                bottom_height=1.0,  # 窗户底部离地高度
                on_face="front",  # 窗户所在面 ('front', 'back', 'right')
            ),
            Window(y_position=3.6, width=1.2, height=1.0, bottom_height=1.0, on_face="right"),
        ],
    )

    # 创建结构模型
    model = StructuralModel(params)

    # 生成结构网格
    model.generate_mesh()

    # 输出结构信息
    print("\n带窗户的模型:")
    print(f"生成的节点数量: {len(model.points)}")
    print(f"生成的梁构件数量: {len(model.beams)}")
    print(f"结构总质量: {model.calculate_mass():.2f} kg")

    # 返回模型供后续可能的可视化
    return model


def test_grid_model():
    """测试网格化建模方法"""
    # 创建结构参数
    params = StructuralParameters(
        length=9.5,  # 舱体长度 (y方向)
        width=3.6,  # 舱体宽度 (x方向)
        height=3.3,  # 舱体高度 (z方向)
        main_beam_spacing=1.2,  # 环向主梁间距
        secondary_beam_spacing=0.5,  # 环向次梁间距
        brace_spacing=0.4,  # 支撑梁间距
    )

    # 创建结构模型
    model = StructuralModel(params)

    # 使用网格化方法生成模型
    model.generate_grid_mesh()

    # 输出结构信息
    print("\n网格化建模方法:")
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
            f"{beam_type.name} 梁数量: {beam_type_counts[beam_type]}, 总长度: {beam_type_lengths[beam_type]:.2f} m"
        )

    # 返回模型供后续可能的可视化
    return model


if __name__ == "__main__":
    # 测试基本模型
    # model = test_structural_model()
    # visualize_model(model)

    # 测试网格化模型
    grid_model = test_grid_model()
    visualize_model(grid_model)

    # 测试带窗户的模型
    # model_with_windows = test_model_with_windows()
    # visualize_model(model_with_windows)
