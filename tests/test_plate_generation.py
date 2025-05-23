"""
测试钢板生成功能的脚本
"""

import os
import sys

# 添加src目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.model_utils import create_structural_model, load_config


def test_plate_generation():
    """测试钢板生成功能"""

    config = load_config("config.yaml")
    model = create_structural_model(config)

    # 打印统计信息
    print("=== 结构统计信息 ===")
    print(f"节点数量: {len(model.points)}")
    print(f"梁构件数量: {len(model.beams)}")
    print(f"钢板数量: {len(model.plates)}")

    # 打印钢板统计
    plate_stats = model.get_plate_statistics()
    print("\n=== 钢板分面统计 ===")
    for face, count in plate_stats.items():
        if face != "total":
            print(f"{face}面: {count}块")
    print(f"总计: {plate_stats['total']}块")

    # 打印总质量
    total_mass = model.calculate_mass()
    print(f"\n结构总质量: {total_mass:.2f} kg")

    # 打印前几个钢板的信息
    print("\n=== 前5块钢板信息 ===")
    for i, (plate_id, plate) in enumerate(model.plates.items()):
        if i >= 5:
            break
        area = plate.area(model.points)
        print(f"钢板{plate_id}: 面={plate.on_face}, 节点={plate.nodes}, 面积={area:.3f}m²")


if __name__ == "__main__":
    test_plate_generation()
