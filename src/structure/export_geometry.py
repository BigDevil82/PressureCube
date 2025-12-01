import os
from typing import Dict, List, Set

from src.structure.structural_model import Beam, Plate, StructuralModel
from src.utils.model_utils import create_structural_model, load_config


def find_beam_by_nodes(n1: int, n2: int, beam_map: Dict[frozenset, int]) -> int:
    """通过节点ID查找梁ID"""
    key = frozenset([n1, n2])
    return beam_map.get(key)


def main():
    """
    主函数，用于读取StructuralModel实例信息，并把结构的几何信息写入文本文件。
    """
    # 加载配置并创建结构模型
    config = load_config("config.yaml")
    model = create_structural_model(config)

    # 创建输出目录
    output_dir = "results/output_geometry"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 导出节点坐标
    nodes_path = os.path.join(output_dir, "nodes.txt")
    with open(nodes_path, "w", encoding="utf-8") as f:
        # 对节点ID进行排序，以确保输出顺序一致
        sorted_node_ids = sorted(model.points.keys())
        for node_id in sorted_node_ids:
            point = model.points[node_id]
            f.write(f"{point.x}\t{point.y}\t{point.z}\n")
    print(f"节点坐标已导出到: {nodes_path}")

    # 2. 导出梁的两端节点编号
    beams_path = os.path.join(output_dir, "beams.txt")
    with open(beams_path, "w", encoding="utf-8") as f:
        sorted_beam_ids = sorted(model.beams.keys())
        for beam_id in sorted_beam_ids:
            beam = model.beams[beam_id]
            f.write(f"{beam.start_node+1},{beam.end_node+1}\n")
    print(f"梁连接信息已导出到: {beams_path}")

    # 3. 导出梁截面信息
    beam_sections_path = os.path.join(output_dir, "beam_sections.txt")
    section_name_to_index_key = {
        "FRAME": "frame_section_index",
        "MAIN": "main_beam_section_index",
        "SECONDARY": "secondary_beam_section_index",
        "BRACE": "brace_section_index",
    }
    with open(beam_sections_path, "w", encoding="utf-8") as f:
        sorted_beam_ids = sorted(model.beams.keys())
        index = 1
        for beam_id in sorted_beam_ids:
            beam = model.beams[beam_id]
            section_num = beam.beam_type.value + 1
            if beam.on_face not in ["top", "bottom"]:
                direction = beam.direction(model.points)
                if beam.on_face in ["front", "back"] and direction[2] > 0:
                    degree = 0
                else:
                    degree = 270
            else:
                degree = 0
            f.write(f"{index}\t{degree}\t{section_num}\n")
            index += 1

    print(f"梁截面信息已导出到: {beam_sections_path}")

    # 4. 导出板信息
    plates_path = os.path.join(output_dir, "plates.txt")

    # 创建一个从节点对到梁ID的映射，以便快速查找
    beam_map = {frozenset([b.start_node, b.end_node]): b.id for b in model.beams.values()}

    top_plates = []
    top_plates_id_path = os.path.join(output_dir, "bottom_plates_id.txt")
    plate_count = 0

    with open(plates_path, "w", encoding="utf-8") as f:
        sorted_plate_ids = sorted(model.plates.keys())
        for plate_id in sorted_plate_ids:
            plate = model.plates[plate_id]
            nodes = plate.nodes
            if len(nodes) != 4:
                print(f"警告: 板 {plate.id} 的节点数不为4，将跳过。")
                continue

            # 获取板的四条边对应的梁
            edges = [
                (nodes[0], nodes[1]),
                (nodes[1], nodes[2]),
                (nodes[2], nodes[3]),
                (nodes[3], nodes[0]),
            ]

            edge_beam_ids = []
            for n1, n2 in edges:
                beam_id = find_beam_by_nodes(n1, n2, beam_map)
                if beam_id is not None:
                    edge_beam_ids.append(beam_id + 1)
                else:
                    print(f"警告: 未能找到连接节点 {n1} 和 {n2} 的梁，板 {plate.id} 信息可能不完整。")

            if len(edge_beam_ids) == 4:
                f.write(",".join(map(str, edge_beam_ids)) + "\n")
                plate_count += 1
                if plate.on_face == "bottom":
                    top_plates.append(plate_count)

    with open(top_plates_id_path, "w", encoding="utf-8") as f:
        for plate_id in top_plates:
            f.write(f"{plate_id}\n")

    print(f"板信息已导出到: {plates_path}")


if __name__ == "__main__":
    main()
