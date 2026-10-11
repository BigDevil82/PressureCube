import os
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.structure.enums import BeamType
from src.structure.layout.params import EngineeringLayoutParameters, WindowParameters
from src.structure.section_assignment import SectionAssignment
from src.structure.section_library import SectionLibrary
from src.structure.structural_model import Beam, StructuralModel
from src.utils.constants import INTERNAL_PRESSURE, LOAD_DEAD_BOTTOM_KPA, LOAD_LIVE_BOTTOM_KPA


LOAD_DIRECTION_LABELS = {
    "X": "X<sub>A</sub>",
    "Y": "Y<sub>A</sub>",
    "Z": "Z<sub>A</sub>",
}

PRESSURE_LOAD_BY_FACE_KPA = {
    "top": ("Z", INTERNAL_PRESSURE / 1000.0),
    "bottom": ("Z", -INTERNAL_PRESSURE / 1000.0),
    "left": ("Y", -INTERNAL_PRESSURE / 1000.0),
    "right": ("Y", INTERNAL_PRESSURE / 1000.0),
    "front": ("X", INTERNAL_PRESSURE / 1000.0),
    "back": ("X", -INTERNAL_PRESSURE / 1000.0),
}


def find_line_by_nodes(n1: int, n2: int, line_map: Dict[frozenset, int]) -> int | None:
    """通过节点ID查找已导出的线编号。"""
    key = frozenset([n1, n2])
    return line_map.get(key)


def add_boundary_line(n1: int, n2: int, line_records: List[tuple[int, int]], line_map: Dict[frozenset, int]) -> int:
    """追加一条只用于围合面的边界线，并返回 1-based 线编号。"""
    line_no = len(line_records) + 1
    line_records.append((n1, n2))
    line_map[frozenset([n1, n2])] = line_no
    return line_no


def is_node_on_segment(node_id: int, start_id: int, end_id: int, model: StructuralModel, tol: float = 1e-6) -> bool:
    node = model.points[node_id]
    start = model.points[start_id]
    end = model.points[end_id]

    vx, vy, vz = end.x - start.x, end.y - start.y, end.z - start.z
    wx, wy, wz = node.x - start.x, node.y - start.y, node.z - start.z
    length_sq = vx * vx + vy * vy + vz * vz
    if length_sq < tol:
        return False

    cross_x = wy * vz - wz * vy
    cross_y = wz * vx - wx * vz
    cross_z = wx * vy - wy * vx
    cross_sq = cross_x * cross_x + cross_y * cross_y + cross_z * cross_z
    if cross_sq > tol * tol * length_sq:
        return False

    dot = wx * vx + wy * vy + wz * vz
    return -tol <= dot <= length_sq + tol


def segment_parameter(node_id: int, start_id: int, end_id: int, model: StructuralModel) -> float:
    node = model.points[node_id]
    start = model.points[start_id]
    end = model.points[end_id]

    vx, vy, vz = end.x - start.x, end.y - start.y, end.z - start.z
    wx, wy, wz = node.x - start.x, node.y - start.y, node.z - start.z
    length_sq = vx * vx + vy * vy + vz * vz
    if length_sq == 0:
        return 0.0
    return (wx * vx + wy * vy + wz * vz) / length_sq


def find_boundary_line_ids(
    n1: int,
    n2: int,
    model: StructuralModel,
    line_records: List[tuple[int, int]],
    line_map: Dict[frozenset, int],
) -> list[int]:
    """返回一条板边对应的 RFEM 边界线编号；可由多段既有线组成。

    若既有线无法连续覆盖整条边，则追加一条边界辅助线，避免板被遗漏。
    """
    direct_line_id = find_line_by_nodes(n1, n2, line_map)
    if direct_line_id is not None:
        return [direct_line_id]

    segment_nodes = [
        node_id for node_id in model.points if is_node_on_segment(node_id, n1, n2, model)
    ]
    segment_nodes = sorted(
        set(segment_nodes),
        key=lambda node_id: (segment_parameter(node_id, n1, n2, model), node_id),
    )

    if segment_nodes[:1] != [n1] or segment_nodes[-1:] != [n2]:
        return [add_boundary_line(n1, n2, line_records, line_map)]

    boundary_line_ids = []
    for start_id, end_id in zip(segment_nodes, segment_nodes[1:]):
        line_id = find_line_by_nodes(start_id, end_id, line_map)
        if line_id is None:
            return [add_boundary_line(n1, n2, line_records, line_map)]
        boundary_line_ids.append(line_id)

    return boundary_line_ids or [add_boundary_line(n1, n2, line_records, line_map)]


def calculate_beam_direction(beam: Beam, model: StructuralModel) -> tuple[float, float, float]:
    """根据梁起止节点坐标计算方向向量。"""
    start = model.points[beam.start_node]
    end = model.points[beam.end_node]
    return end.x - start.x, end.y - start.y, end.z - start.z


def surface_load_row(surface_no: int, axis: str, magnitude_kpa: float) -> str:
    return (
        f"{surface_no}\t力\t均匀\t1 - 全局 XYZ\t"
        f"{LOAD_DIRECTION_LABELS[axis]}\t{magnitude_kpa:.2f}\n"
    )


def build_default_engineering_model() -> StructuralModel:
    params = EngineeringLayoutParameters(
        length=7.2,
        width=3.7,
        height=3.3,
        main_beam_spacing=0.95,
        add_support_beam=True,
        support_beam_max_spacing=0.55,
        window=WindowParameters(front_spans=[], back_spans=[], right_spans=[1, 2]),
    )

    assignment = SectionAssignment.by_beam_type(
        {BeamType.FRAME: 2, BeamType.MAIN: 8, BeamType.SECONDARY: 7, BeamType.BRACE: 14}
    )

    model = StructuralModel(
        params, SectionLibrary("src/data/steel_sections.json"), section_assignment=assignment
    )
    model.generate()
    return model


def main(model: StructuralModel | None = None, output_dir: str | None = None):
    """
    主函数，用于读取StructuralModel实例信息，并把结构的几何信息写入文本文件。
    """
    if model is None:
        model = build_default_engineering_model()

    # 创建输出目录
    if output_dir is None:
        output_dir = "results/output_geometry"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 导出节点坐标
    nodes_path = os.path.join(output_dir, "1_nodes.txt")
    with open(nodes_path, "w", encoding="utf-8") as f:
        # 对节点ID进行排序，以确保输出顺序一致
        sorted_node_ids = sorted(model.points.keys())
        for node_id in sorted_node_ids:
            point = model.points[node_id]
            f.write(f"{point.x}\t{point.y}\t{point.z}\n")
    print(f"节点坐标已导出到: {nodes_path}")

    sorted_beam_ids = sorted(model.beams.keys())
    line_records = [
        (model.beams[beam_id].start_node, model.beams[beam_id].end_node)
        for beam_id in sorted_beam_ids
    ]
    line_map = {
        frozenset([start_node, end_node]): index + 1
        for index, (start_node, end_node) in enumerate(line_records)
    }

    plate_rows = []
    bottom_plates = []
    sorted_plate_ids = sorted(model.plates.keys())
    for plate_id in sorted_plate_ids:
        plate = model.plates[plate_id]
        nodes = plate.nodes
        if len(nodes) != 4:
            print(f"警告: 板 {plate.id} 的节点数不为4，将跳过。")
            continue

        edges = [
            (nodes[0], nodes[1]),
            (nodes[1], nodes[2]),
            (nodes[2], nodes[3]),
            (nodes[3], nodes[0]),
        ]

        boundary_line_ids = []
        for n1, n2 in edges:
            boundary_line_ids.extend(find_boundary_line_ids(n1, n2, model, line_records, line_map))

        plate_type_code = "2" if plate.panel_type == "opening" else "1"
        plate_no = len(plate_rows) + 1
        plate_rows.append(
            {
                "no": plate_no,
                "boundary_line_ids": boundary_line_ids,
                "plate_type_code": plate_type_code,
                "face": plate.on_face,
            }
        )
        if plate.on_face == "bottom":
            bottom_plates.append(plate_no)

    # 2. 导出梁的两端节点编号
    beams_path = os.path.join(output_dir, "2_lines.txt")
    with open(beams_path, "w", encoding="utf-8") as f:
        for start_node, end_node in line_records:
            f.write(f"{start_node+1},{end_node+1}\n")
    print(f"梁连接信息已导出到: {beams_path}")

    # 3. 导出梁截面信息
    beam_sections_path = os.path.join(output_dir, "3_beams.txt")
    with open(beam_sections_path, "w", encoding="utf-8") as f:
        index = 1
        for beam_id in sorted_beam_ids:
            beam = model.beams[beam_id]
            section_num = beam.beam_type.value + 1
            if beam.on_face not in ["top", "bottom"]:
                direction = calculate_beam_direction(beam, model)
                if beam.on_face in ["front", "back"] and direction[2] > 0:
                    degree = 0
                else:
                    degree = 270
            else:
                degree = 0
            f.write(f"{index}\t梁\t均匀\t{degree}\t{section_num}\n")
            index += 1

    print(f"梁截面信息已导出到: {beam_sections_path}")

    # 4. 导出板信息
    plates_path = os.path.join(output_dir, "4_plates.txt")

    bottom_plates_id_path = os.path.join(output_dir, "5_bottom_plates_id.txt")

    with open(plates_path, "w", encoding="utf-8") as f:
        for plate_row in plate_rows:
            f.write(
                ",".join(map(str, plate_row["boundary_line_ids"]))
                + f"\t标准\t平面\t{plate_row['plate_type_code']}\n"
            )

    with open(bottom_plates_id_path, "w", encoding="utf-8") as f:
        for plate_id in bottom_plates:
            f.write(f"{plate_id}\n")

    print(f"板信息已导出到: {plates_path}")

    dead_loads_path = os.path.join(output_dir, "6_dead_loads.txt")
    with open(dead_loads_path, "w", encoding="utf-8") as f:
        for plate_row in plate_rows:
            if plate_row["face"] == "bottom":
                f.write(surface_load_row(plate_row["no"], "Z", -LOAD_DEAD_BOTTOM_KPA))
    print(f"恒载已导出到: {dead_loads_path}")

    live_loads_path = os.path.join(output_dir, "7_live_loads.txt")
    with open(live_loads_path, "w", encoding="utf-8") as f:
        for plate_row in plate_rows:
            if plate_row["face"] == "bottom":
                f.write(surface_load_row(plate_row["no"], "Z", -LOAD_LIVE_BOTTOM_KPA))
    print(f"活载已导出到: {live_loads_path}")

    pressure_loads_path = os.path.join(output_dir, "8_pressure_loads.txt")
    with open(pressure_loads_path, "w", encoding="utf-8") as f:
        for plate_row in plate_rows:
            direction = PRESSURE_LOAD_BY_FACE_KPA.get(plate_row["face"])
            if direction is None:
                continue
            axis, magnitude_kpa = direction
            f.write(surface_load_row(plate_row["no"], axis, magnitude_kpa))
    print(f"压力荷载已导出到: {pressure_loads_path}")


if __name__ == "__main__":
    main()
