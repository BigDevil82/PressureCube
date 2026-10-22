"""Extract BIM section geometry and reference values from supplied standards."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.integrations.webcad.exporter import write_json

STANDARDS = {
    "GB/T 11263-2024": {
        "file": "GBT 11263-2024 热轧H型钢和剖分T型钢.pdf", "table": "A.1",
        "radius_basis": {"clause": "6.1.2", "pdf_page": 9, "printed_page": 3, "note": "圆角半径为参考值"},
    },
    "GB/T 706-2016": {
        "file": "706-2016-热轧型钢.pdf", "table": "A.2",
        "radius_basis": {"pdf_page": 12, "printed_page": 10, "note": "r、r1 用于孔型设计，不做交货条件"},
    },
    "GB/T 3094-2012": {
        "file": "3094-2012-gbt-e-300.pdf", "table": "A.1",
        "radius_basis": {"pdf_page": 14, "printed_page": 12, "note": "表 A.1 注：理论重量计算参考圆角；S≤6 时 R=1.5S，S>6 时 R=2S"},
        "radius_limits": {"clause": "5.4", "table": "3", "pdf_page": 5, "printed_page": 3},
    },
}


def number(cell: str) -> float:
    return float(cell.replace("△", "").replace(" ", "").replace("\n", ""))


def h_sections(document) -> list[dict]:
    table = document[16].find_tables().tables[0].extract()
    records = []
    designation = ""
    for row in table[2:]:
        designation = row[0] or designation
        values = [number(value) for value in row[1:6]]
        records.append({
            "family": "h", "designation": designation,
            "sectionParams": dict(zip(["height", "widthB", "thicknessT1", "thicknessT2", "radius"], values)),
            "area_cm2": number(row[6]), "mass_kg_m": number(row[7]),
            "Ix_cm4": number(row[9]), "Iy_cm4": number(row[10]),
            "source": {"standard": "GB/T 11263-2024", "table": "A.1", "pdf_page": 17, "printed_page": 11},
        })
    return records


def channel_sections(document) -> list[dict]:
    records = []
    for pdf_page, table_index in [(10, 1), (11, 0)]:
        rows = document[pdf_page - 1].find_tables().tables[table_index].extract()
        previous = [None] * 6
        for row in rows[2:]:
            # Height, flange thickness and radii use vertically merged cells.
            cells = [cell if cell is not None else previous[i] for i, cell in enumerate(row[1:7])]
            previous = cells
            records.append({
                "family": "channel", "designation": row[0],
                "sectionParams": dict(zip(
                    ["height", "legWidthB", "thicknessD", "thicknessT", "radiusR", "legArcRadiusR1"],
                    [number(cell) for cell in cells],
                )),
                "area_cm2": number(row[7]), "mass_kg_m": number(row[8]),
                "Ix_cm4": number(row[10]), "Iy_cm4": number(row[11]),
                "centroid_from_back_mm": number(row[17]) * 10,
                "source": {"standard": "GB/T 706-2016", "table": "A.2", "pdf_page": pdf_page, "printed_page": pdf_page - 2},
            })
    return records


def square_sections(document, sizes: set) -> list[dict]:
    rows = document[13].find_tables().tables[0].extract()
    listed = {}
    side = None
    for row in rows[3:]:
        if not row[1] or not row[1][0].isdigit():
            break
        side = number(row[0]) if row[0] else side
        listed[(side, number(row[1]))] = {
            "area_cm2": number(row[2]), "mass_kg_m": number(row[3]), "Ix_cm4": number(row[4]),
        }
    records = []
    for side, thickness in sorted(sizes):
        reference = (1.5 if thickness <= 6 else 2) * thickness
        limit = (2 if thickness <= 6 else 2.5 if thickness <= 10 else 3) * thickness
        records.append({
            "family": "box", "designation": f"{side:g}x{side:g}x{thickness:g}",
            "sectionParams": {"sideLengthA": side, "sideLengthB": side, "wallThicknessT": thickness},
            "reference_radius_mm": reference, "radius_upper_bound_mm": limit,
            "reference_area_cm2": 2 * thickness * (2 * side - (2.8584 if thickness <= 6 else 3.2876) * thickness) / 100,
            "listed_dimensions": (side, thickness) in listed,
            "listed_properties": listed.get((side, thickness)),
            "source": {"standard": "GB/T 3094-2012", "table": "A.1 footnote", "pdf_page": 14, "printed_page": 12},
        })
    return records


def required_sizes(libraries: list[dict]) -> tuple[set, set, set]:
    h_sizes, channels, squares = set(), set(), set()
    for library in libraries:
        for key in ["H型钢-主梁", "H型钢-次梁"]:
            for row in library[key]:
                h_sizes.add(tuple(row[f"截面公称尺寸{field} mm"] for field in ["H", "B", "t", "l"]))
        channels.update(str(row["型号"]) for row in library["槽钢"])
        squares.update((row["边长B mm"], row["壁厚t mm"]) for row in library["方钢"])
    return h_sizes, channels, squares


def write_report(records: list[dict]) -> Path:
    lines = [
        "# 截面标准参数核对", "",
        "范围：steel_sections.json 与 steel_sections_large.json 实际加载的 H 型钢、槽钢、方钢管。尺寸单位均为 mm。",
        "参数单独用于 BIM 导出，未修改原始截面库、有限元计算参数和既有对比结果。", "",
        "## 来源与限定", "",
        "- H 型钢：GB/T 11263—2024 表 A.1，PDF 第 17 页、正文第 11 页。6.1.2 条（PDF 第 9 页）说明圆角半径为参考值。",
        "- 槽钢：GB/T 706—2016 表 A.2，PDF 第 10—11 页、正文第 8—9 页。表尾注（PDF 第 12 页）说明 r、r1 用于孔型设计，不做交货条件。",
        "- 方钢管：GB/T 3094—2012 表 A.1 注，PDF 第 14 页、正文第 12 页，给出理论重量计算的参考圆角；5.4 条表 3（PDF 第 5 页）给出允许上限。参考值和上限都不是厂家实测圆角。",
        "- 原 JSON 中的 C型钢 列表未被 SectionLibrary 加载，不把它自动视作热轧槽钢。", "",
        "## H 型钢", "",
        "两库共匹配 21 种实际尺寸。原 JSON 的 t、l 分别对应标准 t1（腹板厚）、t2（翼缘厚）；圆角均为 8 mm。",
        "平台映射为 hShapedSteelGirder/Pillar，不使用 gongShapedSteel 类型。", "",
        "| 名义型号 | 实际 H | B | t1 | t2 | r |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        if row["family"] == "h":
            values = [row["sectionParams"][key] for key in ["height", "widthB", "thicknessT1", "thicknessT2", "radius"]]
            lines.append("| " + row["designation"] + " | " + " | ".join(f"{value:g}" for value in values) + " |")
    lines.extend([
        "", "## 槽钢", "",
        "工程库匹配 17 种，研究库匹配 20 种；合计 20 种型号，包含 24a、24b、24c。",
        "BIM 使用下表的 t、r、r1。当前分析库未保存 t，仍沿用原有 tf=tw 回退；该差异在导出记录中逐项标注。", "",
        "| 型号 | h | b | d | t | r | r1 | PDF 页 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in records:
        if row["family"] == "channel":
            values = [row["sectionParams"][key] for key in ["height", "legWidthB", "thicknessD", "thicknessT", "radiusR", "legArcRadiusR1"]]
            lines.append("| " + row["designation"] + " | " + " | ".join(f"{value:g}" for value in values) + f" | {row['source']['pdf_page']} |")
    lines.extend([
        "", "## 方钢管", "",
        "S≤6 时，理论重量计算参考 R=1.5S，允许上限 R≤2S；6<S≤10 时参考 R=2S，上限 R≤2.5S；S>10 时参考 R=2S，上限 R≤3S。",
        "工程库 4 个规格、研究库 20 个规格，合计 23 种尺寸。150×150×4/5/6 未列入表 A.1，应按 5.1 条与供货方确认；BIM 采用参考圆角须显式开启 use_box_reference_radius。", "",
        "| 尺寸 A×A×S | 参考 R | R 上限 | 表 A.1 是否列出 |",
        "|---|---:|---:|---|",
    ])
    for row in records:
        if row["family"] == "box":
            lines.append(f"| {row['designation']} | {row['reference_radius_mm']:g} | {row['radius_upper_bound_mm']:g} | {'是' if row['listed_dimensions'] else '否'} |")
    lines.extend([
        "", "## 当前默认导出模型", "",
        "- BOX-150x6：150×150×6；理论计算参考外圆角 9，允许上限 12；是否采用此 BIM 参考值、制造工艺及供货实际圆角待确认。",
        "- H-150x100.0：H=150、B=100、t1=6、t2=10、r=8。",
        "- H-125x60.0：H=125、B=60、t1=6、t2=8、r=8。",
        "- C-200x75x9.0：对应 20b，h=200、b=75、d=9、t=11、r=11、r1=5.5。分析模型现有 tf=9，未改动。", "",
        "方管原库与 GB/T 3094 的参考截面特性不完全一致：150×150×6 原面积为 33.633 cm²，公式(A.1)参考面积约 33.942 cm²；150×150×8 原面积 43.242 cm²，表列为 43.79 cm²。不得直接覆盖原分析库。", "",
        "## 仍待确认", "",
        "方管产品是否按该冷拔标准供货、未列尺寸的供货约定及参考圆角采用策略、平台截面零旋转与有限元局部轴的对应、槽钢开口朝向及形心定位。不能把参数表匹配完成视为整个 BIM 舱体已验收。", "",
    ])
    path = ROOT / "results/webcad/section_standards/section_parameters_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    data_dir = ROOT / "src/data"
    paths = [data_dir / name for name in ["steel_sections.json", "steel_sections_large.json"]]
    libraries = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    h_sizes, channel_names, square_sizes = required_sizes(libraries)
    with pymupdf.open(ROOT / "docs/型钢规范" / STANDARDS["GB/T 11263-2024"]["file"]) as document:
        hs = h_sections(document)
    with pymupdf.open(ROOT / "docs/型钢规范" / STANDARDS["GB/T 706-2016"]["file"]) as document:
        channels = channel_sections(document)
    with pymupdf.open(ROOT / "docs/型钢规范" / STANDARDS["GB/T 3094-2012"]["file"]) as document:
        squares = square_sections(document, square_sizes)
    hs = [r for r in hs if tuple(r["sectionParams"][k] for k in ["height", "widthB", "thicknessT1", "thicknessT2"]) in h_sizes]
    channels = [r for r in channels if r["designation"] in channel_names]
    if len(hs) != len(h_sizes) or len(channels) != len(channel_names):
        raise ValueError("Some library sections do not match the extracted standard tables")
    result = {
        "units": "mm", "standards": STANDARDS,
        "scope": [path.name for path in paths],
        "notes": [
            "BIM geometry supplement only; analysis section libraries are unchanged.",
            "H sections are H-shaped steel, not tapered-flange I sections.",
            "GB/T 3094 square-tube radii are theoretical-weight reference values, not measured or uniquely specified product radii.",
            "Square-tube library properties differ from GB/T 3094 reference properties; do not overwrite analysis data.",
        ],
        "sections": hs + channels + squares,
    }
    output = data_dir / "steel_section_geometry.json"
    write_json(output, result)
    print(f"Extracted {len(hs)} H sections, {len(channels)} channels and {len(squares)} square-tube reference records: {output}")
    print(f"Report: {write_report(result['sections'])}")


if __name__ == "__main__":
    main()
