import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

from src.utils.constants import CM2_TO_M2, CM4_TO_M4, MM_TO_M


# --- 基础配置 ---
class SectionType(Enum):
    BOX = "方钢"
    H_MAIN = "H型钢-主梁"
    H_SEC = "H型钢-次梁"
    CHANNEL = "C型钢"  # 对应JSON中的 "C型钢" 或 "槽钢"


@dataclass
class SectionProperties:
    """通用截面属性基类 (SI 单位: m, kg)"""

    idx: int
    name: str
    area: float  # m²
    Ix: float  # m⁴ (绕强轴)
    Iy: float  # m⁴ (绕弱轴)
    Iz: float  # m⁴ (扭转常数 J)
    density: float = 7850.0  # kg/m³


@dataclass
class BoxSection(SectionProperties):
    B: float = 0.0
    t: float = 0.0


@dataclass
class HShapeSection(SectionProperties):
    H: float = 0.0
    B: float = 0.0
    tw: float = 0.0
    tf: float = 0.0


@dataclass
class ChannelSection(SectionProperties):
    H: float = 0.0
    B: float = 0.0
    tw: float = 0.0
    # C型钢/槽钢惯性矩计算较为特殊，可在初始化时校验或计算


class SectionLibrary:
    """重构后的截面库"""

    def __init__(self, file_path: str = "src/data/steel_sections.json"):
        self._sections: Dict[SectionType, List[SectionProperties]] = {t: [] for t in SectionType}
        self._load_from_json(file_path)

    @property
    def section_counts(self) -> Dict[SectionType, int]:
        """返回每种截面的数量"""
        return {s_type: len(secs) for s_type, secs in self._sections.items()}

    def get_section(self, s_type: SectionType, index: int) -> SectionProperties:
        """安全获取截面，越界则返回默认或抛出异常"""
        return self._sections[s_type][index]

    def _load_from_json(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 1. 加载方钢 (Box)
        for i, item in enumerate(data.get("方钢", [])):
            # 转换单位 mm -> m, cm² -> m²
            b = item["边长B mm"] * 0.001
            t = item["壁厚t mm"] * 0.001
            self._sections[SectionType.BOX].append(
                BoxSection(
                    idx=i,
                    name=f"BOX-{int(item['边长B mm'])}x{int(item['壁厚t mm'])}",
                    area=item["截面面积A cm²"] * CM2_TO_M2,
                    Ix=item["惯性矩Ix=Iy cm⁴"] * CM4_TO_M4,
                    Iy=item["惯性矩Ix=Iy cm⁴"] * CM4_TO_M4,
                    Iz=item["扭转常数I cm"] * CM4_TO_M4,
                    B=b,
                    t=t,
                )
            )

        # 2. 加载 H 型钢 (合并主梁和次梁逻辑)
        self._parse_h_list(data.get("H型钢-主梁", []), SectionType.H_MAIN)
        self._parse_h_list(data.get("H型钢-次梁", []), SectionType.H_SEC)

        # 3. 加载 C 型钢 / 槽钢
        # 优先读取 "C型钢"，如果没有则读取 "槽钢"
        c_list = data.get("槽钢", [])

        for i, item in enumerate(c_list):
            h, b, d = item["h"], item["b"], item["d"]
            area, ix, iy = item["截面面积"], item["Ix"], item["Iy"]

            self._sections[SectionType.CHANNEL].append(
                ChannelSection(
                    idx=i,
                    name=f"C-{h}x{b}x{d}",
                    area=area * CM2_TO_M2,
                    Ix=ix * CM4_TO_M4,
                    Iy=iy * CM4_TO_M4,
                    Iz=iy * CM4_TO_M4,  # 近似
                    H=h * MM_TO_M,
                    B=b * MM_TO_M,
                    tw=d * MM_TO_M,
                )
            )

        # 排序
        for key in self._sections:
            self._sections[key].sort(key=lambda x: x.area)

    def _parse_h_list(self, raw_list, target_type):
        for i, item in enumerate(raw_list):
            try:
                self._sections[target_type].append(
                    HShapeSection(
                        idx=i,
                        name=f"H-{item['截面公称尺寸H mm']}x{item['截面公称尺寸B mm']}",
                        area=item["截面面积cm²"] * CM2_TO_M2,
                        Ix=item["惯性矩Ix cm⁴"] * CM4_TO_M4,
                        Iy=item["惯性矩Iy cm⁴"] * CM4_TO_M4,
                        Iz=item["惯性矩Iy cm⁴"] * CM4_TO_M4,  # 近似
                        H=item["截面公称尺寸H mm"] * MM_TO_M,
                        B=item["截面公称尺寸B mm"] * MM_TO_M,
                        tw=item["截面公称尺寸t mm"] * MM_TO_M,
                        tf=item["截面公称尺寸l mm"] * MM_TO_M,
                    )
                )
            except:
                continue

    def _sort_sections_by_Ix(self):
        """按照Ix排序"""
        for section_type in self._sections:
            self._sections[section_type].sort(key=lambda x: x.Ix)
            # reindex
            for i, section in enumerate(self._sections[section_type]):
                section.idx = i

    def _sort_sections_by_area(self):
        """按照面积排序"""
        for section_type in self._sections:
            self._sections[section_type].sort(key=lambda x: x.area)
            # reindex
            for i, section in enumerate(self._sections[section_type]):
                section.idx = i


# Example Usage (optional, for testing)
if __name__ == "__main__":
    library = SectionLibrary(file_path="src/data/steel_sections.json")  # Adjust path for standalone run

    print("Loaded 方钢 sections:")
    for section_type in library.sections:
        print(f"Loaded {len(library.sections[section_type])} {section_type} sections:")
        for sec in library.sections[section_type]:
            print(sec)
        print("-" * 100)

    # Test get_section
    frame_section = library.get_section("方钢", 0)
    if frame_section:
        print(f"Selected frame section: {frame_section.name}")

    main_beam_section = library.get_section("H型钢-主梁", 0)  # Assuming H型钢 has at least one entry
    if main_beam_section:
        print(f"Selected main beam section: {main_beam_section.name}")
