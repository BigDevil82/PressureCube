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
    tf: float = 0.0  # 新增：为了准确记录，建议增加翼缘厚度字段


class SectionLibrary:
    """重构后的截面库"""

    def __init__(self, file_path: str = "src/data/steel_sections.json"):
        self.file_path = file_path
        self._sections: Dict[SectionType, List[SectionProperties]] = {t: [] for t in SectionType}
        self._load_from_json(file_path)

    @property
    def section_counts(self) -> Dict[SectionType, int]:
        """返回每种截面的数量"""
        return {s_type: len(secs) for s_type, secs in self._sections.items()}

    def get_section(self, s_type: SectionType, index: int) -> SectionProperties:
        """安全获取截面，越界则返回默认或抛出异常"""
        return self._sections[s_type][index]

    def _calc_J_box(self, B: float, t: float) -> float:
        """
        计算方钢(闭口截面)扭转常数 J
        公式: Bredt公式简化 J ≈ t * (B-t)^3
        """
        # 注意：此处假设是方钢。如果是矩形管，公式会有所不同。
        # B: 边长 (m), t: 壁厚 (m)
        return t * (B - t) ** 3

    def _calc_J_open(self, H: float, B: float, tw: float, tf: float) -> float:
        """
        计算 H型钢/槽钢(开口截面)扭转常数 J
        公式: 矩形分割求和法 J ≈ 1/3 * Σ(bi * ti^3)
        J ≈ 1/3 * (2 * B * tf^3 + (H - 2*tf) * tw^3)
        """
        # hw: 腹板净高
        hw = H - 2 * tf
        # 防止出现负数（极端小尺寸校验）
        if hw < 0:
            hw = 0

        return (1.0 / 3.0) * (2 * B * pow(tf, 3) + hw * pow(tw, 3))

    def _load_from_json(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            print(f"Loading section data from {path}...")
            data = json.load(f)

        # 1. 加载方钢 (Box)
        for i, item in enumerate(data.get("方钢", [])):
            # 转换单位 mm -> m, cm² -> m²
            b_val = item["边长B mm"] * MM_TO_M
            t_val = item["壁厚t mm"] * MM_TO_M

            # 计算 J
            j_val = self._calc_J_box(b_val, t_val)

            self._sections[SectionType.BOX].append(
                BoxSection(
                    idx=i,
                    name=f"BOX-{int(item['边长B mm'])}x{int(item['壁厚t mm'])}",
                    area=item["截面面积A cm²"] * CM2_TO_M2,
                    Ix=item["惯性矩Ix=Iy cm⁴"] * CM4_TO_M4,
                    Iy=item["惯性矩Ix=Iy cm⁴"] * CM4_TO_M4,
                    Iz=j_val,  # 使用计算值
                    B=b_val,
                    t=t_val,
                )
            )

        # 2. 加载 H 型钢 (合并主梁和次梁逻辑)
        self._parse_h_list(data.get("H型钢-主梁", []), SectionType.H_MAIN)
        self._parse_h_list(data.get("H型钢-次梁", []), SectionType.H_SEC)

        # 3. 加载 C 型钢 / 槽钢
        c_list = data.get("槽钢", [])

        for i, item in enumerate(c_list):
            h, b, d = item["h"], item["b"], item["d"]
            # 尝试获取翼缘厚度 't'，如果JSON中没有，暂时回退到使用 'd' (腹板厚) 以避免崩溃，
            # 但实际槽钢 t > d，建议JSON数据源补充 't' 字段。
            t_flange_mm = item.get("t", d)

            h_m = h * MM_TO_M
            b_m = b * MM_TO_M
            tw_m = d * MM_TO_M
            tf_m = t_flange_mm * MM_TO_M

            area, ix, iy = item["截面面积"], item["Ix"], item["Iy"]

            # 计算 J
            j_val = self._calc_J_open(H=h_m, B=b_m, tw=tw_m, tf=tf_m)

            self._sections[SectionType.CHANNEL].append(
                ChannelSection(
                    idx=i,
                    name=f"C-{h}x{b}x{d}",
                    area=area * CM2_TO_M2,
                    Ix=ix * CM4_TO_M4,
                    Iy=iy * CM4_TO_M4,
                    Iz=j_val,  # 使用计算值
                    H=h_m,
                    B=b_m,
                    tw=tw_m,
                    tf=tf_m,
                )
            )

        # 排序并重新索引
        # self._sort_sections_by_area()
        self._sort_sections_by_Ix()

    def _parse_h_list(self, raw_list, target_type):
        for i, item in enumerate(raw_list):
            try:
                h_m = item["截面公称尺寸H mm"] * MM_TO_M
                b_m = item["截面公称尺寸B mm"] * MM_TO_M
                tw_m = item["截面公称尺寸t mm"] * MM_TO_M
                tf_m = item["截面公称尺寸l mm"] * MM_TO_M  # 注意：此处沿用原代码逻辑，假设 'l' 对应 tf

                # 计算 J
                j_val = self._calc_J_open(H=h_m, B=b_m, tw=tw_m, tf=tf_m)

                self._sections[target_type].append(
                    HShapeSection(
                        idx=i,
                        name=f"H-{item['截面公称尺寸H mm']}x{item['截面公称尺寸B mm']}",
                        area=item["截面面积cm²"] * CM2_TO_M2,
                        Ix=item["惯性矩Ix cm⁴"] * CM4_TO_M4,
                        Iy=item["惯性矩Iy cm⁴"] * CM4_TO_M4,
                        Iz=j_val,  # 使用计算值
                        H=h_m,
                        B=b_m,
                        tw=tw_m,
                        tf=tf_m,
                    )
                )
            except Exception as e:
                print(f"Error parsing H型钢 entry: {item} -> {e}")

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
    library = SectionLibrary(file_path="src/data/steel_sections_large.json")  # Adjust path for standalone run

    print("Loaded 方钢 sections:")
    for section_type in library._sections:
        print(f"Loaded {len(library._sections[section_type])} {section_type} sections:")
        print("-" * 100)

    # Test get_section
    frame_section = library.get_section(SectionType.BOX, 0)
    if frame_section:
        print(f"Selected frame section: {frame_section.name}")

    main_beam_section = library.get_section(SectionType.H_MAIN, 0)  # Assuming H型钢 has at least one entry
    if main_beam_section:
        print(f"Selected main beam section: {main_beam_section.name}")
