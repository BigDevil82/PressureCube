import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

# Conversion factors
MM_TO_M = 0.001
CM_TO_M = 0.01
CM2_TO_M2 = CM_TO_M * CM_TO_M
CM3_TO_M3 = CM_TO_M * CM_TO_M * CM_TO_M
CM4_TO_M4 = CM_TO_M * CM_TO_M * CM_TO_M * CM_TO_M
MM4_TO_M4 = MM_TO_M * MM_TO_M * MM_TO_M * MM_TO_M
KG_M_TO_N_M3 = 9.80665  # Assuming g = 9.80665 m/s^2 for density calculation from linear weight


@dataclass
class SteelSection:
    """基本截面属性"""

    idx: int  # 截面在该类截面库中的索引
    name: str
    area: float  # 截面面积 (m^2)
    Ix: float  # 惯性矩 (m^4)
    Iy: float
    Iz: float  # 扭转常数 J (m^4) , H型钢和方钢用Iz表示，槽钢用I表示
    Zx: float  # 截面模量 (m^3)
    Zy: float
    Zz: float  # 扭转模量 Wt (m^3), H型钢和方钢用Zz表示
    density: float  # 材料密度 (kg/m^3) - Removed default, will be set during instantiation

    def __repr__(self) -> str:
        return f"{self.name}: Area={self.area:.2e} m^2, Ix={self.Ix:.2e} m^4, Iy={self.Iy:.2e} m^4, Iz={self.Iz:.2e} m^4"


@dataclass
class RectangularTubeSection(SteelSection):
    """方钢管截面属性"""

    B: float  # 边长 (m)
    t: float  # 壁厚 (m)

    def __repr__(self) -> str:
        return f"{self.name}: B={self.B*1000:.1f}mm, t={self.t*1000:.1f}mm, A={self.area:.2e}m², Ix={self.Ix:.2e}m⁴, Iy={self.Iy:.2e}m⁴, Iz={self.Iz:.2e}m⁴"


@dataclass
class HSection(SteelSection):
    """H型钢截面属性"""

    H: float  # 高度 (m)
    B: float  # 宽度 (m)
    t_web: float  # 腹板厚度 (m) - JSON中用 t
    t_flange: float  # 翼缘厚度 (m) - JSON中用 l

    def __repr__(self) -> str:
        return f"{self.name}: H={self.H*1000:.1f}mm, B={self.B*1000:.1f}mm, t_web={self.t_web*1000:.1f}mm, t_flange={self.t_flange*1000:.1f}mm, A={self.area:.2e}m², Ix={self.Ix:.2e}m⁴, Iy={self.Iy:.2e}m⁴, Iz={self.Iz:.2e}m⁴"


@dataclass
class CSection(SteelSection):
    """槽钢截面属性"""

    H: float  # 高度 (m)
    B: float  # 宽度 (m)
    t_web: float  # 腹板厚度 (m) -  没有直接提供，但可以从规格中提取或估算

    def __repr__(self) -> str:
        return f"{self.name}: H={self.H*1000:.1f}mm, B={self.B*1000:.1f}mm, t_web={self.t_web*1000:.1f}mm, A={self.area:.2e}m², Ix={self.Ix:.2e}m⁴, Iy={self.Iy:.2e}m⁴, Iz={self.Iz:.2e}m⁴"


class SectionLibrary:
    """截面库类"""

    def __init__(self, file_path: str = "src/data/steel_sections.json"):
        self.sections: Dict[str, List[Union[RectangularTubeSection, HSection, CSection]]] = {
            "方钢": [],
            "H型钢-主梁": [],
            "H型钢-次梁": [],
            "C型钢": [],
        }
        self._load_sections(file_path)

    @property
    def section_types(self) -> List[str]:
        return list(self.sections.keys())

    @property
    def section_counts(self) -> Dict[str, int]:
        return {section_type: len(self.sections[section_type]) for section_type in self.section_types}

    def _calculate_c_section_I(self, H: float, B: float, t: float) -> Tuple[float, float]:
        """
        计算C型钢的惯性矩
        Ix=I腹板+2I翼缘
        －腹板的惯性矩（矩形，高 $ h-2 t_{f} $ ，宽 $ t_{w} $ ）：
        $$
        I_{\text {腹板 }}=\frac{t_{w} \cdot\left(h-2 t_{f}\right)^{3}}{12}
        $$
        －单个翼缘的惯性矩（矩形，长 $ b $ ，宽 $ t_{f} $ ），需用平行轴定理计算其对 x 轴的贡献：
        $$
        I_{\text {翼缘 }}=\frac{b \cdot t_{f}^{3}}{12}+b \cdot t_{f} \cdot\left(\frac{h-t_{f}}{2}\right)^{2}
        $$

        Iy=2·I翼缘+I腹板
        －单个翼缘的惯性矩（矩形，长 $ t_{f} $ ，宽 $ b $ ）：
        $$
        I_{\text {翼缘 }}=\frac{t_{f} \cdot b^{3}}{12}
        $$
        －腹板的惯性矩（矩形，高 $ h $ ，宽 $ t_{w} $ ，但通常贡献很小）：
        $$
        I_{\text {腹板 }}=\frac{\left(h-2 t_{f}\right) \cdot t_{w}^{3}}{12}
        $$
        """
        # 计算Ix
        # 腹板的惯性矩
        Ix_web = (t * (H - 2 * t) ** 3) / 12
        # 单个翼缘的惯性矩(包含平行轴定理)
        Ix_flange = (B * t**3) / 12 + B * t * ((H - t) / 2) ** 2
        # 总Ix = 腹板 + 2*翼缘
        Ix = Ix_web + 2 * Ix_flange

        # 计算Iy
        # 单个翼缘的惯性矩
        Iy_flange = (t * B**3) / 12
        # 腹板的惯性矩
        Iy_web = ((H - 2 * t) * t**3) / 12
        # 总Iy = 2*翼缘 + 腹板
        Iy = 2 * Iy_flange + Iy_web

        return Ix, Iy

    def _load_sections(self, file_path: str):
        """从JSON文件加载截面数据"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Section file not found at {file_path}")
            return
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {file_path}")
            return

        # 方钢
        for i, item in enumerate(data.get("方钢", [])):
            try:
                name = f"方钢-{item['边长B mm']}x{item['壁厚t mm']}"
                area = item["截面面积A cm²"] * CM2_TO_M2
                Ix = item["惯性矩Ix=Iy cm⁴"] * CM4_TO_M4
                Iy = item["惯性矩Ix=Iy cm⁴"] * CM4_TO_M4  # Assuming square section Ix = Iy
                Iz = item["扭转常数I cm"] * CM4_TO_M4  # 扭转惯性矩
                Zx = item["截面模数Wx=Wy cm³"] * CM3_TO_M3
                Zy = item["截面模数Wx=Wy cm³"] * CM3_TO_M3  # Assuming square section Zx = Zy
                Zz = item["扭转常数C cm³"] * CM3_TO_M3  # 扭转抵抗矩
                density = 7850.0  # Default steel density

                section = RectangularTubeSection(
                    idx=i,
                    name=name,
                    area=area,
                    Ix=Ix,
                    Iy=Iy,
                    Iz=Iz,
                    Zx=Zx,
                    Zy=Zy,
                    Zz=Zz,
                    density=density,
                    B=item["边长B mm"] * MM_TO_M,
                    t=item["壁厚t mm"] * MM_TO_M,
                )
                self.sections["方钢"].append(section)
            except KeyError as e:
                print(f"方钢 section data missing key: {e} in item: {item}")
            except Exception as e:
                print(f"Error processing 方钢 section {item.get('name', 'Unknown')}: {e}")

        # H型钢
        H_sections = data["H型钢-主梁"] + data["H型钢-次梁"]
        for i, item in enumerate(H_sections):
            try:
                name = item["H-型号(高度×宽度) mm×mm"]
                area = item["截面面积cm²"] * CM2_TO_M2
                Ix = item["惯性矩Ix cm⁴"] * CM4_TO_M4
                Iy = item["惯性矩Iy cm⁴"] * CM4_TO_M4
                Iz = Iy  # Placeholder or requires specific calculation if critical
                Zx = item["截面模数Wx cm³"] * CM3_TO_M3
                Zy = item["截面模数Wy cm³"] * CM3_TO_M3
                Zz = Zy  # Placeholder or requires specific calculation for torsional modulus

                density = 7850.0

                section = HSection(
                    idx=i,
                    name=name,
                    area=area,
                    Ix=Ix,
                    Iy=Iy,
                    Iz=Iz,  # Placeholder for torsional constant
                    Zx=Zx,
                    Zy=Zy,
                    Zz=Zz,  # Placeholder for torsional section modulus
                    density=density,
                    H=item["截面公称尺寸H mm"] * MM_TO_M,
                    B=item["截面公称尺寸B mm"] * MM_TO_M,
                    t_web=item["截面公称尺寸t mm"] * MM_TO_M,  # Assuming 't' is t_web
                    t_flange=item["截面公称尺寸l mm"] * MM_TO_M,  # Assuming 'l' is t_flange
                )

                if item["梁规格"] == "环向次梁":
                    self.sections["H型钢-次梁"].append(section)
                elif item["梁规格"] == "环向主梁":
                    self.sections["H型钢-主梁"].append(section)
                else:
                    raise ValueError(f"Unknown梁规格: {item['梁规格']}")

            except KeyError as e:
                print(f"H型钢 section data missing key: {e} in item: {item}")
            except Exception as e:
                print(f"Error processing H型钢 section {item.get('H-型号(高度×宽度) mm×mm', 'Unknown')}: {e}")

        # C型钢
        for i, item in enumerate(data.get("C型钢", [])):
            try:
                # C型钢的JSON结构比较简单，很多参数需要推断或从规格中解析
                # "规格": "80×40×20×2.5" -> H x B x C x t
                spec_parts = item["规格"].split("×")
                H_val = float(spec_parts[0])
                B_val = float(spec_parts[1])
                t_val = float(spec_parts[3])  # thickness

                name = f"C型钢-{item['规格']}"
                linear_weight = item["重量(kg/m)"]
                estimated_area = linear_weight / 7850.0  # kg/m / (kg/m^3) = m^2

                Ix, Iy = self._calculate_c_section_I(H_val, B_val, t_val)
                Ix_placeholder = Ix * MM4_TO_M4
                Iy_placeholder = Iy * MM4_TO_M4
                Iz_placeholder = Iy_placeholder  # Even rougher placeholder for torsion
                Zx_placeholder = Ix_placeholder / (H_val * MM_TO_M / 2)
                Zy_placeholder = Iy_placeholder / (B_val * MM_TO_M / 2)
                Zz_placeholder = Zy_placeholder  # Rough placeholder for torsion

                section = CSection(
                    idx=i,
                    name=name,
                    area=estimated_area,
                    Ix=Ix_placeholder,
                    Iy=Iy_placeholder,
                    Iz=Iz_placeholder,
                    Zx=Zx_placeholder,
                    Zy=Zy_placeholder,
                    Zz=Zz_placeholder,
                    density=7850.0,
                    H=H_val * MM_TO_M,
                    B=B_val * MM_TO_M,
                    t_web=(
                        float(spec_parts[3]) * MM_TO_M if len(spec_parts) > 3 else 0.0025
                    ),  # Assuming last part is thickness
                )
                self.sections["C型钢"].append(section)
            except KeyError as e:
                print(f"C型钢 section data missing key: {e} in item: {item}")
            except Exception as e:
                print(f"Error processing C型钢 section {item.get('规格', 'Unknown')}: {e}")

        self._sort_sections_by_Ix()

    def _sort_sections_by_Ix(self):
        """按照Ix排序"""
        for section_type in self.sections:
            self.sections[section_type].sort(key=lambda x: x.Ix)
            # reindex
            for i, section in enumerate(self.sections[section_type]):
                section.idx = i

    def get_section(
        self, section_type: str, index: int
    ) -> Union[RectangularTubeSection, HSection, CSection, None]:
        """获取指定类型和索引的截面"""
        if section_type in self.sections and 0 <= index < len(self.sections[section_type]):
            return self.sections[section_type][index]
        print(f"Warning: Section type '{section_type}' at index {index} not found.")
        return None


# Example Usage (optional, for testing)
if __name__ == "__main__":
    library = SectionLibrary(file_path="src/data/steel_sections.json")  # Adjust path for standalone run

    print("Loaded 方钢 sections:")
    for sec in library.sections["方钢"]:
        print(
            f"  {sec.name}: Area={sec.area:.2e} m^2, Ix={sec.Ix:.2e} m^4, Iy={sec.Iy:.2e} m^4, Iz={sec.Iz:.2e} m^4, Zx={sec.Zx:.2e} m^3, Zy={sec.Zy:.2e} m^3, Zz={sec.Zz:.2e} m^3"
        )

    print("Loaded H型钢 sections:")
    for sec in library.sections["H型钢-主梁"]:
        print(
            f"  {sec.name}: Area={sec.area:.2e} m^2, Ix={sec.Ix:.2e} m^4, Iy={sec.Iy:.2e} m^4, Iz={sec.Iz:.2e} m^4, Zx={sec.Zx:.2e} m^3, Zy={sec.Zy:.2e} m^3, Zz={sec.Zz:.2e} m^3"
        )

    print("Loaded H型钢-次梁 sections:")
    for sec in library.sections["H型钢-次梁"]:
        print(
            f"  {sec.name}: Area={sec.area:.2e} m^2, Ix={sec.Ix:.2e} m^4, Iy={sec.Iy:.2e} m^4, Iz={sec.Iz:.2e} m^4, Zx={sec.Zx:.2e} m^3, Zy={sec.Zy:.2e} m^3, Zz={sec.Zz:.2e} m^3"
        )

    print("Loaded C型钢 sections:")
    for sec in library.sections["C型钢"]:
        print(
            f"  {sec.name}: Area={sec.area:.2e} m^2, Ix={sec.Ix:.2e} m^4, Iy={sec.Iy:.2e} m^4, Iz={sec.Iz:.2e} m^4, Zx={sec.Zx:.2e} m^3, Zy={sec.Zy:.2e} m^3, Zz={sec.Zz:.2e} m^3"
        )

    # Test get_section
    frame_section = library.get_section("方钢", 0)
    if frame_section:
        print(f"Selected frame section: {frame_section.name}")

    main_beam_section = library.get_section("H型钢-主梁", 0)  # Assuming H型钢 has at least one entry
    if main_beam_section:
        print(f"Selected main beam section: {main_beam_section.name}")
