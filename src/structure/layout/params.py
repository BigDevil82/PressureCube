from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ExtensionParameters:
    """侧面凸出部分参数"""

    active: bool = True
    length: float = 1.1  # Y负方向长度
    width: float = 1.85  # X方向宽度
    height: float = 0.0  # 自动计算
    offset_x: float = 0.0  # 自动计算


@dataclass
class DoorParameters:
    """门洞参数（工程版布局使用）"""

    active: bool = True
    width: float = 0.91
    height: float = 2.21
    bottom_offset: float = 0.3
    face: str = "left"
    center_x: float | None = None


@dataclass
class WindowParameters:
    """窗洞参数（工程版布局使用）"""

    active: bool = True
    sill_height: float = 0.9
    height: float = 1.5
    front_spans: list[int] = field(default_factory=list)
    back_spans: list[int] = field(default_factory=list)
    right_spans: list[int] = field(default_factory=lambda: [1, 2])


@dataclass
class OpeningPanelParameters:
    """门窗封板材料参数，默认采用玻璃钢近似。"""

    elastic_modulus_mpa: float = 70000.0
    poisson_ratio: float = 0.23
    density_kg_m3: float = 2500.0
    thickness: float | None = 0.02
    design_strength_mpa: float = 180.0


@dataclass
class BaseLayoutParameters:
    """两套布局方案共享的参数基类"""

    length: float
    width: float
    height: float
    main_beam_spacing: float
    add_support_beam: bool = True

    extension: ExtensionParameters = field(default_factory=ExtensionParameters)

    # 其他物理参数
    plate_thickness: float = 0.005
    opening_panel: OpeningPanelParameters = field(default_factory=OpeningPanelParameters)

    layout_strategy: str = field(init=False, default="base")

    def __post_init__(self):
        if self.extension.active:
            self.extension.width = self.width / 2
            self.extension.height = self.height
            self.extension.offset_x = self.width - self.extension.width
        if self.opening_panel.thickness is None:
            self.opening_panel.thickness = self.plate_thickness


@dataclass
class ResearchLayoutParameters(BaseLayoutParameters):
    """研究版布局参数"""

    num_secondary_width: int = 3
    num_secondary_height: int = 3
    layout_strategy: Literal["research"] = field(init=False, default="research")


@dataclass
class EngineeringLayoutParameters(BaseLayoutParameters):
    """工程版布局参数"""

    min_edge_remainder: float = 0.1
    support_beam_max_spacing: float = 0.75
    window: WindowParameters = field(default_factory=WindowParameters)
    door: DoorParameters = field(default_factory=DoorParameters)
    layout_strategy: Literal["engineering"] = field(init=False, default="engineering")

    def __post_init__(self):
        super().__post_init__()

        self.door.width = self.width / 2 - self.main_beam_spacing


LayoutParameters = ResearchLayoutParameters | EngineeringLayoutParameters

# 向后兼容旧名称。现有优化与分析代码默认仍使用研究版参数。
StructuralParameters = ResearchLayoutParameters
