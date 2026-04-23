from dataclasses import dataclass, field


@dataclass
class DoorParameters:
    """门洞参数（工程版布局使用）"""

    active: bool = True
    width: float = 0.91
    height: float = 2.21
    bottom_offset: float = 0.3
    face: str = "right"


@dataclass
class WindowParameters:
    """窗洞参数（工程版布局使用）"""

    active: bool = True
    sill_height: float = 0.9
    height: float = 1.5
    front_spans: list[int] = field(default_factory=list)
    back_spans: list[int] = field(default_factory=list)
    right_spans: list[int] = field(default_factory=list)


@dataclass
class EngineeringLayoutParameters:
    """工程版布局超参数"""

    min_edge_remainder: float = 0.1
    support_beam_max_spacing: float = 0.75
    window: WindowParameters = field(default_factory=WindowParameters)
    door: DoorParameters = field(default_factory=DoorParameters)


# ==========================================
# 1. 基础数据结构保持不变
# ==========================================


@dataclass
class ExtensionParameters:
    """侧面凸出部分参数"""

    active: bool = True
    length: float = 1.1  # Y负方向长度
    width: float = 1.85  # X方向宽度
    height: float = 0.0  # 自动计算
    offset_x: float = 0.0  # 自动计算


@dataclass
class StructuralParameters:
    """结构参数类"""

    length: float
    width: float
    height: float
    main_beam_spacing: float
    num_secondary_width: int
    num_secondary_height: int
    add_support_beam: bool = True
    layout_strategy: str = "research"
    engineering: EngineeringLayoutParameters = field(default_factory=EngineeringLayoutParameters)

    # 截面索引
    frame_section_index: int = 0
    main_beam_section_index: int = 0
    secondary_beam_section_index: int = 0
    brace_section_index: int = 0

    extension: ExtensionParameters = field(default_factory=ExtensionParameters)

    # 其他物理参数
    plate_thickness: float = 0.005

    def __post_init__(self):
        if self.extension.active:
            self.extension.width = self.width / 2
            self.extension.height = self.height
            self.extension.offset_x = self.width - self.extension.width
