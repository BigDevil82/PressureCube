"""常量定义"""

# 材料属性
STEEL_E = 2.1e11  # 钢材弹性模量 (Pa)
STEEL_G = 8.1e10  # 钢材剪切模量 (Pa)
STEEL_DENSITY = 7850  # 钢材密度 (kg/m^3)
STEEL_YIELD = 235e6  # 钢材屈服强度 (Pa)

# 约束条件
MAX_STRESS_RATIO = 0.8  # 最大应力比
MAX_DISPLACEMENT_RATIO = 1 / 250  # 最大位移比
MAX_DRIFT_RATIO = 1 / 200  # 最大层间位移角

# 截面类型
SECTION_TYPES = {
    "frame_beam": "□150×6",  # 框架梁
    "main_beam": "HW150×150",  # 环向主梁
    "secondary_beam": "HN100×50",  # 环向次梁
    "brace": "C10",  # 支撑梁
}

# 荷载条件
INTERNAL_PRESSURE = 30e3  # 内压 (Pa)
DEAD_LOAD = 5000  # 自重 (N/m^2)
LIVE_LOAD = 1800  # 活载 (N/m^2)

# 尺寸限制
MAX_LENGTH = 14.0  # 最大长度 (m)
MAX_WIDTH = 3.7  # 最大宽度 (m)
MAX_HEIGHT = 3.3  # 最大高度 (m)

# 间距限制
MIN_BEAM_SPACING = 0.5  # 最小梁间距 (m)
MAX_BEAM_SPACING = 2.0  # 最大梁间距 (m)
MIN_BRACE_SPACING = 1.0  # 最小支撑间距 (m)
MAX_BRACE_SPACING = 3.0  # 最大支撑间距 (m)


# Conversion factors
MM_TO_M = 0.001
CM_TO_M = 0.01
CM2_TO_M2 = CM_TO_M**2
CM3_TO_M3 = CM_TO_M**3
CM4_TO_M4 = CM_TO_M**4
MM4_TO_M4 = MM_TO_M**4
KG_M_TO_N_M3 = 9.80665  # Assuming g = 9.80665 m/s^2 for density calculation from linear weight
