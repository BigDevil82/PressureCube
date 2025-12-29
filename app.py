import streamlit as st

st.set_page_config(layout="wide", page_title="模块化舱体结构智能设计系统", page_icon="🏗️")

# CSS样式
st.markdown(
    """
    <style>
        .stApp {
            background-color: #ffffff;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 1rem;
        }
        h1 {
            padding-top: 0rem;
            margin-top: 0rem;
        }
        /* 侧边栏导航链接样式优化 */
        [data-testid="stSidebarNav"] li a span{
            font-size: 18px !important;
            font-weight: 600 !important;
            padding: 4px 8px !important;
            margin: 4px 0 !important;
        }
        /* 导航链接悬停效果 */
        [data-testid="stSidebarNav"] li a:hover {
            background-color: #f0f2f6 !important;
            border-radius: 8px !important;
        }
        /* 当前激活页面样式 */
        [data-testid="stSidebarNav"] li a[aria-current="page"] {
            background-color: #e8eaf6 !important;
            border-left: 4px solid #1976d2 !important;
            border-radius: 4px !important;
        }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("🏗️ 模块化舱体结构智能设计系统")

st.markdown(
    """
## 欢迎使用增压建筑智能设计系统

本系统集成了参数化建模、有限元分析和优化设计功能，帮助您快速完成模块化舱体结构的设计与分析。

### 🎯 主要功能

#### 📐 参数化模型
- 交互式几何参数调整
- 实时3D可视化
- 自动生成结构拓扑
- 支持梁板组合结构

#### 🔬 有限元模型  
- 基于 OpenSees 的精确分析
- 支持静力与模态分析
- 多种荷载工况组合
- 详细的应力位移结果
- 自动网格细化

### 📋 使用流程

1. **参数化建模** 👈 从左侧导航栏进入
   - 设置舱体尺寸和拓扑参数
   - 查看3D模型预览
   - 确认结构方案

2. **有限元分析**
   - 设置网格和荷载参数
   - 运行静力或模态分析
   - 查看应力、位移等结果

3. **优化设计** (开发中)
   - 基于遗传算法的结构优化
   - 基于强化学习的智能优化
   - 多目标优化

### 💡 快速开始

点击左侧导航栏中的 **"参数化模型"** 开始您的设计！

""",
    unsafe_allow_html=True,
)

# 在侧边栏显示快速导航
st.sidebar.title("🧭 快速导航")
st.sidebar.info(
    """
使用上方的页面选择器切换不同功能模块：

- 🏠 **首页**: 系统介绍
- 📐 **参数化模型**: 几何建模
- 🔬 **有限元模型**: 结构分析
"""
)

st.sidebar.divider()
st.sidebar.caption("© 2025 增压建筑智能设计系统")
