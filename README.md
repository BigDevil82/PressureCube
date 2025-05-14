# 高海拔增压建筑智能设计系统

基于层次图机器学习的三维钢框架优化设计系统，用于高海拔增压建筑的结构优化设计。

## 功能特点

- 基于层次图嵌入(HEBG)的结构表示
- 强化学习驱动的结构优化
- OpenSees结构分析集成
- 自动约束校核
- 多目标优化（质量最小化、满足强度刚度要求）

## 系统要求

- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric 2.0+
- OpenSees 3.3.0+
- 其他依赖见 requirements.txt

## 安装

1. 克隆仓库：
```bash
git clone [repository_url]
cd [repository_name]
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 准备配置文件：
修改 `config.yaml` 中的参数，包括舱体尺寸、窗户位置等。

2. 可视化结构模型：
```bash
python main.py --mode visualize --config config.yaml
```

3. 训练优化智能体：
```bash
python main.py --mode train --config config.yaml
```

4. 测试优化结果：
```bash
python main.py --mode test --config config.yaml
```

## 项目结构

```
.
├── src/
│   ├── structure/         # 结构建模
│   │   └── structural_model.py  # 参数化结构模型
│   ├── graph/             # 图神经网络
│   │   └── hebg.py        # 层次图嵌入模型
│   ├── reinforcement/     # 强化学习
│   │   └── rl_agent.py    # DDPG智能体
│   ├── analysis/          # 结构分析
│   │   └── opensees_analysis.py  # OpenSees接口
│   └── utils/             # 工具函数
│       └── constants.py   # 常量定义
├── tests/                 # 测试文件
│   └── test_structural_model.py  # 结构模型测试
├── data/                  # 数据文件
├── main.py                # 主程序
├── config.yaml            # 配置文件
└── requirements.txt       # 项目依赖
```

## 当前进展

- [x] 完成结构建模模块基本框架
  - [x] 参数化生成三维舱体框架
  - [x] 支持窗户位置的动态适配
  - [x] 支持截面属性的管理和更新
- [ ] 实现HEBG层次图嵌入
  - [ ] 完成池化和反池化操作
  - [ ] 实现结构层次表示
- [ ] 开发强化学习环境
  - [ ] 定义状态空间和动作空间
  - [ ] 实现奖励函数
- [ ] 集成OpenSees分析
  - [ ] 实现荷载生成
  - [ ] 完成约束校核
- [ ] 优化算法性能
  - [ ] 实现并行计算
  - [ ] 提高训练效率

## 下一步计划

1. 完善HEBG层次图嵌入模块，实现特定构件的池化和反池化操作
2. 设计强化学习环境和奖励函数，确保有效优化
3. 完成OpenSees分析接口，实现准确的结构分析和约束校核
4. 开展智能体训练，并评估优化效果

## 贡献

欢迎提交Issue和Pull Request。

## 许可证

MIT License