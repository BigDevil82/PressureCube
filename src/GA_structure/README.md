# 基于遗传算法的钢结构优化

本项目是基于遗传算法在钢结构优化上的对比尝试，集成了结构建模（OpenSees）、遗传算法（Genetic Algorithm）、参数优化、可视化等模块，自动搜索最优钢结构设计参数，在满足结构安全约束（应力、位移、层间位移角）的前提下，实现结构总质量最小化。

---

## 项目结构

```bash
PressureCube-main/
├── src/
│   ├── GA_structure/                  # 遗传算法主逻辑模块
│   │   ├── main.py                    # 主运行脚本（遗传算法入口）
│   │   ├── Common_components/        # 遗传算法通用组件（个体、种群、适应度等）
│   │   ├── StructuralOptimization/   # 钢结构优化专属模块（交叉、变异、适应度）
│   ├── analysis/opensees_analysis.py # 调用 OpenSees 进行结构分析
│   ├── utils/model_utils.py          # 辅助构建结构模型的工具
│   ├── structure/structural_model.py # 钢结构模型定义
├── config.yaml                        # 配置文件，定义结构固定参数
├── best_individual.json               # 记录最优个体参数
├── structure_optimization_results.csv # 所有参数组合的平均适应度
├── fitness_evolution.png             # 最优适应度随迭代次数变化的图像
```
## 环境依赖
同主项目一样

## 快速开始
```bash
cd src  
python -m GA_structure.main
```

## 运行后生成
- structure_optimization_results.csv：不同参数组合的平均适应度（结构质量）  
- best_individual.json：当前找到的最优结构设计参数  
- fitness_evolution.png：最优适应度随迭代次数变化的图像