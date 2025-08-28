# 钢框架结构遗传算法优化

本模块实现了使用遗传算法优化钢框架结构的截面尺寸和构件间距，以在满足结构约束的条件下最小化材料用量。

## 功能特性

### 优化目标
- **主要目标**: 材料用量最小化（体积最小）
- **约束条件**: 
  - 最大应力 ≤ 355 MPa
  - 最大位移 ≤ 14.4 mm

### 设计变量
1. **截面索引** (4个离散变量):
   - 框架梁截面索引：方钢截面库中的索引 (0-3)
   - 主梁截面索引：H型钢-主梁截面库中的索引 (0-12)
   - 次梁截面索引：H型钢-次梁截面库中的索引 (0-9)
   - 支撑梁截面索引：C型钢截面库中的索引 (0-19)

2. **间距参数** (2个离散变量):
   - 主梁间距：[0.3, 1.8] m，步长 0.05 m
   - 支撑梁间距：[0.3, 1.0] m，步长 0.05 m

### 遗传算法特性
- **选择策略**: 轮盘赌选择
- **交叉策略**: 截面索引使用单点交叉，间距参数使用算术交叉
- **变异策略**: 随机选择基因进行变异
- **精英保留**: 保留每代最优个体
- **约束处理**: 惩罚函数法处理不可行解

## 文件结构

```
src/
├── steel_frame_ga.py              # 遗传算法核心实现
├── run_steel_frame_optimization.py # 主运行程序
├── test_ga.py                     # 测试脚本
└── README_GA.md                   # 说明文档
```

## 快速开始

### 1. 基本使用

```python
from src.steel_frame_ga import SteelFrameGA, GAParameters

# 创建遗传算法参数
ga_params = GAParameters(
    population_size=50,
    max_generations=100,
    crossover_rate=0.8,
    mutation_rate=0.1,
    elite_ratio=0.1
)

# 创建优化器
optimizer = SteelFrameGA(config_path="config.yaml", ga_params=ga_params)

# 运行优化
best_solution = optimizer.run_optimization()

# 查看结果
print(f"最优适应度: {best_solution.fitness}")
print(f"材料用量: {best_solution.material_volume} m³")
print(f"最大应力: {best_solution.max_stress} MPa")
print(f"最大位移: {best_solution.max_displacement} mm")
```

### 2. 运行完整优化程序

```bash
cd src
python run_steel_frame_optimization.py
```

程序提供三种运行模式：
1. **单次优化**: 使用默认参数运行一次优化
2. **多参数对比**: 使用不同参数设置运行多次优化进行对比
3. **退出程序**

### 3. 运行测试

```bash
cd src
python test_ga.py
```

测试程序会验证：
- 染色体创建和操作
- 种群初始化
- 遗传操作（交叉、变异）
- 单个个体评估
- 小规模优化运行

## 核心类和方法

### 1. Chromosome 类
表示遗传算法中的个体（染色体）

```python
class Chromosome:
    def __init__(self, frame_idx, main_idx, secondary_idx, brace_idx, 
                 main_spacing, brace_spacing):
        # 截面索引
        self.frame_section_index = frame_idx
        self.main_beam_section_index = main_idx
        self.secondary_beam_section_index = secondary_idx
        self.brace_section_index = brace_idx
        
        # 间距参数
        self.main_beam_spacing = main_spacing
        self.brace_spacing = brace_spacing
        
        # 评估结果
        self.fitness = 0.0
        self.material_volume = 0.0
        self.max_stress = 0.0
        self.max_displacement = 0.0
        self.is_feasible = False
```

### 2. SteelFrameGA 类
遗传算法主类

主要方法：
- `initialize_population()`: 初始化种群
- `evaluate_chromosome()`: 评估个体适应度
- `roulette_wheel_selection()`: 轮盘赌选择
- `crossover()`: 交叉操作
- `mutate()`: 变异操作
- `run_optimization()`: 运行完整优化过程

### 3. 参数配置类

```python
@dataclass
class GAParameters:
    population_size: int = 50      # 种群大小
    max_generations: int = 100     # 最大代数
    crossover_rate: float = 0.8    # 交叉率
    mutation_rate: float = 0.1     # 变异率
    elite_ratio: float = 0.1       # 精英比例

@dataclass
class DesignConstraints:
    max_stress: float = 355.0      # 最大应力约束 (MPa)
    max_displacement: float = 14.4 # 最大位移约束 (mm)
```

## 适应度函数

适应度函数采用惩罚函数法处理约束：

```python
if chromosome.is_feasible:
    # 可行解：适应度 = 1 / 材料用量
    fitness = 1.0 / (material_volume + 1e-10)
else:
    # 不可行解：添加惩罚项
    stress_violation = max(0, max_stress - max_stress_limit)
    displacement_violation = max(0, max_displacement - max_displacement_limit)
    penalty = 10 * (stress_violation/max_stress_limit + 
                      displacement_violation/max_displacement_limit)
    fitness = 1.0 / (material_volume + penalty + 1e-10)
```

## 结果输出

优化结果包含：

1. **JSON格式结果文件** (`optimization_results.json`):
   - 遗传算法参数
   - 约束条件
   - 最优解详细信息
   - 优化历史数据

2. **收敛曲线图** (`convergence_curve.png`):
   - 适应度收敛过程
   - 材料用量优化过程

3. **截面详细信息** (`section_details.json`):
   - 各类型梁的具体截面参数
   - 间距参数

## 性能调优建议

### 1. 参数设置建议

**小规模快速测试**:
```python
GAParameters(
    population_size=20,
    max_generations=10,
    crossover_rate=0.8,
    mutation_rate=0.1
)
```

**中等规模优化**:
```python
GAParameters(
    population_size=50,
    max_generations=50,
    crossover_rate=0.8,
    mutation_rate=0.1
)
```

**大规模深度优化**:
```python
GAParameters(
    population_size=100,
    max_generations=100,
    crossover_rate=0.8,
    mutation_rate=0.05
)
```

### 2. 性能优化

- **并行评估**: 可以考虑使用多进程并行评估个体
- **缓存机制**: 对相同参数组合的评估结果进行缓存
- **早停策略**: 当连续多代无改进时提前终止

## 注意事项

1. **OpenSees依赖**: 需要正确安装OpenSees和相关依赖
2. **内存使用**: 大规模优化可能消耗较多内存
3. **计算时间**: 每次结构分析需要一定时间，建议先用小参数测试
4. **收敛性**: 复杂约束问题可能需要更多代数才能收敛

## 扩展功能

可以考虑的扩展方向：

1. **多目标优化**: 同时优化材料用量和结构性能
2. **自适应参数**: 根据优化过程动态调整GA参数
3. **混合算法**: 结合局部搜索提高收敛速度
4. **约束处理**: 更高级的约束处理方法
5. **并行计算**: 利用多核CPU加速评估过程

