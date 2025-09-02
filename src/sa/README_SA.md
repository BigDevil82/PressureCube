# 钢框架结构模拟退火算法优化

本模块实现了使用模拟退火算法优化钢框架结构的截面尺寸和构件间距，以在满足结构约束的条件下最小化材料用量。

## 算法特性

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

### 模拟退火算法特性
- **邻域操作**: 智能邻域生成，70%概率选择相邻值，30%概率随机选择
- **冷却策略**: 支持线性、指数、对数、自适应四种冷却策略
- **接受准则**: Metropolis准则
- **重启机制**: 长时间无改进时自动重启搜索
- **约束处理**: 惩罚函数法处理不可行解

## 文件结构

```
src/sa/
├── steel_frame_sa.py              # 模拟退火算法核心实现
├── run_sa_optimization.py         # 主运行程序
├── test_sa.py                     # 测试脚本
└── README_SA.md                   # 说明文档
```

## 快速开始

### 1. 基本使用

```python
from src.sa.steel_frame_sa import SteelFrameSA, SAParameters, CoolingSchedule

# 创建模拟退火参数
sa_params = SAParameters(
    initial_temperature=1000.0,
    final_temperature=0.1,
    max_iterations=3000,
    max_iterations_per_temp=50,
    cooling_schedule=CoolingSchedule.EXPONENTIAL,
    cooling_rate=0.95
)

# 创建优化器
optimizer = SteelFrameSA(config_path="config.yaml", sa_params=sa_params)

# 运行优化
best_solution = optimizer.run_optimization()

# 查看结果
print(f"目标函数值: {best_solution.objective_value} m³")
print(f"最大应力: {best_solution.max_stress} MPa")
print(f"最大位移: {best_solution.max_displacement} mm")
print(f"可行性: {best_solution.is_feasible}")
```

### 2. 运行完整优化程序

```bash
cd src/sa
python run_sa_optimization.py
```

程序提供四种运行模式：
1. **单次优化**: 使用默认参数运行一次优化
2. **多冷却策略对比**: 使用不同冷却策略运行多次优化进行对比
3. **参数敏感性分析**: 分析不同参数对优化效果的影响
4. **退出程序**


## 核心类和方法

### 1. Solution 类
表示模拟退火算法中的解

```python
class Solution:
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
        self.objective_value = float('inf')  # 材料用量
        self.fitness = 0.0  # 适应度（考虑约束）
        self.max_stress = 0.0
        self.max_displacement = 0.0
        self.is_feasible = False
        self.penalty = 0.0
```

### 2. SteelFrameSA 类
模拟退火算法主类

主要方法：
- `create_random_solution()`: 创建随机初始解
- `generate_neighbor()`: 生成邻域解
- `evaluate_solution()`: 评估解的目标函数值和约束
- `update_temperature()`: 更新温度（支持多种冷却策略）
- `accept_solution()`: 接受准则判断
- `run_optimization()`: 运行完整优化过程

### 3. 参数配置类

```python
@dataclass
class SAParameters:
    initial_temperature: float = 1000.0     # 初始温度
    final_temperature: float = 0.1          # 最终温度
    max_iterations: int = 5000              # 最大迭代次数
    max_iterations_per_temp: int = 100      # 每温度最大迭代次数
    cooling_schedule: CoolingSchedule = CoolingSchedule.EXPONENTIAL  # 冷却策略
    cooling_rate: float = 0.95              # 冷却率
    restart_threshold: int = 500            # 重启阈值

@dataclass
class DesignConstraints:
    max_stress: float = 355.0      # 最大应力约束 (MPa)
    max_displacement: float = 14.4 # 最大位移约束 (mm)
```

## 冷却策略

### 1. 线性冷却 (LINEAR)
```
T(k) = T₀ × (1 - k/K)
```
其中 T₀ 是初始温度，k 是当前迭代次数，K 是最大迭代次数。

### 2. 指数冷却 (EXPONENTIAL)
```
T(k) = T₀ × α^k
```
其中 α 是冷却率 (0 < α < 1)。

### 3. 对数冷却 (LOGARITHMIC)
```
T(k) = T₀ / (1 + ln(1 + k))
```

### 4. 自适应冷却 (ADAPTIVE)
根据接受率动态调整冷却速度：
- 接受率 > 50%：加快冷却
- 接受率 < 10%：减慢冷却
- 其他：正常冷却

## 邻域操作

### 智能邻域生成策略
对于每个设计变量：
- **70%概率**: 选择相邻值（±1索引或±1步长）
- **30%概率**: 随机选择新值

这种策略既保证了局部搜索的细致性，又保持了全局搜索的多样性。

### 截面索引邻域
```python
# 相邻索引
new_index = max(0, min(max_index, current_index + delta))
# 随机索引
new_index = random.randint(0, max_index)
```

### 间距参数邻域
```python
# 相邻间距值
current_idx = candidates.index(current_spacing)
new_idx = max(0, min(len(candidates)-1, current_idx + delta))
new_spacing = candidates[new_idx]
```

## 约束处理

采用惩罚函数法处理约束：

```python
if solution.is_feasible:
    fitness = objective_value  # 可行解直接使用目标值
else:
    # 不可行解：添加惩罚项
    stress_violation = max(0, max_stress - max_stress_limit)
    displacement_violation = max(0, max_displacement - max_displacement_limit)
    penalty = 1000 * (stress_violation/max_stress_limit + 
                      displacement_violation/max_displacement_limit)
    fitness = objective_value + penalty
```

## 结果输出

优化结果包含：

1. **JSON格式结果文件** (`sa_optimization_results.json`):
   - 模拟退火参数
   - 约束条件
   - 最优解详细信息
   - 优化统计信息
   - 优化历史数据

2. **优化过程图** (`sa_optimization_process.png`):
   - 温度冷却曲线
   - 目标函数值变化
   - 接受率变化
   - 温度与目标值关系散点图


## 参数调优建议

### 1. 温度参数设置

**快速搜索**:
```python
SAParameters(
    initial_temperature=500.0,
    final_temperature=1.0,
    cooling_rate=0.90
)
```

**精细搜索**:
```python
SAParameters(
    initial_temperature=2000.0,
    final_temperature=0.1,
    cooling_rate=0.98
)
```

**平衡搜索**:
```python
SAParameters(
    initial_temperature=1000.0,
    final_temperature=0.5,
    cooling_rate=0.95
)
```

### 2. 迭代参数设置

- **max_iterations**: 总迭代次数，建议 2000-5000
- **max_iterations_per_temp**: 每温度迭代次数，建议 50-100
- **restart_threshold**: 重启阈值，建议为总迭代次数的 10-20%

### 3. 冷却策略选择

- **指数冷却**: 最常用，收敛稳定
- **线性冷却**: 收敛较快，但可能陷入局部最优
- **对数冷却**: 收敛较慢，但全局搜索能力强
- **自适应冷却**: 根据搜索状态动态调整，适应性强



## 算法比较

### 模拟退火 vs 遗传算法

| 特性 | 模拟退火 | 遗传算法 |
|------|----------|----------|
| 搜索方式 | 单点搜索 | 群体搜索 |
| 内存需求 | 低 | 中等 |
| 参数敏感性 | 中等 | 较高 |
| 局部搜索能力 | 强 | 中等 |
| 全局搜索能力 | 中等 | 强 |
| 收敛速度 | 中等 | 较快 |
| 实现复杂度 | 简单 | 中等 |

### 适用场景
- **模拟退火适用于**: 计算资源有限、需要精细局部搜索的场景
- **遗传算法适用于**: 需要快速找到多个优良解、全局搜索的场景

## 使用示例

### 示例1: 基本优化

```python
from src.sa.steel_frame_sa import SteelFrameSA, SAParameters

# 创建优化器
optimizer = SteelFrameSA()

# 运行优化
best_solution = optimizer.run_optimization()

print(f"最优材料用量: {best_solution.objective_value:.6f} m³")
print(f"设计参数: {best_solution}")
```

### 示例2: 自定义参数优化

```python
# 设置自定义参数
sa_params = SAParameters(
    initial_temperature=2000.0,
    final_temperature=0.01,
    max_iterations=5000,
    cooling_schedule=CoolingSchedule.ADAPTIVE,
    cooling_rate=0.98
)

optimizer = SteelFrameSA(sa_params=sa_params)
best_solution = optimizer.run_optimization()
```

### 示例3: 多策略对比

```python
from src.sa.run_sa_optimization import run_optimization_with_different_cooling_schedules

# 运行多种冷却策略对比
run_optimization_with_different_cooling_schedules()
```

## 性能基准

基于典型钢框架结构（7.2m×3.6m×3.3m）的性能表现：

| 参数设置 | 迭代次数 | 优化时间 | 收敛质量 | 内存使用 |
|----------|----------|----------|----------|----------|
| 快速模式 | 1000 | ~10分钟 | 良好 | 低 |
| 标准模式 | 3000 | ~30分钟 | 很好 | 中等 |
| 精细模式 | 5000 | ~50分钟 | 优秀 | 中等 |

*注：实际性能取决于硬件配置和结构复杂度*

## 最佳实践

### 1. 参数设置策略
1. **初始温度**: 通过试验确定，使初始接受率约为80-90%
2. **最终温度**: 设置为初始温度的0.1-1%
3. **冷却率**: 指数冷却建议0.95-0.99
4. **迭代次数**: 根据问题复杂度和时间预算确定

### 2. 优化流程建议
1. 先用小规模参数验证程序正确性
2. 运行参数敏感性分析确定最佳参数
3. 多次运行取最优结果
4. 验证最优解的工程合理性

### 3. 结果分析
1. 检查收敛曲线判断优化质量
2. 分析接受率变化评估搜索效率
3. 对比不同策略的优化效果
4. 验证约束满足情况

## 注意事项

1. **OpenSees依赖**: 需要正确安装OpenSees和相关依赖
2. **计算时间**: 每次结构分析需要时间，建议合理设置迭代次数
3. **参数敏感性**: 温度参数对算法性能影响较大
4. **随机性**: 多次运行可能得到不同结果，建议多次运行取最优
