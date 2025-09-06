# 船舶动力学建模与NMPC控制系统

[English](README.md) | [中文](README_CN.md)

## 🚢 项目概述

本项目是一个完整的船舶动力学建模与非线性模型预测控制（NMPC）系统，支持三种不同复杂度的船舶动力学模型。系统提供了从数据处理、模型辨识、验证到NMPC控制的完整工作流程，具有统一的数据格式和简化的可视化功能。

## ✨ 主要特性

- 🔧 **三种船舶动力学模型**：支持不同复杂度的船舶建模需求
- 📊 **统一数据格式**：标准化的数据输入输出格式
- 🎯 **NMPC控制器**：先进的非线性模型预测控制算法
- 📈 **简化可视化**：关键参数的清晰图表展示
- 🔄 **模块化设计**：易于扩展和维护的代码结构
- ⚙️ **配置管理**：统一的配置文件管理系统
- 🖥️ **命令行界面**：支持批处理和交互式操作
- 🔍 **多种滤波方法**：Savitzky-Golay、扩展卡尔曼滤波等

## 🏗️ 系统架构

### 核心模块

- **统一模型接口** (`model_interface.py`)：支持三种模型的统一调用
- **通用NMPC控制器** (`nmpc_tracking/universal_nmpc_controller.py`)：统一的控制接口
- **数据格式标准** (`src/utils/data_format.py`)：标准化数据处理
- **简化可视化** (`src/utils/simplified_visualizer.py`)：关键参数可视化
- **配置管理** (`config_manager.py`)：统一配置管理
- **参数辨识工具** (`model_identifier.py`)：命令行参数辨识工具

### 三种船舶模型

1. **模型1**：标准模型，适用于基础应用
2. **模型2**：分离推进器输入模型，平衡精度与计算效率
3. **模型3**：简化模型，适用于快速计算场景

## 🔧 安装依赖

```bash
pip install numpy scipy pandas matplotlib casadi control
```

## 📁 项目结构

```
MI/
├── config/                     # 配置文件目录
│   ├── model1_config.json      # 模型1配置
│   ├── model2_config.json      # 模型2配置
│   └── model3_config.json      # 模型3配置
├── datas/                      # 数据文件目录
│   ├── boat1_2_circle.xlsx     # 圆形轨迹数据
│   └── boat1_2_sin.xlsx        # 正弦轨迹数据
├── examples/                   # 示例代码目录
│   ├── model1_complete_demo.py # 模型1完整演示
│   ├── model2_complete_demo.py # 模型2完整演示
│   ├── model3_complete_demo.py # 模型3完整演示
│   └── usage_examples.md       # 使用示例文档
├── nmpc_tracking/              # NMPC控制器模块
│   ├── boat1_2_atwnmpc.py      # 原始NMPC控制器示例
│   ├── identified_model_nmpc_test.py # 识别模型NMPC测试（支持传参）
│   └── universal_nmpc_controller.py # 通用NMPC控制器
├── src/                        # 源代码目录
│   ├── data_processing/        # 数据处理模块
│   ├── model_identification/   # 模型辨识模块
│   ├── simulation_visualization/ # 仿真与可视化模块
│   └── utils/                  # 工具模块
│       ├── data_format.py      # 统一数据格式
│       └── simplified_visualizer.py # 简化可视化
├── config_manager.py           # 配置管理器
├── model_interface.py          # 统一模型接口
├── model_identifier.py         # 参数辨识工具
├── test_nmpc_parameters.py     # NMPC传参验证批量测试脚本
├── NMPC_参数传递使用说明.md    # NMPC传参功能详细说明
├── README.md                   # 项目文档（英文）
└── README_CN.md                # 项目文档（中文）
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 依赖包：numpy, scipy, pandas, matplotlib, casadi, control

### 安装依赖

```bash
pip install numpy scipy pandas matplotlib casadi control
```

## 📖 参数辨识工具使用指南

### 基本命令行使用

运行参数辨识脚本，支持多种模型类型、数据文件和滤波方法选择：

```bash
# 基本使用（使用默认参数）
python model_identifier.py --model model_1
python model_identifier.py --model model_2
python model_identifier.py --model model_3

# 指定数据文件和滤波方法
python model_identifier.py --model model_1 --data datas/boat1_2_circle.xlsx --filter savgol
python model_identifier.py --model model_2 --data datas/boat1_2_sin.xlsx --filter ekf
python model_identifier.py --model model_3 --filter lowpass

# 自定义数据范围和输出目录
python model_identifier.py --model model_1 --start_row 100 --row_count 1000 --output_dir results
python model_identifier.py --model model_1 --data datas/boat1_2_sin.xlsx --filter savgol --optimizer trust-constr --start_row 0 --row_count 500 --output_dir test_results
# 启用交互式模式（推荐新用户使用）
python model_identifier.py --interactive
```

### 命令行参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | int | 1 | 模型类型：1=标准模型, 2=分离推进器输入模型, 3=简化模型 |
| `--data` | str | datas/boat1_2_sin.xlsx | 数据文件路径 |
| `--filter` | str | savgol | 滤波方法：savgol/ekf/lowpass/none |
| `--interactive` | flag | False | 启用交互式模式 |
| `--start_row` | int | 0 | 数据起始行 |
| `--row_count` | int | 1500 | 读取数据行数 |
| `--output_dir` | str | . | 输出文件目录 |

### 滤波方法说明

- **savgol**: Savitzky-Golay平滑滤波（推荐，适用于大多数情况）
- **ekf**: 扩展卡尔曼滤波（适用于噪声较大的数据）
- **lowpass**: 低通滤波（适用于高频噪声明显的数据）
- **none**: 无滤波（适用于已预处理的干净数据）

### 交互式模式

对于新用户，推荐使用交互式模式，系统会引导您完成所有选择：

```bash
python model_identifier.py --interactive
```

交互式模式将提供：
- 模型类型选择菜单
- 自动检测可用数据文件
- 滤波方法选择指导
- 数据范围设置
- 实时参数验证

### 工作流程

参数辨识的完整工作流程包括：

1. **数据读取**：从Excel文件中读取实验数据
2. **数据预处理**：应用选定的滤波方法进行数据处理
3. **参数优化**：使用优化算法辨识模型参数
4. **模型验证**：计算RMSE等性能指标
5. **结果可视化**：生成参数辨识和性能分析图表
6. **数据导出**：保存参数、结果数据和元数据


## 📊 输出文件说明

### 参数辨识工具输出

运行参数辨识后，系统会生成以下文件：

1. **参数文件**: `model_{type}_params.json`
   - 辨识得到的模型参数
   - 优化结果信息
==注意：参数文件保存在`model_results/`目录下，里面的参数请不要超过10，超过则自动消减为0.的小数==
2. **结果数据**: `model_{type}_identification_results.csv`
   - 时间序列数据
   - 实际值与仿真值对比

3. **元数据**: `model_{type}_identification_metadata.json`
   - 处理信息和配置
   - 性能指标

4. **可视化图表**: 
   - `model_{type}_identification_results.png`: 参数辨识结果
   - `model_{type}_performance_analysis.png`: 性能分析

### 性能指标

系统计算以下RMSE（均方根误差）指标：
- u方向（纵荡速度）
- v方向（横荡速度） 
- r方向（艏摇角速度）

## 🔧 API 使用说明


### NMPC轨迹跟踪控制传参验证

系统支持通过命令行参数进行NMPC轨迹跟踪控制验证，提供灵活的测试选项：

```bash
# 基本语法
python nmpc_tracking/identified_model_nmpc_test.py [选项]

# 参数说明
--model [1|2|3]      # 模型类别选择
--trajectory [1|2|3] # 跟踪曲线选择
--adaptive           # 启用自适应NMPC控制
```

#### 支持的模型类别
- **Model 1**: 基础模型 (18参数)
- **Model 2**: 分离模型 (21参数)
- **Model 3**: 简化模型 (16参数)
==注意：参数文件保存在`model_results/`目录下，里面的参数请不要超过10，超过则自动消减为0.的小数==
#### 支持的跟踪曲线
- **轨迹1**: 椭圆轨迹 `x = 40*sin(t) + 1, y = 30*cos(t) + 1`
- **轨迹2**: 正弦直线轨迹 `x = 40*sin(t) + 1, y = t`
- **轨迹3**: 双正弦轨迹 `x = 40*sin(t) + 1, y = 30*sin(0.5*t) + 1`

#### 使用示例

```bash
# Model 1 + 椭圆轨迹 + 自适应控制
python nmpc_tracking/identified_model_nmpc_test.py --model 1 --trajectory 1 --adaptive

# Model 2 + 正弦轨迹 + 非自适应
python nmpc_tracking/identified_model_nmpc_test.py --model 2 --trajectory 2

# Model 3 + 双正弦轨迹 + 自适应控制
python nmpc_tracking/identified_model_nmpc_test.py --model 3 --trajectory 3 --adaptive


```

#### 输出结果
- **控制台输出**: 实时显示仿真进度和性能统计
- **CSV文件**: `nmpc_identified_model_results.csv` - 详细仿真数据
- **性能报告**: `nmpc_performance_report.txt` - 包含配置信息和性能指标

详细使用说明请参考：[NMPC参数传递使用说明](NMPC_参数传递使用说明.md)

## 🎯 模型选择指南

| 模型 | 复杂度 | 精度 | 计算速度 | 适用场景 |
|------|--------|------|----------|----------|
| 模型1 | 低 | 中等 | 快 | 快速原型、实时控制 |
| 模型2 | 中等 | 高 | 中等 | 平衡精度与效率 |
| 模型3 | 高 | 最高 | 慢 | 高精度仿真、离线分析 |

## 🔍 故障排除

### 常见问题

1. **CasADi导入错误**
   ```bash
   pip install casadi
   ```

2. **数值溢出警告**
   - 检查模型参数是否合理
   - 调整NMPC控制器参数
   - 减小仿真步长

3. **参数辨识失败**
   - 检查数据文件路径
   - 确认数据格式正确
   - 调整优化器参数

4. **绘图显示问题**
   - 确保安装了matplotlib
   - 检查字体设置
   - 验证数据完整性

### 性能优化建议

- 对于实时应用，推荐使用模型1
- 调整NMPC预测时域以平衡性能和计算速度
- 使用并行计算加速参数辨识过程
- 选择合适的滤波方法以提高数据质量

## 📚 配置文件说明

每个模型都有对应的配置文件，位于 `config/` 目录：

- `model1_config.json`：模型1的参数配置
- `model2_config.json`：模型2的参数配置  
- `model3_config.json`：模型3的参数配置

配置文件包含：
- 模型参数初值
- NMPC控制器参数
- 仿真设置
- 数据处理参数

## 📊 数据格式说明

系统使用统一的数据格式，包含以下标准列：

- **时间列**：`time` - 仿真时间
- **状态变量**：`u`, `v`, `r`, `x`, `y`, `psi` - 船舶状态
- **控制输入**：`Ts`, `Tp` - 推进器控制量
- **参考信号**：`u_ref`, `v_ref`, `r_ref`, `x_ref`, `y_ref`, `psi_ref`
- **跟踪误差**：`error_x`, `error_y`, `error_psi`

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 项目Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 邮箱: your-email@example.com

---

**注意**: 本系统仅用于研究和教育目的，实际应用时请根据具体需求进行适当修改和验证。

## 🔮 未来工作

- 实现更多辨识算法，如遗传算法、粒子群优化等
- 添加模型验证功能，使用交叉验证评估模型性能
- 开发图形用户界面，提高系统易用性
- 实现实时数据处理和在线辨识功能
- 支持更多船舶模型类型
- 集成机器学习方法进行参数辨识

## 📖 相关文档

- [使用示例](examples/usage_examples.md) - 详细的使用示例和最佳实践
- [NMPC参数传递使用说明](NMPC_参数传递使用说明.md) - NMPC轨迹跟踪控制传参验证详细说明
- [English README](README.md) - English version of this document