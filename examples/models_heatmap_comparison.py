"""
Comparison of Parameters for Three Ship Dynamic Models
Displays the parameter distribution and comparison of Model 1, Model 2, and Model 3,
including parameter values and corresponding state variables
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from matplotlib.colors import LinearSegmentedColormap

# Configure matplotlib for proper LaTeX rendering
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'text.usetex': False,  # Use matplotlib's own LaTeX parser
    'mathtext.fontset': 'dejavusans'  # Use DejaVu Sans for math symbols
})

# Adjust font size for better visibility of parameters and state variables

class ModelParamsVisualizer:
    """模型参数可视化工具类"""
    
    def __init__(self):
        # Model parameters file paths (using relative paths)
        self.model1_params_path = 'model_results/model_1_identification_metadata.json'
        self.model2_params_path = 'model_results/model_2_identification_metadata.json'
        self.model3_params_path = 'model_results/model_3_identification_metadata.json'
        
        # 读取模型参数
        self.model1_params = self._load_params(self.model1_params_path)
        self.model2_params = self._load_params(self.model2_params_path)
        self.model3_params = self._load_params(self.model3_params_path)
        
        # 验证参数数量
        self._validate_params()
        
        # 定义每种模型的参数对应的状态变量组合
        self._define_param_state_combinations()
        
    def _load_params(self, file_path):
        """加载模型参数：从JSON提取`parameters`列表，兼容直接列表"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 支持两种结构：
            # 1) { "parameters": [...] }
            # 2) [ ... ] 直接为列表
            if isinstance(data, dict):
                if 'parameters' in data and isinstance(data['parameters'], list):
                    params_list = data['parameters']
                elif 'params' in data and isinstance(data['params'], list):
                    params_list = data['params']
                else:
                    raise ValueError(f"文件 {file_path} 不包含'parameters'或'params'列表")
            elif isinstance(data, list):
                params_list = data
            else:
                raise ValueError(f"不支持的参数文件格式：{type(data)}")
            return np.array(params_list, dtype=float)
        except Exception as e:
            print(f"加载参数文件 {file_path} 时出错: {e}")
            raise
    
    def _validate_params(self):
        """验证参数数量是否正确"""
        assert len(self.model1_params) == 18, f"模型1参数数量错误: 期望18个，实际{len(self.model1_params)}个"
        assert len(self.model2_params) == 21, f"模型2参数数量错误: 期望21个，实际{len(self.model2_params)}个"
        assert len(self.model3_params) == 16, f"模型3参数数量错误: 期望16个，实际{len(self.model3_params)}个"
    
    def _define_param_state_combinations(self):
        """Define the state variable combinations corresponding to the parameters of each model"""
        # State variable combinations for Model 1 parameters
        self.model1_param_states = {
            'u': ['v*r', 'u', 'v', 'r', '(Tp+Ts)', '1'],
            'v': ['u*r', 'u', 'v', 'r', '(Tp-Ts)', '1'],
            'r': ['u*v', 'u', 'v', 'r', '(Tp-Ts)', '1']
        }
        
        # State variable combinations for Model 2 parameters
        self.model2_param_states = {
            'u': ['v*r', 'u', 'v', 'r', 'Tp', 'Ts', '1'],
            'v': ['u*r', 'u', 'v', 'r', 'Tp', 'Ts', '1'],
            'r': ['u*v', 'u', 'v', 'r', 'Tp', 'Ts', '1']
        }
        
        # State variable combinations for Model 3 parameters
        self.model3_param_states = {
            'u': ['v*r', 'u', 'r', '1', '(Tp+Ts)'],
            'v': ['u*r', 'v', 'r', '1', '(Tp-Ts)'],
            'r': ['u*v', 'u', 'v', 'r', '(Tp-Ts)', '1']
        }
    
    def _prepare_model_data(self):
        """准备模型参数数据，按u、v、r分组"""
        # 模型1参数分组 (18参数)
        # du: a1, a2, a3, a4, a5, a6
        # dv: b1, b2, b3, b4, b5, b6
        # dr: c1, c2, c3, c4, c5, c6
        model1_data = {
            'u': self.model1_params[0:6],
            'v': self.model1_params[6:12],
            'r': self.model1_params[12:18]
        }
        
        # 模型2参数分组 (21参数)
        # du: a1, a2, a3, a4, a5, a6, a7
        # dv: b1, b2, b3, b4, b5, b6, b7
        # dr: c1, c2, c3, c4, c5, c6, c7
        model2_data = {
            'u': self.model2_params[0:7],
            'v': self.model2_params[7:14],
            'r': self.model2_params[14:21]
        }
        
        # 模型3参数分组 (16参数)
        # du: a1, a2, a3, a4, a5
        # dv: b1, b2, b3, b4, b5
        # dr: c1, c2, c3, c4, c5, c6
        model3_data = {
            'u': self.model3_params[0:5],
            'v': self.model3_params[5:10],
            'r': self.model3_params[10:16]
        }
        
        return model1_data, model2_data, model3_data
    
    def _create_annotations(self, model_data, model_type, max_cols):
        """创建热力图的标注文本，包含参数值和状态变量组合"""
        # 获取对应的参数-状态组合
        if model_type == 1:
            param_states = self.model1_param_states
        elif model_type == 2:
            param_states = self.model2_param_states
        elif model_type == 3:
            param_states = self.model3_param_states
        else:
            raise ValueError("model_type must be 1, 2, or 3")
        
        # 创建标注矩阵，使用传入的max_cols确保与数据矩阵形状一致
        annotations = np.empty((3, max_cols), dtype=object)
        annotations.fill('')
        
        # 填充标注文本
        for i, var in enumerate(['u', 'v', 'r']):
            for j in range(len(model_data[var])):
                param_value = model_data[var][j]
                state_var = param_states[var][j]
                # 格式化参数值，根据大小调整小数位数
                if abs(param_value) >= 100:
                    formatted_value = f"{param_value:.1f}"
                elif abs(param_value) >= 10:
                    formatted_value = f"{param_value:.2f}"
                elif abs(param_value) >= 0.1:
                    formatted_value = f"{param_value:.3f}"
                else:
                    formatted_value = f"{param_value:.4f}"
                # 创建标注文本
                annotations[i, j] = f"{formatted_value}*\n{state_var}"
        
        return annotations
    
    def _create_combined_data_matrix(self, model1_data, model2_data, model3_data):
        """创建组合数据矩阵用于热力图绘制"""
        # 确定每个模型的列数
        model1_cols = max(len(model1_data['u']), len(model1_data['v']), len(model1_data['r']))  # 应该是6
        model2_cols = max(len(model2_data['u']), len(model2_data['v']), len(model2_data['r']))  # 应该是7
        model3_cols = max(len(model3_data['u']), len(model3_data['v']), len(model3_data['r']))  # 应该是6
        
        # 创建3个矩阵，每个对应一个模型，使用各自的列数
        model1_matrix = np.full((3, model1_cols), np.nan)
        model2_matrix = np.full((3, model2_cols), np.nan)
        model3_matrix = np.full((3, model3_cols), np.nan)
        
        # 填充数据
        for i, var in enumerate(['u', 'v', 'r']):
            model1_matrix[i, :len(model1_data[var])] = model1_data[var]
            model2_matrix[i, :len(model2_data[var])] = model2_data[var]
            model3_matrix[i, :len(model3_data[var])] = model3_data[var]
        
        # Create annotation matrices, passing the respective cols for each model
        annotations1 = self._create_annotations(model1_data, 1, model1_cols)
        annotations2 = self._create_annotations(model2_data, 2, model2_cols)
        annotations3 = self._create_annotations(model3_data, 3, model3_cols)
        
        return model1_matrix, model2_matrix, model3_matrix, annotations1, annotations2, annotations3, model1_cols, model2_cols, model3_cols
    
    def plot_models_heatmap_comparison(self):
        """绘制三种模型参数的热力图对比"""
        # 准备数据
        model1_data, model2_data, model3_data = self._prepare_model_data()
        model1_matrix, model2_matrix, model3_matrix, annotations1, annotations2, annotations3, model1_cols, model2_cols, model3_cols = self._create_combined_data_matrix(
            model1_data, model2_data, model3_data
        )
        
        # 创建掩码处理NaN值
        mask1 = np.isnan(model1_matrix)
        mask2 = np.isnan(model2_matrix)
        mask3 = np.isnan(model3_matrix)
        
        # 创建自定义颜色映射
        colors = [(0.9, 0.2, 0.2), (1, 1, 1), (0.2, 0.6, 0.9)]  # 红-白-蓝
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
        
        # 创建图形和子图，增大尺寸以便更好地显示参数和状态变量
        fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=True)
        fig.subplots_adjust(wspace=0.05)
        
        # 绘制每个模型的热力图
        vmin = min(np.nanmin(model1_matrix), np.nanmin(model2_matrix), np.nanmin(model3_matrix))
        vmax = max(np.nanmax(model1_matrix), np.nanmax(model2_matrix), np.nanmax(model3_matrix))
        
        # 模型1热力图 - 只显示6个参数列
        sns.heatmap(model1_matrix, ax=axes[0],
                    mask=mask1,
                    annot=annotations1,
                    fmt='',  # No automatic formatting, use custom annotations
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    cbar=False,
                    xticklabels=[f'$\\theta_{{{i+1}}}$' for i in range(model1_cols)],
                    yticklabels=[r'$\dot{u}$', r'$\dot{v}$', r'$\dot{r}$'])
        axes[0].set_title('Model 1 (18 params)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Param Index', fontsize=12)
        axes[0].set_ylabel('Derivatives', fontsize=12)
        
        # 模型2热力图 - 显示7个参数列
        sns.heatmap(model2_matrix, ax=axes[1],
                    mask=mask2,
                    annot=annotations2,
                    fmt='',  # 不使用自动格式化，使用自定义标注
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    cbar=False,
                    xticklabels=[f'$\\theta_{{{i+1}}}$' for i in range(model2_cols)],
                    yticklabels=[])
        axes[1].set_title('Model 2 (21 params)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Param Index', fontsize=12)
        
        # 模型3热力图 - 只显示6个参数列
        im = sns.heatmap(model3_matrix, ax=axes[2],
                        mask=mask3,
                        annot=annotations3,
                        fmt='',  # 不使用自动格式化，使用自定义标注
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        cbar_kws={'label': 'Parameter Value'},
                        xticklabels=[f'$\theta_{{{i+1}}}$' for i in range(model3_cols)],
                        yticklabels=[])
        axes[2].set_title('Model 3 (16 params)', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Param Index', fontsize=12)

        # 确保y轴标签正确显示 - 先设置ticks位置，再设置标签
        axes[0].set_yticks([0, 1, 2])
        axes[0].set_yticklabels([r'$\dot{u}$', r'$\dot{v}$', r'$\dot{r}$'], fontsize=12)
        axes[0].tick_params(axis='y', rotation=0)
        axes[0].yaxis.set_label_position('left')
        axes[0].yaxis.set_ticks_position('left')
        
        # Add overall title
        # fig.suptitle('Comparison of Parameters for Three Ship Dynamic Models', fontsize=18, fontweight='bold', y=0.98)
        
        # Add model equations explanation
        models_info = [
            "Model 1 (Basic Model):",
            "du = a1*v*r + a2*u + a3*v + a4*r + a5*(Tp+Ts) + a6",
            "dv = b1*u*r + b2*u + b3*v + b4*r + b5*(Tp-Ts) + b6",
            "dr = c1*u*v + c2*u + c3*v + c4*r + c5*(Tp-Ts) + c6",
            "",
            "Model 2 (Separated Model):",
            "du = a1*v*r + a2*u + a3*v + a4*r + a5*Tp + a6*Ts + a7",
            "dv = b1*u*r + b2*u + b3*v + b4*r + b5*Tp + b6*Ts + b7",
            "dr = c1*u*v + c2*u + c3*v + c4*r + c5*Tp + c6*Ts + c7",
            "",
            "Model 3 (Simplified Model):",
            "du = a1*v*r + a2*u + a3*r + a4 + a5*(Tp+Ts)",
            "dv = b1*u*r + b2*v + b3*r + b4 + b5*(Tp-Ts)",
            "dr = c1*u*v + c2*u + c3*v + c4*r + c5*(Tp-Ts) + c6"
        ]
        
        # Create text box to display model equations
        # plt.figtext(0.5, 0.01, '\n'.join(models_info), ha='center', fontsize=9, 
        #             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=1'))
        
        # 调整布局
        plt.tight_layout(rect=[0, 0.15, 1, 0.95])
        filename = "model_results/models_heatmap_comparison.png"
        # Save figure
        plt.savefig(filename, dpi=600, bbox_inches='tight')
        print("Generated files:")
        print(f"- {filename}")
        plt.show()

    def plot_single_model_heatmap(self, model_type=1):
        """Plot heatmap for a single model, including parameter values and corresponding state variables"""
        # Prepare data
        model1_data, model2_data, model3_data = self._prepare_model_data()
        
        if model_type == 1:
            data = model1_data
            filename = "model_results/model_1_params_heatmap.png"
        elif model_type == 2:
            data = model2_data
            filename = "model_results/model_2_params_heatmap.png"
        elif model_type == 3:
            data = model3_data
            filename = "model_results/model_3_params_heatmap.png"
        else:
            raise ValueError("model_type must be 1, 2, or 3")
        
        # 确定最大列数
        max_cols = max(len(data['u']), len(data['v']), len(data['r']))
        
        # 创建数据矩阵 - 添加参数处理：将百、十位数降为个位数（仅在单个热力图中应用）
        def process_parameter(value):
            """将百位数或十位数降为个位数，保留符号"""
            if abs(value) >= 100:
                return value / 100  # 将百位数降为个位数
            elif abs(value) >= 10:
                return value / 10   # 将十位数降为个位数
            return value           # 个位数保持不变
        
        # 创建处理后的数据字典
        processed_data_dict = {}
        matrix = np.full((3, max_cols), np.nan)
        for i, var in enumerate(['u', 'v', 'r']):
            processed_data = [process_parameter(val) for val in data[var]]
            processed_data_dict[var] = processed_data
            matrix[i, :len(processed_data)] = processed_data
        
        # 创建掩码处理NaN值
        mask = np.isnan(matrix)
        
        # Create annotation matrix using processed data
        annotations = self._create_annotations(processed_data_dict, model_type, max_cols)
        
        # 创建自定义颜色映射
        colors = [(0.9, 0.2, 0.2), (1, 1, 1), (0.2, 0.6, 0.9)]  # 红-白-蓝
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
        
        # 绘制热力图，增大尺寸以便更好地显示参数和状态变量
        plt.figure(figsize=(16, 8))
        
        sns.heatmap(matrix, 
                    mask=mask,
                    annot=annotations,
                    fmt='',  # 不使用自动格式化，使用自定义标注
                    cmap=cmap,
                    cbar_kws={'label': 'Parameter Value'},
                    xticklabels=[f'$\theta_{{{i+1}}}$' for i in range(max_cols)],
                    yticklabels=[r'$\mathbf{\dot{u}}$', r'$\mathbf{\dot{v}}$', r'$\mathbf{\dot{r}}$ '])
        
        # plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Param Index', fontsize=12)
        plt.ylabel('Derivatives', fontsize=12)
        
        # 调整布局
        plt.tight_layout()
        
        # Save figure
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print("Generated files:")
        print(f"- {filename}")
        plt.show()

def main():
    """Main function"""
    print("=== Three Ship Dynamic Models Parameter Heatmap Comparison Tool ===")
    
    # Create visualizer instance
    visualizer = ModelParamsVisualizer()
    
    # Plot comparison heatmap of three models
    print("\nPlotting comparison heatmap of three models...")
    # visualizer.plot_models_heatmap_comparison()
    
    # You can also plot each model's heatmap individually
    # visualizer.plot_single_model_heatmap(model_type=1)
    visualizer.plot_single_model_heatmap(model_type=2)
    # visualizer.plot_single_model_heatmap(model_type=3)

if __name__ == "__main__":
    main()