import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

class ModelTrackingComparison:
    def __init__(self):
        # 设置字体支持 - 移除可能不存在的Helvetica字体
        plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        plt.rcParams['font.size'] = 10  # 设置全局字体大小
        
        # 文件路径
        self.model_1_file = "model_results/model_1_results.csv"
        self.model_2_file = "model_results/model_2_results.csv"
        self.model_3_file = "model_results/model_3_results.csv"
        # 使用Excel作为原始数据来源
        self.raw_data_file = "datas/boat1_2_sin.xlsx"
        
        # 加载数据
        self.model_1_data = self._load_data(self.model_1_file)
        self.model_2_data = self._load_data(self.model_2_file)
        self.model_3_data = self._load_data(self.model_3_file)
        
        # 加载原始数据
        self.raw_data = self._load_raw_data(self.raw_data_file)
        
        # 数据范围
        self.data_range = "0-1500"
        
        # 模型名称映射
        self.model_names = {
            "model_1": "Model 1 (Basic Model)",
            "model_2": "Model 2 (Improved Model)",
            "model_3": "Model 3 (Complex Model)"
        }
        
        # 模型颜色映射
        self.model_colors = {
            "model_1": "#1f77b4",  # 蓝色
            "model_2": "#ff7f0e",  # 橙色
            "model_3": "#2ca02c"   # 绿色
        }
        
        # 准备数据
        self._prepare_data()
        
    def _load_data(self, file_path):
        """加载CSV数据文件"""
        try:
            data = pd.read_csv(file_path)
            # 限制数据范围为0-1500行
            data = data.iloc[0:1501].copy()
            return data
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {e}")
            return pd.DataFrame()
            
    def _load_raw_data(self, file_path):
        """加载原始CSV数据文件（使用相对路径）"""
        try:
            # 直接使用相对路径加载原始数据
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                data = pd.read_excel(file_path)
            else:
                raise ValueError("不支持的文件格式。请使用CSV或Excel文件。")
            # 限制数据范围为0-1500行，与模型数据保持一致
            data = data.iloc[0:1501].copy()
            
            # 重命名小写的x和y列为大写的X和Y，以匹配模型数据的格式
            if 'x' in data.columns and 'y' in data.columns:
                data.rename(columns={'x': 'X', 'y': 'Y'}, inplace=True)
                print(f"成功加载并处理原始数据: {file_path} (将x/y列重命名为X/Y)")
            else:
                print(f"成功加载原始数据: {file_path} (未找到x/y列)")
                
            return data
        except Exception as e:
            print(f"加载原始数据文件时出错: {e}")
            return pd.DataFrame()
    
    def _prepare_data(self):
        """准备数据用于可视化"""
        # 确保所有模型数据非空
        if not self.model_1_data.empty and not self.model_2_data.empty and not self.model_3_data.empty:
            # 计算路径长度（用于累计距离图）
            for model_name, model_data in [
                ("model_1", self.model_1_data),
                ("model_2", self.model_2_data),
                ("model_3", self.model_3_data)
            ]:
                # 计算相邻点之间的距离
                dx = model_data['X'].diff().fillna(0)
                dy = model_data['Y'].diff().fillna(0)
                distance = np.sqrt(dx**2 + dy**2)
                # 计算累计距离
                cumulative_distance = distance.cumsum()
                # 存储到模型数据中
                model_data['Cumulative_Distance'] = cumulative_distance
        else:
            print("警告: 某些模型数据为空，可视化可能不完整")
    
    def plot_trajectory_comparison(self):
        """绘制轨迹跟踪对比图"""
        if self.model_1_data.empty or self.model_2_data.empty or self.model_3_data.empty:
            print("无法绘制轨迹对比图，数据不完整")
            return
        
        # 创建一个2x2的子图布局
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        # fig.suptitle(f'Three Ship Dynamic Models Trajectory Tracking Comparison (Data Range: {self.data_range})', fontsize=16)
        
        # 1. 位置轨迹(北东地坐标系)
        ax1 = axes[0, 0]
        
        # 设置每隔 n 个点绘制一个船图标，增加间隔以减少密度
        n = 200  # 从50增加到100
        
        # 首先绘制原始轨迹（如果有）
        if not self.raw_data.empty and 'X' in self.raw_data.columns and 'Y' in self.raw_data.columns:
            # 北东地坐标系：X代表North，Y代表East
            ax1.plot(self.raw_data['Y'], self.raw_data['X'], 
                     label='Original Trajectory', 
                     color='black', 
                     linewidth=2.0, 
                     linestyle='--')
                       
            # 真实轨迹上的船图标（如果有psi列）
            if 'psi' in self.raw_data.columns:
                # 不缩放坐标值，直接使用原始坐标
                Q_measured = ax1.quiver(
                    self.raw_data['Y'][::n], self.raw_data['X'][::n], 
                    np.sin(self.raw_data['psi'][::n]), np.cos(self.raw_data['psi'][::n]),
                    scale=20, color='#000000', width=0.008
                )
                # 创建自定义图例项
                ax1.quiverkey(Q_measured, X=0.07, Y=0.65, U=1,
                            label='Measured USV', labelpos='E', color='#000000')
        else:
            print("警告: 无法绘制原始轨迹，数据不完整或格式不符")
            
        # 绘制各模型轨迹
        for model_key, model_data in [
            ("model_1", self.model_1_data),
            ("model_2", self.model_2_data),
            ("model_3", self.model_3_data)
        ]:
            # 北东地坐标系：X代表North，Y代表East
            ax1.plot(model_data['Y'], model_data['X'], 
                     label=self.model_names[model_key], 
                     color=self.model_colors[model_key], 
                     linewidth=1.5)
            
            # 模拟轨迹上的船图标（如果有psi列）
            if 'psi' in model_data.columns:
                # 不缩放坐标值，直接使用原始坐标，并为不同模型设置不同的微小偏移量避免重叠
                offset_x, offset_y = 0, 0
                if model_key == "model_1":
                    offset_x, offset_y = 0.5, 0.5
                elif model_key == "model_2":
                    offset_x, offset_y = 0, 0
                elif model_key == "model_3":
                    offset_x, offset_y = -0.5, -0.5
                
                Q_simulated = ax1.quiver(
                    model_data['Y'][::n] + offset_x, model_data['X'][::n] + offset_y, 
                    np.sin(model_data['psi'][::n]), np.cos(model_data['psi'][::n]),
                    scale=20, color=self.model_colors[model_key], width=0.008
                )
        
        # 标记起始点和终点
        for model_key, model_data in [
            ("model_1", self.model_1_data),
            ("model_2", self.model_2_data),
            ("model_3", self.model_3_data)
        ]:
            ax1.plot(model_data['Y'].iloc[0], model_data['X'].iloc[0], 
                     'o', markersize=6, color=self.model_colors[model_key])
            ax1.plot(model_data['Y'].iloc[-1], model_data['X'].iloc[-1], 
                     's', markersize=6, color=self.model_colors[model_key])
        
        ax1.set_title('Ship Trajectory Comparison (North-East-Down)', fontsize=12)
        ax1.set_xlabel('East (m)', fontsize=10)
        ax1.set_ylabel('North (m)', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(fontsize=10)
        
        # 2. 航向角对比
        ax2 = axes[0, 1]
        for model_key, model_data in [
            ("model_1", self.model_1_data),
            ("model_2", self.model_2_data),
            ("model_3", self.model_3_data)
        ]:
            ax2.plot(model_data['Time'], model_data['psi'],  
                     label=self.model_names[model_key], 
                     color=self.model_colors[model_key], 
                     linewidth=1.5)
        
        ax2.set_title('Heading Angle Comparison (psi)', fontsize=12)
        ax2.set_xlabel('Time (s)', fontsize=10)
        ax2.set_ylabel('Heading Angle (rad)', fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(fontsize=10)
        
        # 3. 横向误差对比
        ax3 = axes[1, 0]
        for model_key, model_data in [
            ("model_1", self.model_1_data),
            ("model_2", self.model_2_data),
            ("model_3", self.model_3_data)
        ]:
            ax3.plot(model_data['Time'], model_data['Lateral_Error'], 
                     label=self.model_names[model_key], 
                     color=self.model_colors[model_key], 
                     linewidth=1.5)
        
        ax3.set_title('Lateral Error Comparison', fontsize=12)
        ax3.set_xlabel('Time (s)', fontsize=10)
        ax3.set_ylabel('Lateral Error (m)', fontsize=10)
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend(fontsize=10)
        
        # 4. 航向误差对比
        ax4 = axes[1, 1]
        for model_key, model_data in [
            ("model_1", self.model_1_data),
            ("model_2", self.model_2_data),
            ("model_3", self.model_3_data)
        ]:
            ax4.plot(model_data['Time'], model_data['Heading_Error'], 
                     label=self.model_names[model_key], 
                     color=self.model_colors[model_key], 
                     linewidth=1.5)
        
        ax4.set_title('Heading Error Comparison', fontsize=12)
        ax4.set_xlabel('Time (s)', fontsize=10)
        ax4.set_ylabel('Heading Error (rad)', fontsize=10)
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.legend(fontsize=10)
        
        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存图表
        plt.savefig("model_results/model_tracking_comparison.png", dpi=300, bbox_inches='tight')
        print("Trajectory tracking comparison figure saved to model_results/model_tracking_comparison.png")
        
        # 显示图表
        plt.show()
        
    def run_all_analyses(self):
        """运行所有分析"""
        
        # 运行轨迹对比
        print("Generating trajectory tracking comparison figure...")
        self.plot_trajectory_comparison()
        
        print("All analyses completed!")

# Main function
if __name__ == "__main__":
    print("=== Three Ship Dynamic Models Trajectory Tracking Comparison Tool ===")
    
    # 创建比较器实例
    comparator = ModelTrackingComparison()
    
    # 运行所有分析
    comparator.run_all_analyses()