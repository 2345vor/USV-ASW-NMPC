import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

class DataLoader:
    """
    数据加载类，用于从Excel文件中加载船舶运行数据并进行初步处理
    """
    def __init__(self, file_path, start_row=0, row_count=1500):
        """
        初始化数据加载器
        
        参数:
            file_path: Excel文件路径
            start_row: 起始行
            row_count: 读取的行数
        """
        self.file_path = file_path
        self.start_row = start_row
        self.row_count = row_count
        self.data = None
        self.timestamp = None
        self.x = None
        self.y = None
        self.psi = None
        self.Ts = None
        self.Tp = None
        self.dt = None
        
    def load_data(self):
        """
        加载Excel数据
        
        返回:
            self: 返回自身实例以支持链式调用
        """
        try:
            # 读取Excel文件
            self.data = pd.read_excel(self.file_path).iloc[self.start_row:self.start_row+self.row_count]
            
            # 提取关键数据
            self.timestamp = self.data['DateTime'].values
            self.x = self.data['x'].values
            self.y = self.data['y'].values
            self.psi = self.data['course'].values * np.pi / 180
            self.Ts = self.data['PWM_R'].values - 1500  # 右推进器PWM值归一化
            self.Tp = self.data['PWM_L'].values - 1500  # 左推进器PWM值归一化
            
            # 计算时间步长
            dt_values = np.diff(np.array([pd.to_datetime(t).timestamp() for t in self.timestamp]))
            dt_values = np.append(dt_values, dt_values[-1])
            self.dt = np.mean(dt_values[dt_values > 0])
            
            return self
        except Exception as e:
            print(f"数据加载错误: {e}")
            raise
    
    def get_data(self):
        """
        获取加载的数据
        
        返回:
            tuple: 包含所有加载的数据
        """
        return (self.timestamp, self.x, self.y, self.psi, 
                self.Ts, self.Tp, self.dt)