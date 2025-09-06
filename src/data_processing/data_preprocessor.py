import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt
from sklearn.preprocessing import StandardScaler

# 可选依赖：filterpy（用于扩展卡尔曼滤波）
try:
    from filterpy.kalman import ExtendedKalmanFilter
    from filterpy.common import Q_discrete_white_noise
    FILTERPY_AVAILABLE = True
except ImportError:
    FILTERPY_AVAILABLE = False
    print("警告: filterpy未安装，扩展卡尔曼滤波功能将不可用。可通过 'pip install filterpy' 安装。")

class DataPreprocessor:
    """
    数据预处理类，用于对船舶运行数据进行预处理
    支持多种滤波方法：平滑滤波、扩展卡尔曼滤波、低通滤波
    """
    def __init__(self, x, y, psi, Ts, Tp, dt, filter_method='savgol'):
        """
        初始化数据预处理器
        
        参数:
            x: x坐标数据
            y: y坐标数据
            psi: 航向角数据
            Ts: 右推进器PWM值
            Tp: 左推进器PWM值
            dt: 时间步长
        """
        self.x = x
        self.y = y
        self.psi = psi
        self.Ts = Ts
        self.Tp = Tp
        self.dt = dt
        self.filter_method = filter_method  # 'savgol', 'ekf', 'lowpass', 'none'
        
        # 预处理后的数据
        self.psi_unwrapped = None
        self.u = None
        self.v = None
        self.r = None
        self.X = None
        self.U = None
        self.U_scaled = None
        self.dX = None
        self.U_scaler = None
        
    def unwrap_angle(self, psi):
        """
        角度预处理函数，将角度转换为弧度并标准化到[-pi, pi]范围
        
        参数:
            psi: 角度数据
            
        返回:
            numpy.ndarray: 处理后的角度数据
        """
        # return (psi + np.pi) % (2 * np.pi) - np.pi
        return psi
    
    def calculate_angular_velocity(self, psi, dt):
        """
        计算角速度
        
        参数:
            psi: 角度数据
            dt: 时间步长
            
        返回:
            numpy.ndarray: 角速度数据
        """
        psi_unwrapped = np.unwrap(psi)
        r = np.gradient(psi_unwrapped, dt)
        return r
    
    def apply_savgol_filter(self, data, window_length=15, polyorder=3):
        """
        应用Savitzky-Golay滤波
        
        参数:
            data: 输入数据
            window_length: 窗口长度
            polyorder: 多项式阶数
            
        返回:
            numpy.ndarray: 滤波后的数据
        """
        return savgol_filter(data, window_length=window_length, polyorder=polyorder)
    
    def apply_lowpass_filter(self, data, cutoff_freq=0.8, order=2):
        """
        应用低通滤波
        
        参数:
            data: 输入数据
            cutoff_freq: 截止频率（归一化频率）
            order: 滤波器阶数
            
        返回:
            numpy.ndarray: 滤波后的数据
        """
        b, a = butter(order, cutoff_freq, btype='low')
        return filtfilt(b, a, data)
    
    def apply_ekf_filter(self, data):
        """
        应用扩展卡尔曼滤波
        
        参数:
            data: 输入数据
            
        返回:
            numpy.ndarray: 滤波后的数据
        """
        if not FILTERPY_AVAILABLE:
            print("警告: filterpy未安装，使用简化的卡尔曼滤波实现")
            return self._simple_kalman_filter(data)
        
        # 简化的EKF实现，用于平滑数据
        n = len(data)
        filtered_data = np.zeros(n)
        
        # 初始化
        x = data[0]  # 初始状态
        P = 1.0      # 初始协方差
        Q = 0.01     # 过程噪声
        R = 0.1      # 测量噪声
        
        for i in range(n):
            # 预测步骤
            x_pred = x  # 简单的状态转移模型
            P_pred = P + Q
            
            # 更新步骤
            K = P_pred / (P_pred + R)  # 卡尔曼增益
            x = x_pred + K * (data[i] - x_pred)
            P = (1 - K) * P_pred
            
            filtered_data[i] = x
        
        return filtered_data
    
    def _simple_kalman_filter(self, data):
        """
        简化的卡尔曼滤波实现（不依赖filterpy）
        
        参数:
            data: 输入数据
            
        返回:
            numpy.ndarray: 滤波后的数据
        """
        n = len(data)
        filtered_data = np.zeros(n)
        
        # 初始化
        x = data[0]  # 初始状态
        P = 1.0      # 初始协方差
        Q = 0.01     # 过程噪声
        R = 0.1      # 测量噪声
        
        for i in range(n):
            # 预测步骤
            x_pred = x
            P_pred = P + Q
            
            # 更新步骤
            K = P_pred / (P_pred + R)
            x = x_pred + K * (data[i] - x_pred)
            P = (1 - K) * P_pred
            
            filtered_data[i] = x
        
        return filtered_data
    
    def apply_filter(self, data):
        """
        根据选择的方法应用滤波
        
        参数:
            data: 输入数据
            
        返回:
            numpy.ndarray: 滤波后的数据
        """
        if self.filter_method == 'savgol':
            return self.apply_savgol_filter(data)
        elif self.filter_method == 'ekf':
            return self.apply_ekf_filter(data)
        elif self.filter_method == 'lowpass':
            return self.apply_lowpass_filter(data)
        elif self.filter_method == 'none':
            return data
        else:
            print(f"警告: 未知的滤波方法 '{self.filter_method}'，使用默认的Savgol滤波")
            return self.apply_savgol_filter(data)
    
    def preprocess(self):
        """
        执行数据预处理
        
        返回:
            self: 返回自身实例以支持链式调用
        """
        # 角度预处理
        self.psi_unwrapped = self.unwrap_angle(self.psi)
        
        # 计算角速度
        self.r = self.calculate_angular_velocity(self.psi_unwrapped, self.dt)
        
        # 计算线速度
        dx_dt = np.gradient(self.x, self.dt)
        dy_dt = np.gradient(self.y, self.dt)
        
        # 计算船体坐标系下的速度
        self.u = dx_dt * np.cos(self.psi_unwrapped) + dy_dt * np.sin(self.psi_unwrapped)
        self.v = -dx_dt * np.sin(self.psi_unwrapped) + dy_dt * np.cos(self.psi_unwrapped)
        
        # 应用选择的滤波方法
        print(f"应用滤波方法: {self.filter_method}")
        self.u = self.apply_filter(self.u)
        self.v = self.apply_filter(self.v)
        self.r = self.apply_filter(self.r)
        
        # 构造状态变量和控制输入
        self.X = np.column_stack((self.u[:-1], self.v[:-1], self.r[:-1]))
        self.U = np.column_stack((self.Ts[:-1], self.Tp[:-1]))
        
        # 控制输入归一化
        self.U_scaler = StandardScaler()
        self.U_scaled = self.U_scaler.fit_transform(self.U)
        
        # 导数计算
        du = np.gradient(self.u, self.dt)
        dv = np.gradient(self.v, self.dt)
        dr = np.gradient(self.r, self.dt)
        self.dX = np.column_stack((du[1:], dv[1:], dr[1:]))
        
        return self
    
    def get_processed_data(self):
        """
        获取预处理后的数据
        
        返回:
            tuple: 包含所有预处理后的数据
        """
        return (self.u, self.v, self.r, self.X, self.U, 
                self.U_scaled, self.dX, self.U_scaler)