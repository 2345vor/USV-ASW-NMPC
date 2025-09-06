import numpy as np

class Simulator:
    """
    仿真类，用于模拟船舶运动
    """
    def __init__(self, params, X0, U_all, x0, y0, psi0, dt, N_samples, model_type='model_1'):
        """
        初始化仿真器
        
        参数:
            params: 模型参数
            X0: 初始状态 [u0, v0, r0]
            U_all: 所有控制输入
            x0, y0, psi0: 初始位置和航向角
            dt: 时间步长
            N_samples: 样本数量
            model_type: 模型类型 ('model_1', 'model_2', 'model_3')
        """
        self.params = params
        self.model_type = model_type
        self.X0 = X0
        self.U_all = U_all
        self.x0 = x0
        self.y0 = y0
        self.psi0 = psi0
        self.dt = dt
        self.N_samples = N_samples
        
        # 仿真结果
        self.u_sim = None
        self.v_sim = None
        self.r_sim = None
        self.x_sim = None
        self.y_sim = None
        self.psi_sim = None
        self.du_est = None
        self.dv_est = None
        self.dr_est = None
        
    def simulate(self):
        """
        执行仿真
        
        返回:
            self: 返回自身实例以支持链式调用
        """
        # Initialize simulation arrays
        self.u_sim = np.zeros(self.N_samples)
        self.v_sim = np.zeros(self.N_samples)
        self.r_sim = np.zeros(self.N_samples)
        self.x_sim = np.zeros(self.N_samples)
        self.y_sim = np.zeros(self.N_samples)
        self.psi_sim = np.zeros(self.N_samples)

        self.u_sim[0], self.v_sim[0], self.r_sim[0] = self.X0
        self.x_sim[0], self.y_sim[0], self.psi_sim[0] = self.x0, self.y0, self.psi0

        self.du_est = np.zeros(self.N_samples)
        self.dv_est = np.zeros(self.N_samples)
        self.dr_est = np.zeros(self.N_samples)

        for k in range(1, self.N_samples):
            Ts = self.U_all[k-1, 0]
            Tp = self.U_all[k-1, 1]

            u = self.u_sim[k-1]
            v = self.v_sim[k-1]
            r = self.r_sim[k-1]
            psi = self.psi_sim[k-1]

            # Calculate dynamics based on model type
            if self.model_type == 'model_1':
                # Standard model (18 parameters)
                a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, b6, c1, c2, c3, c4, c5, c6 = self.params
                du = a1 * v * r + a2 * u + a3 * v + a4 * r + a5 * (Tp + Ts) + a6
                dv = b1 * u * r + b2 * u + b3 * v + b4 * r + b5 * (Tp - Ts) + b6
                dr = c1 * u * v + c2 * u + c3 * v + c4 * r + c5 * (Tp - Ts) + c6
            elif self.model_type == 'model_2':
                # Separated thruster input model (21 parameters)
                a1, a2, a3, a4, a5, a6, a7, b1, b2, b3, b4, b5, b6, b7, c1, c2, c3, c4, c5, c6, c7 = self.params
                du = a1 * v * r + a2 * u + a3 * v + a4 * r + a5 * Tp + a6 * Ts + a7
                dv = b1 * u * r + b2 * u + b3 * v + b4 * r + b5 * Tp + b6 * Ts + b7
                dr = c1 * u * v + c2 * u + c3 * v + c4 * r + c5 * Tp + c6 * Ts + c7
            elif self.model_type == 'model_3':
                # Simplified model (16 parameters)
                a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6 = self.params
                du = a1 * v * r + a2 * u + a3 * r + a4 + a5 * (Tp + Ts)
                dv = b1 * u * r + b2 * v + b3 * r + b4 + b5 * (Tp - Ts)
                dr = c1 * u * v + c2 * u + c3 * v + c4 * r + c5 * (Tp - Ts) + c6
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            self.u_sim[k] = self.u_sim[k-1] + self.dt * du
            self.v_sim[k] = self.v_sim[k-1] + self.dt * dv
            self.r_sim[k] = self.r_sim[k-1] + self.dt * dr

            self.x_sim[k] = self.x_sim[k-1] + self.dt * (u * np.cos(psi) - v * np.sin(psi))
            self.y_sim[k] = self.y_sim[k-1] + self.dt * (u * np.sin(psi) + v * np.cos(psi))
            self.psi_sim[k] = self.psi_sim[k-1] + self.dt * self.r_sim[k-1]
            self.psi_sim[k] = (self.psi_sim[k] + np.pi) % (2 * np.pi) - np.pi

            self.du_est[k] = du
            self.dv_est[k] = dv
            self.dr_est[k] = dr
            
        return self
    
    def get_simulation_results(self):
        """
        获取仿真结果
        
        返回:
            tuple: 包含所有仿真结果
        """
        if self.x_sim is None:
            raise ValueError("仿真尚未执行，请先调用simulate方法")
            
        return (self.x_sim, self.y_sim, self.psi_sim, 
                self.u_sim, self.v_sim, self.r_sim, 
                self.du_est, self.dv_est, self.dr_est)