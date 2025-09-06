import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

class Visualizer:
    """
    可视化类，用于可视化仿真结果
    """
    def __init__(self, time, u, v, r, psi, x, y, Ts, Tp, 
                 u_sim, v_sim, r_sim, psi_sim, x_sim, y_sim, 
                 du, dv, dr, du_est, dv_est, dr_est, coef_df=None):
        """
        初始化可视化器
        
        参数:
            time: 时间序列
            u, v, r, psi, x, y: 真实状态变量
            Ts, Tp: 控制输入
            u_sim, v_sim, r_sim, psi_sim, x_sim, y_sim: 仿真状态变量
            du, dv, dr: 真实状态导数
            du_est, dv_est, dr_est: 估计状态导数
            coef_df: 系数矩阵的DataFrame表示（用于PySINDy模型）
        """
        self.time = time
        self.u = u
        self.v = v
        self.r = r
        self.psi = psi
        self.x = x
        self.y = y
        self.Ts = Ts
        self.Tp = Tp
        self.u_sim = u_sim
        self.v_sim = v_sim
        self.r_sim = r_sim
        self.psi_sim = psi_sim
        self.x_sim = x_sim
        self.y_sim = y_sim
        self.du = du
        self.dv = dv
        self.dr = dr
        self.du_est = du_est
        self.dv_est = dv_est
        self.dr_est = dr_est
        self.coef_df = coef_df
        
    def plot_state_derivatives(self):
        """
        绘制状态变量导数对比图
        """
        plt.figure(figsize=(10, 8))
        plt.subplot(3, 1, 1)
        plt.plot(self.time[:-1], self.du[:-1], label='True du')
        plt.plot(self.time[:-1], self.du_est[1:], '--', label='Estimated du')
        plt.legend()
        plt.title('du: True vs Estimated')

        plt.subplot(3, 1, 2)
        plt.plot(self.time[:-1], self.dv[:-1], label='True dv')
        plt.plot(self.time[:-1], self.dv_est[1:], '--', label='Estimated dv')
        plt.legend()
        plt.title('dv: True vs Estimated')

        plt.subplot(3, 1, 3)
        plt.plot(self.time[:-1], self.dr[:-1], label='True dr')
        plt.plot(self.time[:-1], self.dr_est[1:], '--', label='Estimated dr')
        plt.legend()
        plt.title('dr: True vs Estimated')
        plt.tight_layout()
        
        return self
    
    def plot_state_variables(self):
        """
        绘制状态变量对比图
        """
        plt.figure(figsize=(10, 10))
        plt.subplot(4, 1, 1)
        plt.plot(self.time[:-1], self.u[:-1], label='True u')
        plt.plot(self.time[:-1], self.u_sim[:-1], '--', label='Estimated u')
        plt.legend()
        plt.title('u: True vs Estimated')

        plt.subplot(4, 1, 2)
        plt.plot(self.time[:-1], self.v[:-1], label='True v')
        plt.plot(self.time[:-1], self.v_sim[:-1], '--', label='Estimated v')
        plt.legend()
        plt.title('v: True vs Estimated')

        plt.subplot(4, 1, 3)
        plt.plot(self.time[:-1], self.r[:-1], label='True r')
        plt.plot(self.time[:-1], self.r_sim[:-1], '--', label='Estimated r')
        plt.legend()
        plt.title('r: True vs Estimated')

        plt.subplot(4, 1, 4)
        # 将弧度转换为角度进行显示
        plt.plot(self.time[:-1], np.degrees(self.psi[:-1]), label='True psi')
        plt.plot(self.time[:-1], np.degrees(self.psi_sim[:-1]), '--', label='Estimated psi')
        plt.legend()
        plt.title('psi: True vs Estimated (degrees)')
        plt.ylabel('Angle (degrees)')
        plt.tight_layout()
        
        return self
    
    def plot_trajectory(self):
        """
        绘制轨迹对比图
        """
        plt.figure(figsize=(8, 8))
        plt.plot(self.x, self.y, label='True Trajectory')
        plt.plot(self.x_sim, self.y_sim, '--', label='Estimated Trajectory')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title('Trajectory Comparison')
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        
        return self
    
    def plot_propeller_outputs(self):
        """
        绘制推进器输出图
        """
        plt.figure(figsize=(10, 6))

        # Ts（右推进器）
        plt.subplot(2, 1, 1)
        plt.plot(self.time, self.Ts, label='Right Propeller (Ts)', color='blue')
        plt.legend(loc='upper center')
        plt.title('Propeller Outputs')
        plt.xlabel('Time (s)')
        plt.ylabel('PWM Value (Ts)')
        plt.grid(True)

        # Tp（左推进器）
        plt.subplot(2, 1, 2)
        plt.plot(self.time, self.Tp, label='Left Propeller (Tp)', color='blue')
        plt.legend(loc='upper center')
        plt.xlabel('Time (s)')
        plt.ylabel('PWM Value (Tp)')
        plt.grid(True)

        plt.tight_layout()
        
        return self
    
    
    def plot_coefficients(self):
        """
        绘制PySINDy模型的系数热图
        """
        if self.coef_df is not None:
            plt.figure(figsize=(12, 8))
            sns.heatmap(self.coef_df, annot=True, cmap='coolwarm', fmt='.3f')
            plt.title('SINDy Model Coefficients')
            plt.tight_layout()
        return self
    
    def show_all_plots(self):
        """
        显示所有图表
        """
        self.plot_state_derivatives()
        self.plot_state_variables()
        self.plot_trajectory()
        self.plot_propeller_outputs()
        if self.coef_df is not None:
            self.plot_coefficients()
        
        plt.show()
        
        return self