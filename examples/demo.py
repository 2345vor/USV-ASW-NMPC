import numpy as np
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
# ========================================
# 1. 加载 CSV 数据
# ========================================
Start_row = 0
row = 1500
# 读取Excel文件
# df = pd.read_excel('datas/boat1_2_circle.xlsx').iloc[Start_row:Start_row+row]
df = pd.read_excel('datas/boat1_2_sin.xlsx').iloc[Start_row:Start_row+row]

timestamp = df['DateTime'].values
x = df['x'].values
y = df['y'].values
psi = df['Heading'].values
Ts = df['PWM_R'].values-1500
Tp = df['PWM_L'].values-1500

# ========================================
# 2. 角度预处理函数（改进版：加入弧度转换）
# ========================================

def unwrap_angle(psi):
    return (psi * np.pi / 180 + np.pi) % (2 * np.pi) - np.pi

def calculate_angular_velocity(psi, dt):
    # psi_centered = unwrap_angle(psi)
    psi_unwrapped = np.unwrap(psi)
    r = np.gradient(psi_unwrapped, dt)
    return r

# ========================================
# 3. 参数设置与数据准备（加入平滑滤波）
# ========================================

N_samples = len(x)
dt_values = np.diff(np.array([pd.to_datetime(t).timestamp() for t in timestamp]))
dt_values = np.append(dt_values, dt_values[-1])
dt = np.mean(dt_values[dt_values > 0])

dx_dt = np.gradient(x, dt)
dy_dt = np.gradient(y, dt)
psi = unwrap_angle(psi)
r = calculate_angular_velocity(psi, dt)

u = dx_dt * np.cos(psi) + dy_dt * np.sin(psi)
v = -dx_dt * np.sin(psi) + dy_dt * np.cos(psi)

# 平滑滤波
u = savgol_filter(u, window_length=15, polyorder=1)
v = savgol_filter(v, window_length=15, polyorder=1)
r = savgol_filter(r, window_length=15, polyorder=1)

# 构造状态变量和控制输入
X = np.column_stack((u[:-1], v[:-1], r[:-1]))
U = np.column_stack((Ts[:-1], Tp[:-1]))

# 控制输入归一化
U_scaler = StandardScaler()
U_scaled = U_scaler.fit_transform(U)
# U_scaled = U
# 导数计算（基于滤波后变量）
du = np.gradient(u, dt)
dv = np.gradient(v, dt)
dr = np.gradient(r, dt)
dX = np.column_stack((du[1:], dv[1:], dr[1:]))

# ========================================
# 4. 模型方程定义
# ========================================

def model_equations(params, X, U):
    a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, b6, c1, c2, c3, c4, c5, c6 = params
    u, v, r = X[:, 0], X[:, 1], X[:, 2]
    Ts, Tp = U[:, 0], U[:, 1]

    du_model = a1 * v * r + a2 * u + a3 * v + a4 * r + a5 * (Tp + Ts) + a6
    dv_model = b1 * u * r + b2 * u + b3 * v + b4 * r + b5 * (Tp - Ts) + b6
    dr_model = c1 * u * v + c2 * u + c3 * v + c4 * r + c5 * (Tp - Ts) + c6

    return np.column_stack((du_model, dv_model, dr_model))

# ========================================
# 5. 损失函数（目标函数）
# ========================================

def loss_function(params, X, U, dX):
    pred = model_equations(params, X, U)
    error = pred - dX
    return np.sum(error ** 2)

# ========================================
# 6. 初始参数猜测
# ========================================

initial_params = np.array([
        -1.1391,   # a1
     0.0028,   # a2
     0.6836,   # a3
     0.6836,   # a4
     0.6836,   # a5
     0.6836,   # a6
     0.0161,   # b1
    -0.0052,   # b2
     0.002,    # b3
     0.6836,   # b4
     0.6836,   # b5
     0.6836,   # b6
     8.2861,   # c1
    -0.9860,   # c2
     0.0307,    # c3
     0.0307,    # c4
     0.0307,    # c5
     0.6836   # c6
])

# ========================================
# 7. 使用 SLSQP 进行参数优化
# ========================================

result = minimize(loss_function, initial_params, args=(X, U_scaled, dX), method='SLSQP', options={'disp': True})

if result.success:
    estimated_params = result.x
    print("参数优化成功！")
else:
    print("参数优化失败。")

# ========================================
# 8. 修复后的仿真函数（含运动学）
# ========================================

def simulate_with_params_and_kinematics(params, X0, U_all, x0, y0, psi0, dt, N_samples):
    a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, b6, c1, c2, c3, c4, c5, c6 = params

    u_sim = np.zeros(N_samples)
    v_sim = np.zeros(N_samples)
    r_sim = np.zeros(N_samples)
    x_sim = np.zeros(N_samples)
    y_sim = np.zeros(N_samples)
    psi_sim = np.zeros(N_samples)

    u_sim[0], v_sim[0], r_sim[0] = X0
    x_sim[0], y_sim[0], psi_sim[0] = x0, y0, psi0

    du_est = np.zeros(N_samples)
    dv_est = np.zeros(N_samples)
    dr_est = np.zeros(N_samples)

    for k in range(1, N_samples):
        Ts = U_all[k-1, 0]
        Tp = U_all[k-1, 1]

        u = u_sim[k-1]
        v = v_sim[k-1]
        r = r_sim[k-1]
        psi = psi_sim[k-1]

        du = a1 * v * r + a2 * u + a3 * v + a4 * r + a5 * (Tp + Ts) + a6
        dv = b1 * u * r + b2 * u + b3 * v + b4 * r + b5 * (Tp - Ts) + b6
        dr = c1 * u * v + c2 * u + c3 * v + c4 * r + c5 * (Tp - Ts) + c6

        u_sim[k] = u_sim[k-1] + dt * du
        v_sim[k] = v_sim[k-1] + dt * dv
        r_sim[k] = r_sim[k-1] + dt * dr

        x_sim[k] = x_sim[k-1] + dt * (u * np.cos(psi) - v * np.sin(psi))
        y_sim[k] = y_sim[k-1] + dt * (u * np.sin(psi) + v * np.cos(psi))
        psi_sim[k] = psi_sim[k-1] + dt * r_sim[k-1]
        psi_sim[k] = (psi_sim[k] + np.pi) % (2 * np.pi) - np.pi

        du_est[k] = du
        dv_est[k] = dv
        dr_est[k] = dr

    return x_sim, y_sim, psi_sim, u_sim, v_sim, r_sim, du_est, dv_est, dr_est

# ========================================
# 9. 结果分析与绘图
# ========================================

if result.success:
    estimated_params = result.x
    X0 = [u[0], v[0], r[0]]
    x0, y0, psi0 = x[0], y[0], psi[0]
    U_all = U_scaled

    x_est, y_est, psi_est, u_est, v_est, r_est, du_est, dv_est, dr_est = simulate_with_params_and_kinematics(
        estimated_params, X0, U_all, x0, y0, psi0, dt, N_samples
    )

    time = np.arange(N_samples) * dt

    estimated_params = result.x
    a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, b6, c1, c2, c3, c4, c5, c6 = estimated_params
    
    # 打印模型方程
    print("\n辨识的模型方程:")
    print(f"{a1:.6f} * v * r + {a2:.6f} * u + {a3:.6f} * v + {a4:.6f} * r + {a5:.6f} * (Tp + Ts) + {a6:.6f},")
    print(f"{b1:.6f} * u * r + {b2:.6f} * u + {b3:.6f} * v + {b4:.6f} * r + {b5:.6f} * (Tp - Ts) + {b6:.6f},")
    print(f"{c1:.6f} * u * v + {c2:.6f} * u + {c3:.6f} * v + {c4:.6f} * r + {c5:.6f} * (Tp - Ts) + {c6:.6f},")
    # 可视化状态变量导数
    plt.figure(0)
    plt.subplot(3, 1, 1)
    plt.plot(time[:-1], du[:-1], label='True du')
    plt.plot(time[:-1], du_est[1:], '--', label='Estimated du')
    plt.legend()
    plt.title('du: True vs Estimated')

    plt.subplot(3, 1, 2)
    plt.plot(time[:-1], dv[:-1], label='True dv')
    plt.plot(time[:-1], dv_est[1:], '--', label='Estimated dv')
    plt.legend()
    plt.title('dv: True vs Estimated')

    plt.subplot(3, 1, 3)
    plt.plot(time[:-1], dr[:-1], label='True dr')
    plt.plot(time[:-1], dr_est[1:], '--', label='Estimated dr')
    plt.legend()
    plt.title('dr: True vs Estimated')
    plt.tight_layout()

    # 可视化状态变量
    plt.figure(1)
    plt.subplot(4, 1, 1)
    plt.plot(time[:-1], u[:-1], label='True u')
    plt.plot(time[:-1], u_est[:-1], '--', label='Estimated u')
    plt.legend()
    plt.title('u: True vs Estimated')

    plt.subplot(4, 1, 2)
    plt.plot(time[:-1], v[:-1], label='True v')
    plt.plot(time[:-1], v_est[:-1], '--', label='Estimated v')
    plt.legend()
    plt.title('v: True vs Estimated')

    plt.subplot(4, 1, 3)
    plt.plot(time[:-1], r[:-1], label='True r')
    plt.plot(time[:-1], r_est[:-1], '--', label='Estimated r')
    plt.legend()
    plt.title('r: True vs Estimated')

    plt.subplot(4, 1, 4)
    plt.plot(time[:-1], psi[:-1], label='True psi')
    plt.plot(time[:-1], psi_est[:-1], '--', label='Estimated psi')
    plt.legend()
    plt.title('psi: True vs Estimated')
    plt.tight_layout()

    # 轨迹对比
    plt.figure(2)
    plt.plot(x, y, label='True Trajectory')
    plt.plot(x_est, y_est, '--', label='Estimated Trajectory')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Trajectory Comparison')
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    # ========================================
    # 新增：7. 推进器输出差值对比 (plt.figure(3))
    # ========================================

    plt.figure(3)

    # Ts（右推进器）
    plt.subplot(2, 1, 1)
    plt.plot(time, Ts, label='Right Propeller (Ts)', color='blue')
    plt.legend(loc='upper center')
    plt.title('Propeller Outputs')
    plt.xlabel('Time (s)')
    plt.ylabel('PWM Value (Ts)')
    plt.grid(True)

    # Tp（左推进器）
    plt.subplot(2, 1, 2)
    plt.plot(time, Tp, label='Left Propeller (Tp)', color='blue')
    plt.legend(loc='upper center')
    plt.xlabel('Time (s)')
    plt.ylabel('PWM Value (Tp)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
