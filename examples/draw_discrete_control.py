import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ========================================
# 1. 加载 CSV 数据
# ========================================
Start_row = 1
row = 1500
df = pd.read_excel('datas/boat1_2_sin.xlsx').iloc[Start_row:Start_row+row]

timestamp = df['DateTime'].values
x = df['x'].values
y = df['y'].values
psi = df['Heading'].values
Ts = df['PWM_L'].values  # 控制输入 1
Tp = df['PWM_R'].values  # 控制输入 2

# ========================================
# 2. 时间向量构建
# ========================================
dt_values = np.diff([pd.to_datetime(t).timestamp() for t in timestamp])
dt_values = np.append(dt_values, dt_values[-1])  # 补齐长度
dt = np.mean(dt_values[dt_values > 0])
time = np.arange(len(x)) * dt  # 构造时间向量

# ========================================
# 3. 角度预处理函数
# ========================================
def unwrap_angle(psi):
    return (psi * np.pi / 180 + np.pi) % (2 * np.pi) - np.pi

def calculate_angular_velocity(psi, dt):
    psi_centered = unwrap_angle(psi)
    psi_unwrapped = np.unwrap(psi_centered)
    r = np.gradient(psi_unwrapped, dt)
    return r

# ========================================
# 4. 参数设置与状态变量计算
# ========================================
dx_dt = np.gradient(x, dt)
dy_dt = np.gradient(y, dt)
psi = unwrap_angle(psi)
r = calculate_angular_velocity(psi, dt)

u = dx_dt * np.cos(psi) + dy_dt * np.sin(psi)
v = -dx_dt * np.sin(psi) + dy_dt * np.cos(psi)

# ========================================
# 5. 绘制 8 幅图（每图单信号）
# ========================================
plt.figure(3, figsize=(12, 14))
linewidth_ = 4
# 子图 1: u
plt.subplot(8, 1, 1)
plt.plot(time, u, color='blue', linewidth=linewidth_)
plt.title('u (cm/s)')
plt.xlabel('Time (s)')
plt.ylabel('u')

# 子图 2: v
plt.subplot(8, 1, 2)
plt.plot(time, v, color='green', linewidth=linewidth_)
plt.title('v (cm/s)')
plt.xlabel('Time (s)')
plt.ylabel('v')

# 子图 3: r
plt.subplot(8, 1, 3)
plt.plot(time, r, color='orange', linewidth=linewidth_)
plt.title('r (rad/s)')
plt.xlabel('Time (s)')
plt.ylabel('r')

# 子图 4: Ts（右推进器）
plt.subplot(8, 1, 4)
plt.step(time, Ts, where='post', color='red', linewidth=linewidth_)
plt.title('Right Propeller Output (Ts)')
plt.xlabel('Time (s)')
plt.ylabel('PWM (Ts)')

# 子图 5: Tp（左推进器）
plt.subplot(8, 1, 5)
plt.step(time, Tp, where='post', color='purple', linewidth=linewidth_)
plt.title('Left Propeller Output (Tp)')
plt.xlabel('Time (s)')
plt.ylabel('PWM (Tp)')

# 计算导数
du = np.gradient(u, dt)
dv = np.gradient(v, dt)
dr = np.gradient(r, dt)

# 子图 6: du/dt
plt.subplot(8, 1, 6)
plt.plot(time, du, color='blue', linewidth=linewidth_)
plt.title('du/dt (cm/s²)')
plt.xlabel('Time (s)')
plt.ylabel('du/dt')

# 子图 7: dv/dt
plt.subplot(8, 1, 7)
plt.plot(time, dv, color='green', linewidth=linewidth_)
plt.title('dv/dt (cm/s²)')
plt.xlabel('Time (s)')
plt.ylabel('dv/dt')

# 子图 8: dr/dt
plt.subplot(8, 1, 8)
plt.plot(time, dr, color='orange', linewidth=linewidth_)
plt.title('dr/dt (rad/s²)')
plt.xlabel('Time (s)')
plt.ylabel('dr/dt')

plt.tight_layout()
plt.show()