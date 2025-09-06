import casadi as ca
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt



# ==================== 常量定义 ====================
M_PI = np.pi
N = 10           # 预测步长
T = 0.1          # 采样时间
motor_power = 500
Adaptive_flag = True
position_noise_std = 0.01   # 位置噪声标准差
loop_num = 1
tol = 210*loop_num  # 仿真总时间s
t = np.linspace(0, 2*M_PI*loop_num, int(tol / T) + 1)  # 时间向量

# ==================== 读取CSV文件并提取目标路径点 ====================
x_r1 = 40 * np.sin(t) + 1
y_r1 = 30 *  np.cos(t) + 1

x_r1 = x_r1[:, np.newaxis]
y_r1 = y_r1[:, np.newaxis]
# path_points = [x_r1, y_r1]
path_points = np.concatenate([x_r1, y_r1], axis=1)
# path_points = df[['LosXF', 'LosYF']].values.tolist()
sim_steps = len(path_points)

# ==================== 权重参数 ====================
Q_low = [0.5, 0.1, 0.1, 30, 30, 0.1]
Q_high = [0.5, 0.1, 0.1, 1000, 1000, 1]
R_low = [0.001, 0.001]
R_high = [0.001, 0.001]
F_low = [0.1, 0.2, 0.2, 10, 10, 0.15]
F_high = [0.1, 0.2, 0.2, 5000, 5000, 0.5]

# ==================== 状态和控制变量定义 ====================
u = ca.SX.sym('u')
v = ca.SX.sym('v')
r = ca.SX.sym('r')
x_pos = ca.SX.sym('x')
y_pos = ca.SX.sym('y')
psi = ca.SX.sym('psi')
Tp = ca.SX.sym('Tp')
Ts = ca.SX.sym('Ts')

state = ca.vertcat(u, v, r, x_pos, y_pos, psi)
control = ca.vertcat(Tp, Ts)

# ==================== 动力学模型 ====================
rhs = ca.vertcat(
    #SLSQP2
    # -0.608532 * v * r + -0.879879 * u + 1.558240 * v + -0.289793 * r + 0.018922 * Tp + 0.018380 * Ts + 0.861627,
    # 0.059533 * u * r + 0.000476 * u + -0.542198 * v + -0.207847 * r + 0.000417 * Tp + -0.006426 * Ts + 0.001430,         
    # -0.157841 * u * v + -0.225022 * u + 4.817326 * v + -1.115397 * r + 0.023376 * Tp + -0.022981 * Ts + 0.233705,
    #SLSQP
    -0.737479 * v * r + -0.880204 * u + 1.563570 * v + -0.284868 * r + 0.018785 * (Tp + Ts) + 0.861869,
    0.209936 * u * r + -0.007101 * u + -0.558597 * v + -0.317765 * r + 0.002451 * (Tp - Ts) + 0.008543,
    -0.269228 * u * v + -0.248181 * u + 6.781629 * v + -0.924642 * r + 0.032856 * (Tp - Ts) + 0.253862,
    #SINDy_auto
    # 0.089487 + -0.095543 *u + -0.147440 *v  *r + 0.031546 *Ts + 0.042889 *Tp,
    # 0.018478 + -0.013773 *u *r + -0.300011 *v  + -0.018374 *Ts + 0.01205 *Tp,       
    # 0.012093 + -0.015564 *u  *v + -0.102018 *r + -0.00342 *Ts + 0.002441 *Tp,
    # 0.089487 + -0.095543 *u + -0.147440 *v + 0.371778 *r + 0.031546 *Ts + 0.042889 *Tp,
    # 0.018478 + -0.013773 *u + -0.300011 *v + -0.398682 *r + -0.018374 *Ts + 0.001205 *Tp,       
    # 0.012093 + -0.215564 *u + 6.489318 *v + 0.102018 *r + -0.0142 *Ts + 0.022441 *Tp,
    u * ca.cos(psi) - v * ca.sin(psi),
    u * ca.sin(psi) + v * ca.cos(psi),
    r  
)

f = ca.Function('f', [state, control], [rhs])

# ==================== 初始状态 ====================
x0 = [0, 0, 0, path_points[0][0] - 5, path_points[0][1] - 5, 0]
xs_list = [[0, 0, 0, point[0], point[1], 0] for point in path_points]
state_history = [x0]
u0 = np.zeros(2 * N).tolist()
control_history = []

# ==================== 构建NLP问题 ====================
U = ca.SX.sym('U', 2, N)
P_state = ca.SX.sym('P_state', 12)
P_weight = ca.SX.sym('P_weight', 14)
P = ca.vertcat(P_state, P_weight)

X = ca.SX.sym('X', 6, N + 1)
X[:, 0] = P_state[0:6]

for k in range(N):
    X[:, k + 1] = X[:, k] + T * f(X[:, k], U[:, k])

obj = 0
Qk = ca.diagcat(P_weight[0], P_weight[1], P_weight[2], P_weight[3], P_weight[4], P_weight[5])
Rk = ca.diagcat(P_weight[6], P_weight[7])
Qf = ca.diagcat(P_weight[8], P_weight[9], P_weight[10], P_weight[11], P_weight[12], P_weight[13])

for k in range(N):
    st_err = X[:, k] - P_state[6:12]
    con = U[:, k]
    obj += st_err.T @ Qk @ st_err + con.T @ Rk @ con

obj += (X[:, N] - P_state[6:12]).T @ Qf @ (X[:, N] - P_state[6:12])

g = []
for k in range(N + 1):
    g.append(X[0, k])
    g.append(X[1, k])
    g.append(X[2, k])
g = ca.vertcat(*g)

OPT_variables = ca.reshape(U, 2 * N, 1)
nlp = {'x': OPT_variables, 'f': obj, 'g': g, 'p': P}

opts = {
    'ipopt.print_level': 0,
    'print_time': 0,
    'ipopt.max_iter': 30,
    'ipopt.acceptable_tol': 1e-5
}
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

# ==================== 约束边界 ====================
lbx = [-motor_power,-motor_power] * (N)
ubx = [motor_power,motor_power] * (N)
lbg = [-3, -1, -0.3*M_PI] * (N+1)
ubg = [3, 1, 0.3*M_PI] * (N+1)

# ==================== 初始化误差记录 ====================
lateral_errors = []
heading_errors = []

def normalize_angle_diff(delta):
    if delta > M_PI:
        return delta - 2 * M_PI
    elif delta < -M_PI:
        return delta + 2 * M_PI
    else:
        return delta

def compute_target_psi(xs_list, sim_index):
    if sim_index >= len(xs_list) - 1:
        return xs_list[sim_index][5]
    current_x = xs_list[sim_index][3]
    current_y = xs_list[sim_index][4]
    next_x = xs_list[sim_index + 1][3]
    next_y = xs_list[sim_index + 1][4]
    return math.atan2(next_y - current_y, next_x - current_x)

# ==================== NMPC 仿真 ====================
for sim in range(sim_steps):
    xs = xs_list[sim] if sim < len(xs_list) else xs_list[-1]

    current_x, current_y = x0[3], x0[4]
    target_x, target_y = xs[3], xs[4]
    x0[5] = normalize_angle_diff(x0[5])
    current_psi = x0[5]
    target_psi = compute_target_psi(xs_list, sim)
    # target_psi = normalize_angle_diff(target_psi)
    distance_error = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
    lateral_errors.append(distance_error)

    delta_psi = normalize_angle_diff(target_psi-current_psi)
    heading_errors.append(abs(delta_psi))

    distance_coeff = max(0, min(1.0, distance_error / 100.0))
    if Adaptive_flag:
        Q_current = Q_low
        R_current = R_low
        # F_current = [F_low[i] + distance_coeff * (F_high[i] - F_low[i]) for i in range(6)]
        distance_coeff = distance_error* np.cos(delta_psi) / 100.0
        F_current = [(F_high[i] + F_low[i])/2 + distance_coeff /(1+np.abs(distance_coeff))* (F_high[i] - F_low[i]) for i in range(6)]
    else:
        Q_current = Q_low
        R_current = R_low
        F_current = F_low

    p_state = x0 + xs
    p_weight = Q_current + R_current + F_current
    p = p_state + p_weight

    res = solver(x0=u0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p)

    u_opt = res['x'].full().flatten().tolist()
    u0_step = u_opt[:2]
    # 改进方案2：添加控制输入滤波
    if len(control_history) > 0:
        # 对控制输入进行平滑处理，减少突变
        prev_control = np.array(control_history[-1])
        current_control = np.array(u_opt[:2])
        # 加权平均，更多依赖优化结果，少量依赖历史控制
        u0_step = (0.9 * current_control + 0.1 * prev_control).tolist()
    else:
        u0_step = u_opt[:2]

    next_state = f(x0, u0_step)
    x_next = x0 + T * next_state.full().flatten()

    # 加入白噪声
    noise = np.random.normal(0, position_noise_std, size=2)
    x_next[3] += noise[0]
    x_next[4] += noise[1]
    x0 = x_next.tolist()
    control_history.append(u0_step)
    state_history.append(x0)
    u0 = u_opt[2:] + u_opt[-2:]

# ==================== 提取轨迹用于绘图 ====================
traj_u =   [s[0] for s in state_history][1:]
traj_v =   [s[1] for s in state_history][1:]
traj_r =   [s[2] for s in state_history][1:]
traj_x =   [s[3] for s in state_history][1:]
traj_y =   [s[4] for s in state_history][1:]
traj_psi = [s[5] for s in state_history][1:]
ref_x = [p[0] for p in path_points]
ref_y = [p[1] for p in path_points]
PWML = [u[0] for u in control_history]
PWMR = [u[1] for u in control_history]
# ==================== 输出统计数据 ====================
print(f"Lateral Error: Mean={np.mean(lateral_errors):.2f}, Std={np.std(lateral_errors):.2f}")
print(f"Heading Error: Mean={np.mean(heading_errors):.2f} rad, Std={np.std(heading_errors):.2f} rad")
# ==================== 绘图 ====================
plt.figure(figsize=(10, 6))
plt.plot(ref_x, ref_y, 'r--', label='Reference Path (LosXF/YF)')
plt.plot(traj_x, traj_y, 'b-', label='NMPC Trajectory')
plt.scatter([traj_x[0]], [traj_y[0]], c='g', label='Start')
plt.scatter([traj_x[-1]], [traj_y[-1]], c='orange', label='End')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('NMPC Path Tracking vs Reference Path')
plt.legend()
plt.grid(True)
plt.axis('equal')
# plt.show()

# ==================== 误差绘图 ====================
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(lateral_errors, label='Lateral Tracking Error', color='b')
plt.axhline(y=np.mean(lateral_errors), color='b', linestyle='--', label=f'Mean Lateral Error: {np.mean(lateral_errors):.2f}')
plt.subplot(2, 1, 2)
plt.plot(heading_errors, label='Heading Angle Error (rad)', color='r')
plt.axhline(y=np.mean(heading_errors), color='r', linestyle='--', label=f'Mean Heading Error: {np.mean(heading_errors):.2f} rad')
plt.xlabel('Time Step')
plt.ylabel('Error')
plt.title('Lateral and Heading Errors During NMPC Path Tracking')
plt.legend()
plt.grid(True)

# 可视化状态变量
plt.figure(figsize=(12, 6))
plt.subplot(4, 1, 1)
plt.plot(t[:-1], traj_u[:-1], label='u')
# plt.plot(t[:-1], u_est[:-1], '--', label='Estimated u')
plt.legend()
# plt.title('u: True vs Estimated')

plt.subplot(4, 1, 2)
plt.plot(t[:-1], traj_v[:-1], label='v')
# plt.plot(t[:-1], v_est[:-1], '--', label='Estimated v')
plt.legend()
# plt.title('v: True vs Estimated')

plt.subplot(4, 1, 3)
plt.plot(t[:-1], traj_r[:-1], label=' r')
# plt.plot(t[:-1], r_est[:-1], '--', label='Estimated r')
plt.legend()
# plt.title('r: True vs Estimated')

plt.subplot(4, 1, 4)
plt.plot(t[:-1], traj_psi[:-1], label='True psi')
# plt.plot(t[:-1], psi_est[:-1], '--', label='Estimated psi')
plt.legend()
# plt.title('psi: True vs Estimated')
plt.tight_layout()

# 新增：7. 推进器输出差值对比 (plt.figure(3))
plt.figure(figsize=(12, 6))
# Ts（右推进器）
plt.subplot(2, 1, 1)
plt.plot(t, PWML, label=' Ts', color='blue')
plt.legend(loc='upper center')
plt.title('Propeller Outputs')
plt.xlabel('Time (s)')
plt.ylabel('PWM Value (us)')
plt.grid(True)

# Tp（左推进器）
plt.subplot(2, 1, 2)
plt.plot(t, PWMR, label='Tp', color='blue')
plt.legend(loc='upper center')
plt.xlabel('Time (s)')
plt.ylabel('PWM Value (us)')
plt.grid(True)

plt.tight_layout()
plt.show()


# ==================== 保存为CSV ====================
import os

output_data = {
    'X': traj_x,
    'Y': traj_y,
    'LosXF': ref_x,
    'LosYF': ref_y,
    'PWML': PWML,
    'PWMR': PWMR
}

df_output = pd.DataFrame(output_data)
# output_path = os.path.join(os.path.dirname(csv_file_path), 's0621atnmpc.csv')
# df_output.to_csv(output_path, index=False)

# print(f"数据已保存至: {output_path}")