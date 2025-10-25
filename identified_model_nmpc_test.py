import casadi as ca
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import argparse
import sys

# ==================== 模型类型定义 ====================
class ModelType:
    MODEL_1 = 1  # 基础模型 (18参数)
    MODEL_2 = 2  # 分离模型 (21参数) 
    MODEL_3 = 3  # 简化模型 (16参数)

# ==================== 加载识别的模型参数 ====================
def load_identified_params(params_file):
    """加载识别的模型参数：从metadata JSON的`parameters`字段读取，兼容直接列表"""
    with open(params_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        if 'parameters' in data and isinstance(data['parameters'], list):
            params_raw = data['parameters']
        elif 'params' in data and isinstance(data['params'], list):
            params_raw = data['params']
        else:
            raise ValueError(f"参数文件不包含'parameters'或'params'列表: {params_file}")
    elif isinstance(data, list):
        params_raw = data
    else:
        raise ValueError(f"不支持的参数文件格式: {type(data)} - {params_file}")
    
    # 对参数进行限制，保留小数部分（与原逻辑保持一致）
    params = [x - int(x) + 1 if x > 10 else x - int(x) - 1 if x < -10 else x for x in params_raw]
    
    return params

def get_available_models():
    """获取可用的模型参数文件"""
    available_models = []
    test_results_dir = 'model_results'
    
    for model_num in [1, 2, 3]:
        params_file = os.path.join(test_results_dir, f'model_{model_num}_identification_metadata.json')
        if os.path.exists(params_file):
            available_models.append((model_num, params_file))
    
    return available_models

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='NMPC轨迹跟踪控制验证')
    parser.add_argument('--model', type=int, choices=[1, 2, 3], default=1,
                       help='模型类别 (1: 基础模型18参数, 2: 分离模型21参数, 3: 简化模型16参数)')
    parser.add_argument('--trajectory', type=int, choices=[1, 2, 3], default=1,
                       help='跟踪曲线 (1: 椭圆, 2: 正弦直线, 3: 双正弦)')
    parser.add_argument('--predict_step', type=int, default=10,
                       help='预测步长,最大值20,最小值5,默认10')
    parser.add_argument('--dt', type=float, default=0.1,
                       help='采样时间,最大值0.5,最小值0.05,默认0.1')
    parser.add_argument('--cycle_time', type=int, default=210,
                       help='轨迹周期时间,最大值360,最小值210,默认210,推荐是3的倍数')
    parser.add_argument('--loop_num', type=int, default=1,
                       help='循环次数,最大值5,最小值1,默认1')
    parser.add_argument('--noise_mean', type=float, default=-0.01,
                       help='轨迹噪声均值,默认0.0')
    parser.add_argument('--noise_std', type=float, default=0.01,
                       help='轨迹噪声标准差,默认0.0')
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='自适应NMPC控制参数,默认0.1')
    parser.add_argument('--adaptive', action='store_true', default=False,
                       help='是否启用自适应NMPC控制')
    parser.add_argument('--output_dir', type=str, default='nmpc_results',
                       help='输出目录')
    return parser.parse_args()

def get_trajectory_equations(trajectory_type, t):
    """根据轨迹类型生成轨迹方程"""
    if trajectory_type == 1:
        # 椭圆轨迹
        x_r = 40 * np.sin(t) + 1
        y_r = 30 * np.cos(t) + 1
        name = "Elliptical Trajectory: x = 40*sin(t) + 1, y = 30*cos(t) + 1"
    elif trajectory_type == 2:
        # 正弦直线轨迹
        x_r = 40 * np.sin(t) + 1
        y_r = 10*t
        name = "SIN Trajectory: x = 40*sin(t) + 1, y = t"
    elif trajectory_type == 3:
        # 双正弦轨迹
        x_r = 25 * np.cos(2*t +1.7)
        y_r = 25 * np.sin(t +1.7)
        name = "Lissajous Trajectory: x = 25*cos(2*t +1.7), y = 25*sin(t +1.7)"
    else:
        raise ValueError(f"不支持的轨迹类型: {trajectory_type}")
    
    return x_r, y_r, name

# 解析命令行参数
args = parse_arguments()

# 设置并创建输出目录
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# 获取可用模型
available_models = get_available_models()
if not available_models:
    print("错误: 未找到任何模型参数文件")
    exit(1)

# 根据参数选择模型
model_found = False
for model_num, params_file in available_models:
    if model_num == args.model:
        model_type = model_num
        identified_params = load_identified_params(params_file)
        print(f"使用模型: Model {model_type}")
        print(f"参数文件: {params_file}")
        model_found = True
        break

if not model_found:
    print(f"错误: 未找到Model {args.model}的参数文件")
    print("可用的模型:")
    for model_num, params_file in available_models:
        print(f"  Model {model_num} - {params_file}")
    exit(1)

# ==================== 常量定义 ====================
M_PI = np.pi
args.predict_step = min(max(args.predict_step, 5), 20)
args.dt = min(max(args.dt, 0.05), 0.5)
args.cycle_time = min(max(args.cycle_time, 210), 3600)
args.loop_num = min(max(args.loop_num, 1), 5)
N = args.predict_step           # 预测步长
T = args.dt                     # 采样时间
motor_power = 500
alpha = args.alpha
Adaptive_flag = args.adaptive  # 使用命令行参数控制自适应功能
position_noise_mean = args.noise_mean      # 位置噪声均值
position_noise_std = args.noise_std      # 位置噪声标准差
loop_num = args.loop_num
tol = args.cycle_time*loop_num  # 仿真总时间s
t = np.linspace(0, 2*M_PI*loop_num, int(tol / T) + 1)  # 时间向量

# ==================== 生成参考轨迹 ====================
# 使用指定的轨迹参数
x_r1, y_r1, trajectory_name = get_trajectory_equations(args.trajectory, t)

x_r1 = x_r1[:, np.newaxis]
y_r1 = y_r1[:, np.newaxis]
path_points = np.concatenate([x_r1, y_r1], axis=1)
sim_steps = len(path_points)

print(f"轨迹点数量: {sim_steps}")
print(f"轨迹参数: {trajectory_name}")
print(f"自适应控制: {'启用' if Adaptive_flag else '禁用'}")
print(f"模型类型: Model {model_type}")

# ==================== 权重参数 ====================
if args.trajectory == 2 or args.trajectory == 1:
    Q_low = [0.5, 0.1, 0.1, 10, 10, 0.1]
    Q_high = [5, 1, 1, 100, 100, 1]
    R = [0.01, 0.01]
    F = [0.5, 0.1, 0.1, 100, 100, 0.1]
elif args.trajectory == 3:
    Q_low = [0.5, 0.1, 0.1, 60, 60, 0.1]      # q_j^min
    Q_high = [0.5, 0.1, 0.1, 900, 900, 0.1]    # q_j^max
    R = [0.002, 0.002]                     # 控制权重固定
    F = [0.5, 0.1, 0.1, 100, 100, 0.1]

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

# ==================== 使用识别的动力学模型 ====================
def build_dynamics_model(model_type, params, u, v, r, psi, Tp, Ts):
    """根据模型类型构建动力学模型"""
    
    if model_type == ModelType.MODEL_1:
        # 基础模型 (18参数)
        a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, b6, c1, c2, c3, c4, c5, c6 = params
        du_model = a1 * v * r + a2 * u + a3 * v + a4 * r + a5 * (Tp + Ts) + a6
        dv_model = b1 * u * r + b2 * u + b3 * v + b4 * r + b5 * (Tp - Ts) + b6
        dr_model = c1 * u * v + c2 * u + c3 * v + c4 * r + c5 * (Tp - Ts) + c6
        
        print(f"\n使用识别的模型参数 (Model 1 - 18参数):")
        print(f"du方程系数: a1={a1:.6f}, a2={a2:.6f}, a3={a3:.6f}, a4={a4:.6f}, a5={a5:.6f}, a6={a6:.6f}")
        print(f"dv方程系数: b1={b1:.6f}, b2={b2:.6f}, b3={b3:.6f}, b4={b4:.6f}, b5={b5:.6f}, b6={b6:.6f}")
        print(f"dr方程系数: c1={c1:.6f}, c2={c2:.6f}, c3={c3:.6f}, c4={c4:.6f}, c5={c5:.6f}, c6={c6:.6f}")
        
    elif model_type == ModelType.MODEL_2:
        # 分离模型 (21参数)
        a1, a2, a3, a4, a5, a6, a7, b1, b2, b3, b4, b5, b6, b7, c1, c2, c3, c4, c5, c6, c7 = params
        du_model = a1 * v * r + a2 * u + a3 * v + a4 * r + a5 * Tp + a6 * Ts + a7
        dv_model = b1 * u * r + b2 * u + b3 * v + b4 * r + b5 * Tp + b6 * Ts + b7
        dr_model = c1 * u * v + c2 * u + c3 * v + c4 * r + c5 * Tp + c6 * Ts + c7
        
        print(f"\n使用识别的模型参数 (Model 2 - 21参数):")
        print(f"du方程系数: a1={a1:.6f}, a2={a2:.6f}, a3={a3:.6f}, a4={a4:.6f}, a5={a5:.6f}, a6={a6:.6f}, a7={a7:.6f}")
        print(f"dv方程系数: b1={b1:.6f}, b2={b2:.6f}, b3={b3:.6f}, b4={b4:.6f}, b5={b5:.6f}, b6={b6:.6f}, b7={b7:.6f}")
        print(f"dr方程系数: c1={c1:.6f}, c2={c2:.6f}, c3={c3:.6f}, c4={c4:.6f}, c5={c5:.6f}, c6={c6:.6f}, c7={c7:.6f}")
        
    elif model_type == ModelType.MODEL_3:
        # 简化模型 (16参数)
        a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6 = params
        du_model = a1 * v * r + a2 * u + a3 * r + a4 + a5 * (Tp + Ts)
        dv_model = b1 * u * r + b2 * v + b3 * r + b4 + b5 * (Tp - Ts)
        dr_model = c1 * u * v + c2 * u + c3 * v + c4 * r + c5 * (Tp - Ts) + c6
        
        print(f"\n使用识别的模型参数 (Model 3 - 16参数):")
        print(f"du方程系数: a1={a1:.6f}, a2={a2:.6f}, a3={a3:.6f}, a4={a4:.6f}, a5={a5:.6f}")
        print(f"dv方程系数: b1={b1:.6f}, b2={b2:.6f}, b3={b3:.6f}, b4={b4:.6f}, b5={b5:.6f}")
        print(f"dr方程系数: c1={c1:.6f}, c2={c2:.6f}, c3={c3:.6f}, c4={c4:.6f}, c5={c5:.6f}, c6={c6:.6f}")
    
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 构建完整的动力学模型
    rhs = ca.vertcat(
        du_model,
        dv_model,
        dr_model,
        # 位置和航向角的运动学方程
        u * ca.cos(psi) - v * ca.sin(psi),
        u * ca.sin(psi) + v * ca.cos(psi),
        r
    )
    
    return rhs

# 构建识别的动力学模型
rhs = build_dynamics_model(model_type, identified_params, u, v, r, psi, Tp, Ts)

f = ca.Function('f', [state, control], [rhs])

# ==================== 初始状态 ====================
x0 = [0, 0, 0, path_points[0][0]+2, path_points[0][1]-2, 0]
xs_list = [[0, 0, 0, point[0], point[1], 0] for point in path_points]
state_history = [x0]
u0 = np.zeros(2 * N).tolist()
control_history = []

print(f"\n初始状态: {x0}")
print(f"目标轨迹起点: [{path_points[0][0]:.2f}, {path_points[0][1]:.2f}]")

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
lbg = [-3, -0.5, -0.3*M_PI] * (N+1)
ubg = [3, 0.5, 0.3*M_PI] * (N+1)

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

print("\n开始NMPC仿真...")

# ==================== NMPC 仿真 ====================
for sim in range(sim_steps):
    if sim % 500 == 0:
        print(f"仿真进度: {sim}/{sim_steps} ({sim/sim_steps*100:.1f}%)")
    
    xs = xs_list[sim] if sim < len(xs_list) else xs_list[-1]

    current_x, current_y = x0[3], x0[4]
    target_x, target_y = xs[3], xs[4]
    x0[5] = normalize_angle_diff(x0[5])
    current_psi = x0[5]
    target_psi = compute_target_psi(xs_list, sim)
    distance_error = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
    lateral_errors.append(distance_error)

    delta_psi = normalize_angle_diff(target_psi-current_psi)
    heading_errors.append(abs(delta_psi))
    Q_current = []
    if Adaptive_flag:
        ej = distance_error/100  # e_j = x_j - x_ref_j        
        for j in range(6):
            # ej = abs(state_error[j])
            qj = Q_low[j] + (Q_high[j] - Q_low[j]) * (1 - math.exp(-alpha * ej))
            Q_current.append(qj)
    else:
        for j in range(6):
            Q_current.append((Q_low[j] + Q_high[j]) / 2)
    R_current = R
    F_current = F
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
    noise = np.random.normal(position_noise_mean, position_noise_std, size=2)
    x_next[3] += noise[0]
    x_next[4] += noise[1]
    x0 = x_next.tolist()
    control_history.append(u0_step)
    state_history.append(x0)
    u0 = u_opt[2:] + u_opt[-2:]

print("仿真完成！")

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
print(f"\n=== 跟踪性能统计 ===")
print(f"横向误差: 平均={np.mean(lateral_errors):.4f}, 标准差={np.std(lateral_errors):.4f}")
print(f"航向误差: 平均={np.mean(heading_errors):.4f} rad, 标准差={np.std(heading_errors):.4f} rad")
print(f"最大横向误差: {np.max(lateral_errors):.4f}")
print(f"最大航向误差: {np.max(heading_errors):.4f} rad")

# ==================== 绘图 ====================
plt.figure(figsize=(10, 6))
plt.plot(ref_y, ref_x, 'r--', linewidth=2, label='Reference Path')
plt.plot(traj_y, traj_x, 'b-', linewidth=1.5, label='NMPC Trajectory (Identified Model)')
plt.scatter([traj_y[0]], [traj_x[0]], c='g', s=100, label='Start', zorder=5)
plt.scatter([traj_y[-1]], [traj_x[-1]], c='orange', s=100, label='End', zorder=5)
plt.xlabel('East Position (m)')
plt.ylabel('North Position (m)')
plt.title(f'NMPC Path Tracking with Identified Model (NED Coordinate)\n{trajectory_name}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

# 保存轨迹对比图
trajectory_plot_path = os.path.join(output_dir, f"nmpc_trajectory_{model_type}_for_trajectory_{args.trajectory}.png")
plt.savefig(trajectory_plot_path)

# ==================== 误差绘图 ====================
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(lateral_errors, label='Lateral Tracking Error', color='b', linewidth=1)
plt.axhline(y=np.mean(lateral_errors), color='b', linestyle='--', 
           label=f'Mean Lateral Error: {np.mean(lateral_errors):.4f}')
plt.ylabel('Lateral Error')
plt.title('Tracking Errors with Identified Model')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(heading_errors, label='Heading Angle Error (rad)', color='r', linewidth=1)
plt.axhline(y=np.mean(heading_errors), color='r', linestyle='--', 
           label=f'Mean Heading Error: {np.mean(heading_errors):.4f} rad')
plt.xlabel('Time Step')
plt.ylabel('Heading Error (rad)')
plt.legend()
plt.grid(True, alpha=0.3)

# 保存误差图
error_plot_path = os.path.join(output_dir, f"nmpc_error_{model_type}_for_trajectory_{args.trajectory}.png")
plt.savefig(error_plot_path)

# 可视化状态变量
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(t[:len(traj_u)], traj_u, label='u (surge velocity)', color='blue')
plt.ylabel('u (m/s)')
plt.title('State Variables with Identified Model')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(4, 1, 2)
plt.plot(t[:len(traj_v)], traj_v, label='v (sway velocity)', color='green')
plt.ylabel('v (m/s)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(4, 1, 3)
plt.plot(t[:len(traj_r)], traj_r, label='r (yaw rate)', color='red')
plt.ylabel('r (rad/s)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(4, 1, 4)
plt.plot(t[:len(traj_psi)], traj_psi, label='psi (heading angle)', color='purple')
plt.ylabel('psi (rad)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存状态变量图
state_plot_path = os.path.join(output_dir, f"nmpc_state_variables_{model_type}_for_trajectory_{args.trajectory}.png")
plt.savefig(state_plot_path)

# 推进器输出对比
plt.figure(figsize=(12, 6))
t_plot = t[:len(PWML)]
plt.subplot(2, 1, 1)
plt.plot(t_plot, PWML, label='Ts (Starboard Thruster)', color='blue')
plt.ylabel('Thrust (N)')
plt.title('Thruster Outputs with Identified Model')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(t_plot, PWMR, label='Tp (Port Thruster)', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Thrust (N)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存推进器输出图
thruster_plot_path = os.path.join(output_dir, f"nmpc_thruster_outputs_{model_type}_for_trajectory_{args.trajectory}.png")
plt.savefig(thruster_plot_path)

plt.show()

# ==================== 保存结果 ====================
# 确保所有数组长度一致
min_length = min(len(traj_x), len(traj_y), len(traj_u), len(traj_v), len(traj_r), len(traj_psi), 
                len(PWMR), len(PWML), len(lateral_errors), len(heading_errors))

output_data = {
    'Time': t[:min_length],
    'X': traj_x[:min_length],
    'Y': traj_y[:min_length],
    'Ref_X': ref_x[:min_length],
    'Ref_Y': ref_y[:min_length],
    'u': traj_u[:min_length],
    'v': traj_v[:min_length],
    'r': traj_r[:min_length],
    'psi': traj_psi[:min_length],
    'Tp': PWMR[:min_length],
    'Ts': PWML[:min_length],
    'Lateral_Error': lateral_errors[:min_length],
    'Heading_Error': heading_errors[:min_length]
}

df_output = pd.DataFrame(output_data)

output_path = os.path.join(output_dir, f"nmpc_identified_model_{model_type}_for_trajectory_{args.trajectory}_results.csv")
df_output.to_csv(output_path, index=False)

# 保存性能报告
model_type_names = {1: "基础模型 (18参数)", 2: "分离模型 (21参数)", 3: "简化模型 (16参数)"}
performance_report = f"""NMPC轨迹跟踪性能报告 - 使用识别模型
==========================================

轨迹参数:
- {trajectory_name}

模型信息:
- 模型类型: Model {model_type} - {model_type_names[model_type]}
- 参数文件: {params_file}
- 参数数量: {len(identified_params)}

跟踪性能:
- 平均横向误差: {np.mean(lateral_errors):.6f}
- 横向误差标准差: {np.std(lateral_errors):.6f}
- 最大横向误差: {np.max(lateral_errors):.6f}
- 平均航向误差: {np.mean(heading_errors):.6f} rad
- 航向误差标准差: {np.std(heading_errors):.6f} rad
- 最大航向误差: {np.max(heading_errors):.6f} rad

仿真参数:
- 预测步长: {N}
- 采样时间: {T} s
- 仿真步数: {sim_steps}
- 总仿真时间: {tol} s
- 自适应控制: {'启用' if Adaptive_flag else '禁用'}

识别的模型参数:
{identified_params}
"""

report_path = os.path.join(output_dir, f"nmpc_performance_model_{model_type}_for_trajectory_{args.trajectory}_report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(performance_report)

print(f"\n=== 结果保存 ===")
print(f"轨迹数据已保存至: {output_path}")
print(f"性能报告已保存至: {report_path}")
print(f"轨迹对比图已保存至: {trajectory_plot_path}")
print(f"误差图已保存至: {error_plot_path}")
print(f"状态变量图已保存至: {state_plot_path}")
print(f"推进器输出图已保存至: {thruster_plot_path}")

print(f"\n使用识别模型的NMPC轨迹跟踪测试完成！")