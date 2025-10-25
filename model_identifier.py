import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import json

# Add project root directory to system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from src.data_processing.data_loader import DataLoader
from src.data_processing.data_preprocessor import DataPreprocessor
from src.model_identification.model_equations import ModelEquations
from src.model_identification.parameter_optimizer import ParameterOptimizer
from src.simulation_visualization.simulator import Simulator
from src.simulation_visualization.visualizer import Visualizer

# Note: Using unified ModelEquations and ParameterOptimizer classes
# The old separate model classes (ModelEquations2, ModelEquations3, etc.) are now integrated
# into the unified ModelEquations class with model_type parameter

# Import unified data format and simplified visualization modules
from src.utils.data_format import UnifiedDataFormat, DataExporter
# SimplifiedVisualizer will be imported locally when needed for supplementary metrics

def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Ship Model Identification System')
    parser.add_argument('--model', type=str, default='model_1', choices=['model_1', 'model_2', 'model_3'],
                        help='Select model type: model_1=Standard model, model_2=Separated thruster input model, model_3=Simplified model')
    parser.add_argument('--data', type=str, default='datas/boat1_2_sin.xlsx',
                        help='Data file path: "datas/boat1_2_sin.xlsx" is Left and right differentials data,"datas/boat1_2_circle.xlsx" is Single Turn left ')
    parser.add_argument('--filter', type=str, default='savgol', 
                        choices=['savgol', 'ekf', 'lowpass', 'none'],
                        help='Data processing filter method: savgol=Savitzky-Golay filter, ekf=Extended Kalman filter, lowpass=Low-pass filter, none=No filter')
    parser.add_argument('--optimizer', type=str, choices=['SLSQP', 'trust-constr'], default='SLSQP',
                        help='Optimization method: SLSQP=Sequential Least SQuares Programming, trust-constr=Trust Region Constrained')
    parser.add_argument('--interactive', action='store_true',
                        help='Enable interactive mode with user-friendly interface')
    parser.add_argument('--start_row', type=int, default=0,
                        help='Data start row (default: 0)')
    parser.add_argument('--row_count', type=int, default=1500,
                        help='Number of data rows to read (default: 1500)')
    parser.add_argument('--output_dir', type=str, default='model_results/',
                        help='Output file directory (default: current directory)')
    return parser.parse_args()

def interactive_mode():
    """
    Interactive mode with user-friendly interface
    
    Returns:
        dict: User selected configuration
    """
    print("\n=== Ship Model Identification System - Interactive Mode ===")
    
    # Model selection
    print("\nPlease select model type:")
    print("1. Standard model (model_1)")
    print("2. Separated thruster input model (model_2)")
    print("3. Simplified model (model_3)")
    model_types = ['model_1', 'model_2', 'model_3']
    while True:
        try:
            model_choice = int(input("Please enter model number (1-3): "))
            if model_choice in [1, 2, 3]:
                model_type = model_types[model_choice - 1]
                break
            else:
                print("Please enter a valid model number (1-3)")
        except ValueError:
            print("Please enter a valid number")
    
    # Data file selection
    print("\nAvailable data files:")
    import glob
    data_files = glob.glob("datas/*.xlsx") + glob.glob("datas/*.csv")
    if data_files:
        for i, file in enumerate(data_files, 1):
            print(f"{i}. {file}")
        print(f"{len(data_files) + 1}. Custom file path")
        
        while True:
            try:
                file_choice = int(input(f"Please select data file (1-{len(data_files) + 1}): "))
                if 1 <= file_choice <= len(data_files):
                    data_file = data_files[file_choice - 1]
                    break
                elif file_choice == len(data_files) + 1:
                    data_file = input("Please enter data file path: ")
                    break
                else:
                    print(f"Please enter a valid file number (1-{len(data_files) + 1})")
            except ValueError:
                print("Please enter a valid number")
    else:
        data_file = input("Please enter data file path: ")
    
    # Filter method selection
    print("\nPlease select data processing filter method:")
    print("1. Savitzky-Golay smoothing filter (recommended)")
    print("2. Extended Kalman filter")
    print("3. Low-pass filter")
    print("4. No filter")
    filter_methods = ['savgol', 'ekf', 'lowpass', 'none']
    while True:
        try:
            filter_choice = int(input("Please enter filter method number (1-4): "))
            if 1 <= filter_choice <= 4:
                filter_method = filter_methods[filter_choice - 1]
                break
            else:
                print("Please enter a valid filter method number (1-4)")
        except ValueError:
            print("Please enter a valid number")
    
    # Optimization method selection
    print("\nPlease select optimization method:")
    print("1. SLSQP - Sequential Least Squares Programming (recommended)")
    print("2. trust-constr - Trust Region Constrained")
    optimization_methods = ['SLSQP', 'trust-constr']
    while True:
        try:
            opt_choice = int(input("Please enter optimization method number (1-2): "))
            if 1 <= opt_choice <= 2:
                optimization_method = optimization_methods[opt_choice - 1]
                break
            else:
                print("Please enter a valid optimization method number (1-2)")
        except ValueError:
            print("Please enter a valid number")
    
    # Data range settings
    print("\nData range settings:")
    start_row = int(input("Start row (default 0): ") or "0")
    row_count = int(input("Number of rows to read (default 1500): ") or "1500")
    
    return {
        'model': model_type,
        'data': data_file,
        'filter': filter_method,
        'optimizer': optimization_method,
        'start_row': start_row,
        'row_count': row_count
    }

def validate_data_file(file_path):
    """
    Validate if data file exists and has correct format
    
    Args:
        file_path (str): Data file path
        
    Returns:
        bool: Whether the file is valid
    """
    import os
    
    if not os.path.exists(file_path):
        print(f"Error: Data file '{file_path}' does not exist")
        return False
    
    if not (file_path.endswith('.xlsx') or file_path.endswith('.csv')):
        print(f"Error: Unsupported file format, please use .xlsx or .csv files")
        return False
    
    try:
        # Try to read file header to validate format
        if file_path.endswith('.xlsx'):
            import pandas as pd
            df = pd.read_excel(file_path, nrows=5)
        else:
            import pandas as pd
            df = pd.read_csv(file_path, nrows=5)
        
        print(f"Data file validation successful: {file_path}")
        print(f"File contains {len(df.columns)} columns of data")
        return True
    except Exception as e:
        print(f"Error: Cannot read data file '{file_path}': {str(e)}")
        return False

def main():
    """
    Main function that integrates all modules and runs the system
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # If interactive mode is enabled
    if args.interactive:
        config = interactive_mode()
        data_file = config['data']
        model_type = config['model']
        filter_method = config['filter']
        optimization_method = config['optimizer']
        start_row = config['start_row']
        row_count = config['row_count']
        output_dir = args.output_dir
    else:
        # Use command line arguments
        data_file = args.data
        model_type = args.model
        filter_method = args.filter
        optimization_method = args.optimizer
        start_row = args.start_row
        row_count = args.row_count
        output_dir = args.output_dir
    
    # Validate data file
    if not validate_data_file(data_file):
        print("Program exit")
        return
    
    # Display selected configuration
    model_names = {
        'model_1': "Standard Model",
        'model_2': "Separated Thruster Input Model",
        'model_3': "Simplified Model"
    }
    filter_names = {
        'savgol': 'Savitzky-Golay Smoothing Filter',
        'ekf': 'Extended Kalman Filter',
        'lowpass': 'Low-pass Filter',
        'none': 'No Filter'
    }
    
    # 优化方法映射
    optimizer_names = {
        'SLSQP': 'Sequential Least Squares Programming',
        'trust-constr': 'Trust Region Constrained'
    }
    
    print(f"\n=== Configuration Information ===")
    print(f"Model type: {model_names[model_type]}")
    print(f"Data file: {data_file}")
    print(f"Filter method: {filter_names[filter_method]}")
    print(f"Optimization method: {optimizer_names[optimization_method]}")
    print(f"Data range: Starting from row {start_row}, total {row_count} rows")
    print(f"Output directory: {output_dir}")
    
    # 1. 数据加载
    print("\n1. 加载数据...")
    data_loader = DataLoader(data_file, start_row=start_row, row_count=row_count)
    timestamp, x, y, psi, Ts, Tp, dt = data_loader.load_data().get_data()
    
    # 2. 数据预处理
    print("2. 数据预处理...")
    preprocessor = DataPreprocessor(x, y, psi, Ts, Tp, dt, filter_method=filter_method)
    u, v, r, X, U, U_scaled, dX, U_scaler = preprocessor.preprocess().get_processed_data()
    
    # 3. 参数优化
    print("3. Parameter optimization...")
    N_samples = len(x)
    X0 = [u[0], v[0], r[0]]
    x0, y0, psi0 = x[0], y[0], psi[0]
    
    # 使用统一的参数优化器
    optimizer = ParameterOptimizer(X, U_scaled, dX, model_type=model_type, optimization_method=optimization_method)
    optimizer.optimize()
    estimated_params = optimizer.get_estimated_params()
    
    # Print model equations
    optimizer.print_model_equations()
    
    # 4. Simulation
    print("4. Simulation...")
    simulator = Simulator(estimated_params, X0, U_scaled, x0, y0, psi0, dt, N_samples, model_type=model_type)
    simulator.simulate()
    x_sim, y_sim, psi_sim, u_sim, v_sim, r_sim, du_est, dv_est, dr_est = simulator.get_simulation_results()
    
    # Model type 3 is already handled above
    
    # 5. Data export and visualization
    print("5. Data export and visualization...")
    time = np.arange(N_samples) * dt
    
    # 使用统一数据格式导出结果
    data_format = UnifiedDataFormat()
    
    # 导出仿真结果
    simulation_data = DataExporter.export_simulation_results(
        time_array=time,
        states=np.column_stack([u_sim, v_sim, r_sim, x_sim, y_sim, psi_sim]),
        controls=np.column_stack([Ts, Tp]),
        reference_trajectory=np.column_stack([x, y]),
        errors=np.column_stack([x_sim - x, y_sim - y]),
        filepath=os.path.join(output_dir, f"{model_type}_results.csv")
    )    
    
    # 保存元数据
    metadata = {
        'model_type': model_type,
        'model_name': model_names[model_type],
        'filter_method': filter_method,
        'filter_name': filter_names[filter_method],
        'data_file': data_file,
        'data_range': f'{start_row}-{start_row + row_count}',
        'parameters': estimated_params.tolist(),
        'performance_metrics': {
            'rmse_u': np.sqrt(np.mean((u[:len(u_sim)] - u_sim)**2)),
            'rmse_v': np.sqrt(np.mean((v[:len(v_sim)] - v_sim)**2)),
            'rmse_r': np.sqrt(np.mean((r[:len(r_sim)] - r_sim)**2))
        }
    }
    metadata_filename = os.path.join(output_dir, f"{model_type}_identification_metadata.json")
    data_format.save_metadata(metadata, metadata_filename)
    
    # Extract data for visualization (using correct column names from DataExporter)
    time_data = simulation_data['Time']  # 标准列名是'Time'
    u_data = u  # 使用原始数据
    v_data = v
    r_data = r
    psi_data = psi
    x_data = x
    y_data = y
    Ts_data = Ts
    Tp_data = Tp
    u_sim_data = u_sim  # 使用仿真结果
    v_sim_data = v_sim
    r_sim_data = r_sim
    psi_sim_data = psi_sim
    x_sim_data = x_sim
    y_sim_data = y_sim
    
    # Calculate derivatives for visualization
    du = np.gradient(u_data, time_data)
    dv = np.gradient(v_data, time_data)
    dr = np.gradient(r_data, time_data)
    du_est = np.gradient(u_sim_data, time_data)
    dv_est = np.gradient(v_sim_data, time_data)
    dr_est = np.gradient(r_sim_data, time_data)
    
    # Create visualizer with focus on tracking performance and state comparison
    visualizer = Visualizer(time_data, u_data, v_data, r_data, psi_data, x_data, y_data, Ts_data, Tp_data,
                           u_sim_data, v_sim_data, r_sim_data, psi_sim_data, x_sim_data, y_sim_data,
                           du, dv, dr, du_est, dv_est, dr_est)
    
    # Plot state variables comparison (main focus)
    fig1 = plt.figure(figsize=(12, 10))
    plt.subplot(4, 1, 1)
    plt.plot(time_data, u_data, label='True u', linewidth=2)
    plt.plot(time_data, u_sim_data, '--', label='Estimated u', linewidth=2)
    plt.legend()
    plt.title('u: True vs Estimated')
    plt.ylabel('u (m/s)')
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(time_data, v_data, label='True v', linewidth=2)
    plt.plot(time_data, v_sim_data, '--', label='Estimated v', linewidth=2)
    plt.legend()
    plt.title('v: True vs Estimated')
    plt.ylabel('v (m/s)')
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(time_data, r_data, label='True r', linewidth=2)
    plt.plot(time_data, r_sim_data, '--', label='Estimated r', linewidth=2)
    plt.legend()
    plt.title('r: True vs Estimated')
    plt.ylabel('r (rad/s)')
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(time_data, np.degrees(psi_data), label='True psi', linewidth=2)
    plt.plot(time_data, np.degrees(psi_sim_data), '--', label='Estimated psi', linewidth=2)
    plt.legend()
    plt.title('psi: True vs Estimated')
    plt.ylabel('psi (degrees)')
    plt.xlabel('Time (s)')
    plt.grid(True)
    
    plt.tight_layout()
    fig1.suptitle(f'{model_names[model_type]} - State Variables Tracking Performance', fontsize=16, y=0.98)
    
    # Plot trajectory comparison with NED coordinate system (North-East-Down)
    fig2 = plt.figure(figsize=(10, 10))
    plt.plot(y_data, x_data, label='True Trajectory', linewidth=2)  # y为东向(East), x为北向(North)
    plt.plot(y_sim_data, x_sim_data, '--', label='Estimated Trajectory', linewidth=2)
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.legend()
    plt.title(f'{model_names[model_type]} - Trajectory Tracking Performance (NED Coordinate System)')
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    
    # Plot control inputs
    fig3 = plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_data, Ts_data, label='Right Propeller (Ts)', color='blue', linewidth=2)
    plt.legend()
    plt.title('Right Propeller Output')
    plt.ylabel('PWM Value (Ts)')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(time_data, Tp_data, label='Left Propeller (Tp)', color='red', linewidth=2)
    plt.legend()
    plt.title('Left Propeller Output')
    plt.ylabel('PWM Value (Tp)')
    plt.xlabel('Time (s)')
    plt.grid(True)
    
    plt.tight_layout()
    fig3.suptitle(f'{model_names[model_type]} - Control Input Analysis', fontsize=16, y=0.98)
    
    # Plot performance metrics as supplementary information
    from src.utils.simplified_visualizer import SimplifiedVisualizer
    simple_viz = SimplifiedVisualizer()
    fig4, performance_metrics = simple_viz.plot_performance_summary(simulation_data, model_names[model_type])
    fig4.suptitle(f'{model_names[model_type]} - Performance Metrics (Supplementary)', fontsize=16)
    
    # Save plots with descriptive names
    plot1_filename = os.path.join(output_dir, f"{model_type}_state_variables_tracking.png")
    plot2_filename = os.path.join(output_dir, f"{model_type}_trajectory_tracking_ned.png")
    plot3_filename = os.path.join(output_dir, f"{model_type}_control_inputs.png")
    plot4_filename = os.path.join(output_dir, f"{model_type}_performance_metrics.png")
    
    fig1.savefig(plot1_filename, dpi=300, bbox_inches='tight')
    fig2.savefig(plot2_filename, dpi=300, bbox_inches='tight')
    fig3.savefig(plot3_filename, dpi=300, bbox_inches='tight')
    fig4.savefig(plot4_filename, dpi=300, bbox_inches='tight')
    
    print(f"Plots saved:")
    print(f"  - State variables: {plot1_filename}")
    print(f"  - Trajectory (NED): {plot2_filename}")
    print(f"  - Control inputs: {plot3_filename}")
    print(f"  - Performance metrics: {plot4_filename}")
    
    plt.show()
    print("\nShip model identification completed!")

if __name__ == "__main__":
    main()