import numpy as np

class ModelEquations:
    """
    统一的模型方程类，支持三种船舶动力学模型
    
    模型类型:
    - model_1: 基础模型，使用组合推进器输入 (Tp+Ts) 和 (Tp-Ts)
    - model_2: 分离模型，分别使用 Tp 和 Ts 作为独立输入
    - model_3: 简化模型，减少了某些交叉项
    """
    
    # 模型类型常量
    MODEL_1 = 'model_1'
    MODEL_2 = 'model_2'
    MODEL_3 = 'model_3'
    
    AVAILABLE_MODELS = [MODEL_1, MODEL_2, MODEL_3]
    
    @staticmethod
    def model_equations(params, X, U, model_type='model_1'):
        """
        统一的模型方程定义
        
        参数:
            params: 模型参数
            X: 状态变量 [u, v, r]
            U: 控制输入 [Ts, Tp]
            model_type: 模型类型 ('model_1', 'model_2', 'model_3')
            
        返回:
            numpy.ndarray: 状态导数 [du, dv, dr]
        """
        u, v, r = X[:, 0], X[:, 1], X[:, 2]
        Ts, Tp = U[:, 0], U[:, 1]
        
        if model_type == ModelEquations.MODEL_1:
            # 基础模型 (18参数)
            a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, b6, c1, c2, c3, c4, c5, c6 = params
            du_model = a1 * v * r + a2 * u + a3 * v + a4 * r + a5 * (Tp + Ts) + a6
            dv_model = b1 * u * r + b2 * u + b3 * v + b4 * r + b5 * (Tp - Ts) + b6
            dr_model = c1 * u * v + c2 * u + c3 * v + c4 * r + c5 * (Tp - Ts) + c6
            
        elif model_type == ModelEquations.MODEL_2:
            # 分离模型 (21参数)
            a1, a2, a3, a4, a5, a6, a7, b1, b2, b3, b4, b5, b6, b7, c1, c2, c3, c4, c5, c6, c7 = params
            du_model = a1 * v * r + a2 * u + a3 * v + a4 * r + a5 * Tp + a6 * Ts + a7
            dv_model = b1 * u * r + b2 * u + b3 * v + b4 * r + b5 * Tp + b6 * Ts + b7
            dr_model = c1 * u * v + c2 * u + c3 * v + c4 * r + c5 * Tp + c6 * Ts + c7
            
        elif model_type == ModelEquations.MODEL_3:
            # 简化模型 (16参数)
            a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6 = params
            du_model = a1 * v * r + a2 * u + a3 * r + a4 + a5 * (Tp + Ts)
            dv_model = b1 * u * r + b2 * v + b3 * r + b4 + b5 * (Tp - Ts)
            dr_model = c1 * u * v + c2 * u + c3 * v + c4 * r + c5 * (Tp - Ts) + c6
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Available types: {ModelEquations.AVAILABLE_MODELS}")

        return np.column_stack((du_model, dv_model, dr_model))
    
    @staticmethod
    def loss_function(params, X, U, dX, model_type='model_1'):
        """
        损失函数（目标函数）
        
        参数:
            params: 模型参数
            X: 状态变量
            U: 控制输入
            dX: 实际状态导数
            model_type: 模型类型
            
        返回:
            float: 误差平方和
        """
        pred = ModelEquations.model_equations(params, X, U, model_type)
        error = pred - dX
        return np.sum(error ** 2)
    
    @staticmethod
    def get_initial_params(model_type='model_1'):
        """
        获取不同模型类型的初始参数猜测
        
        参数:
            model_type: 模型类型
            
        返回:
            numpy.ndarray: 初始参数
        """
        if model_type == ModelEquations.MODEL_1:
            # 基础模型 (18参数)
            return np.array([
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
                 0.0307,   # c3
                 0.0307,   # c4
                 0.0307,   # c5
                 0.6836    # c6
            ])
            
        elif model_type == ModelEquations.MODEL_2:
            # 分离模型 (21参数)
            return np.array([
                -1.1391,   # a1
                 0.0028,   # a2
                 0.6836,   # a3
                 0.6836,   # a4
                 0.6836,   # a5
                 0.6836,   # a6
                 0.6836,   # a7
                 0.0161,   # b1
                -0.0052,   # b2
                 0.002,    # b3
                 0.0068,   # b4
                 0.002,    # b5
                 0.0068,   # b6
                 0.0161,   # b7
                 8.2861,   # c1
                -0.9860,   # c2
                 0.0307,   # c3
                 1.3276,   # c4
                 0.0307,   # c5
                 1.3276,   # c6
                 0.001     # c7
            ])
            
        elif model_type == ModelEquations.MODEL_3:
            # 简化模型 (16参数)
            return np.array([
                -1.1391,   # a1
                 0.0028,   # a2
                 0.6836,   # a3
                 0.6836,   # a4
                 0.6836,   # a5
                 0.0161,   # b1
                -0.0052,   # b2
                 0.002,    # b3
                 0.6836,   # b4
                 0.6836,   # b5
                 8.2861,   # c1
                -0.9860,   # c2
                 0.0307,   # c3
                 0.0307,   # c4
                 0.0307,   # c5
                 0.6836    # c6
            ])
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Available types: {ModelEquations.AVAILABLE_MODELS}")
    
    @staticmethod
    def print_model_equations(params, model_type='model_1'):
        """
        打印不同类型的模型方程
        
        参数:
            params: 模型参数
            model_type: 模型类型
        """
        print(f"\nIdentified Model Equations ({model_type}):")
        
        if model_type == ModelEquations.MODEL_1:
            # 基础模型 (18参数)
            a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, b6, c1, c2, c3, c4, c5, c6 = params
            print(f"du = {a1:.6f} * v * r + {a2:.6f} * u + {a3:.6f} * v + {a4:.6f} * r + {a5:.6f} * (Tp + Ts) + {a6:.6f}")
            print(f"dv = {b1:.6f} * u * r + {b2:.6f} * u + {b3:.6f} * v + {b4:.6f} * r + {b5:.6f} * (Tp - Ts) + {b6:.6f}")
            print(f"dr = {c1:.6f} * u * v + {c2:.6f} * u + {c3:.6f} * v + {c4:.6f} * r + {c5:.6f} * (Tp - Ts) + {c6:.6f}")
            
        elif model_type == ModelEquations.MODEL_2:
            # 分离模型 (21参数)
            a1, a2, a3, a4, a5, a6, a7, b1, b2, b3, b4, b5, b6, b7, c1, c2, c3, c4, c5, c6, c7 = params
            print(f"du = {a1:.6f} * v * r + {a2:.6f} * u + {a3:.6f} * v + {a4:.6f} * r + {a5:.6f} * Tp + {a6:.6f} * Ts + {a7:.6f}")
            print(f"dv = {b1:.6f} * u * r + {b2:.6f} * u + {b3:.6f} * v + {b4:.6f} * r + {b5:.6f} * Tp + {b6:.6f} * Ts + {b7:.6f}")
            print(f"dr = {c1:.6f} * u * v + {c2:.6f} * u + {c3:.6f} * v + {c4:.6f} * r + {c5:.6f} * Tp + {c6:.6f} * Ts + {c7:.6f}")
            
        elif model_type == ModelEquations.MODEL_3:
            # 简化模型 (16参数)
            a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6 = params
            print(f"du = {a1:.6f} * v * r + {a2:.6f} * u + {a3:.6f} * r + {a4:.6f} + {a5:.6f} * (Tp + Ts)")
            print(f"dv = {b1:.6f} * u * r + {b2:.6f} * v + {b3:.6f} * r + {b4:.6f} + {b5:.6f} * (Tp - Ts)")
            print(f"dr = {c1:.6f} * u * v + {c2:.6f} * u + {c3:.6f} * v + {c4:.6f} * r + {c5:.6f} * (Tp - Ts) + {c6:.6f}")
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Available types: {ModelEquations.AVAILABLE_MODELS}")