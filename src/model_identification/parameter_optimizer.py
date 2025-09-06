from scipy.optimize import minimize
import numpy as np
from .model_equations import ModelEquations

class ParameterOptimizer:
    """
    增强的参数优化类，支持多种优化方法和模型类型
    
    支持的优化方法:
    - COBYLA: 约束优化线性逼近法
    - SLSQP: 序列最小二乘规划法
    - trust-constr: 信赖域约束优化法
    """
    
    # 支持的优化方法常量
    SLSQP = 'SLSQP'
    TRUST_CONSTR = 'trust-constr'
    
    AVAILABLE_METHODS = [SLSQP, TRUST_CONSTR]
    
    def __init__(self, X, U, dX, model_type='model_1', optimization_method='SLSQP', initial_params=None):
        """
        初始化参数优化器
        
        参数:
            X: 状态变量
            U: 控制输入
            dX: 状态导数
            model_type: 模型类型 ('model_1', 'model_2', 'model_3')
            optimization_method: 优化方法 ('SLSQP', 'trust-constr')
            initial_params: 初始参数猜测，如果为None则使用默认值
        """
        self.X = X
        self.U = U
        self.dX = dX
        self.model_type = model_type
        self.optimization_method = optimization_method
        self.initial_params = initial_params if initial_params is not None else ModelEquations.get_initial_params(model_type)
        self.result = None
        self.estimated_params = None
        
        # 验证输入参数
        if model_type not in ModelEquations.AVAILABLE_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}. Available types: {ModelEquations.AVAILABLE_MODELS}")
        if optimization_method not in self.AVAILABLE_METHODS:
            raise ValueError(f"Unsupported optimization method: {optimization_method}. Available methods: {self.AVAILABLE_METHODS}")
        
    def optimize(self):
        """
        执行参数优化
        
        返回:
            self: 返回自身实例以支持链式调用
        """
        print(f"Starting parameter optimization using {self.optimization_method} method for {self.model_type}...")
        
        # 设置优化选项
        options = {'disp': True}
        if self.optimization_method == self.TRUST_CONSTR:
            options['verbose'] = 1
        
        # 执行优化
        self.result = minimize(
            ModelEquations.loss_function, 
            self.initial_params, 
            args=(self.X, self.U, self.dX, self.model_type), 
            method=self.optimization_method, 
            options=options
        )
        
        # Always save the final parameters, even if optimization didn't fully converge
        self.estimated_params = self.result.x
        
        if self.result.success:
            print(f"Parameter optimization successful using {self.optimization_method}!")
            print(f"Final cost: {self.result.fun:.6f}")
            if hasattr(self.result, 'nit'):
                print(f"Number of iterations: {self.result.nit}")
        else:
            print(f"Parameter optimization failed using {self.optimization_method}.")
            print(f"Reason: {self.result.message}")
            print(f"Final cost: {self.result.fun:.6f}")
            print("Using best parameters found so far...")
            
        return self
    
    def get_estimated_params(self):
        """
        获取估计的参数
        
        返回:
            numpy.ndarray: 估计的参数
        """
        if self.estimated_params is None:
            raise ValueError("Parameters not optimized yet, please call optimize method first")
        return self.estimated_params
    
    def print_model_equations(self):
        """
        打印模型方程
        """
        if self.estimated_params is None:
            raise ValueError("Parameters not optimized yet, please call optimize method first")
        ModelEquations.print_model_equations(self.estimated_params, self.model_type)