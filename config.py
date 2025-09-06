#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一配置管理模块
支持模型1/2/3的配置管理
"""

import os
import json
from typing import Dict, Any, Optional

class ModelConfig:
    """模型配置管理类"""
    
    def __init__(self, model_type: int = 3):
        """
        初始化配置
        
        Args:
            model_type: 模型类型 (1, 2, 3)
        """
        self.model_type = model_type
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 模型配置
        self.model_configs = {
            1: {
                'name': 'Model 1 - Basic',
                'equations_module': 'src.model_identification.model_equations',
                'optimizer_module': 'src.model_identification.parameter_optimizer',
                'simulator_module': 'src.simulation_visualization.simulator',
                'params_file': 'model_1_params.json',
                'description': 'Basic dynamics model with fundamental equations'
            },
            2: {
                'name': 'Model 2 - Enhanced',
                'equations_module': 'src.model_identification.model_equations2',
                'optimizer_module': 'src.model_identification.parameter_optimizer2',
                'simulator_module': 'src.simulation_visualization.simulator2',
                'params_file': 'model_2_params.json',
                'description': 'Enhanced dynamics model with improved accuracy'
            },
            3: {
                'name': 'Model 3 - Advanced',
                'equations_module': 'src.model_identification.model_equations3',
                'optimizer_module': 'src.model_identification.parameter_optimizer3',
                'simulator_module': 'src.simulation_visualization.simulator3',
                'params_file': 'model_3_params.json',
                'description': 'Advanced dynamics model with comprehensive features'
            }
        }
        
        # 数据配置
        self.data_config = {
            'data_dir': os.path.join(self.base_dir, 'datas'),
            'circle_data': 'boat1_2_circle.xlsx',
            'sin_data': 'boat1_2_sin.xlsx',
            'default_data': 'boat1_2_circle.xlsx'
        }
        
        # 输出配置
        self.output_config = {
            'results_dir': os.path.join(self.base_dir, 'results'),
            'plots_dir': os.path.join(self.base_dir, 'plots'),
            'reports_dir': os.path.join(self.base_dir, 'reports')
        }
        
        # NMPC配置
        self.nmpc_config = {
            'tracking_dir': os.path.join(self.base_dir, 'nmpc_tracking'),
            'reference_path': os.path.join(self.base_dir, 'datas', 'boat1_2_circle.xlsx'),
            'control_params': {
                'N': 20,  # 预测步长
                'dt': 0.1,  # 采样时间
                'Q': [1, 1, 0.1],  # 状态权重
                'R': [0.01, 0.01],  # 控制权重
                'max_thrust': 500,  # 最大推力
                'min_thrust': -500  # 最小推力
            }
        }
        
    def get_model_config(self, model_type: Optional[int] = None) -> Dict[str, Any]:
        """获取指定模型的配置"""
        if model_type is None:
            model_type = self.model_type
            
        if model_type not in self.model_configs:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        return self.model_configs[model_type]
    
    def get_data_path(self, data_name: str = 'default') -> str:
        """获取数据文件路径"""
        if data_name == 'default':
            data_name = self.data_config['default_data']
        elif data_name == 'circle':
            data_name = self.data_config['circle_data']
        elif data_name == 'sin':
            data_name = self.data_config['sin_data']
            
        return os.path.join(self.data_config['data_dir'], data_name)
    
    def get_params_path(self, model_type: Optional[int] = None) -> str:
        """获取模型参数文件路径"""
        if model_type is None:
            model_type = self.model_type
            
        config = self.get_model_config(model_type)
        return os.path.join(self.base_dir, config['params_file'])
    
    def ensure_output_dirs(self):
        """确保输出目录存在"""
        for dir_path in self.output_config.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def save_config(self, filepath: str):
        """保存配置到文件"""
        config_data = {
            'model_type': self.model_type,
            'model_configs': self.model_configs,
            'data_config': self.data_config,
            'output_config': self.output_config,
            'nmpc_config': self.nmpc_config
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    def load_config(self, filepath: str):
        """从文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            
        self.model_type = config_data.get('model_type', self.model_type)
        self.model_configs.update(config_data.get('model_configs', {}))
        self.data_config.update(config_data.get('data_config', {}))
        self.output_config.update(config_data.get('output_config', {}))
        self.nmpc_config.update(config_data.get('nmpc_config', {}))
    
    def print_config(self):
        """打印当前配置"""
        print(f"\n=== Model Configuration ===")
        print(f"Current Model Type: {self.model_type}")
        
        current_config = self.get_model_config()
        print(f"Model Name: {current_config['name']}")
        print(f"Description: {current_config['description']}")
        print(f"Parameters File: {current_config['params_file']}")
        
        print(f"\n=== Data Configuration ===")
        print(f"Data Directory: {self.data_config['data_dir']}")
        print(f"Default Data: {self.data_config['default_data']}")
        
        print(f"\n=== NMPC Configuration ===")
        print(f"Prediction Horizon: {self.nmpc_config['control_params']['N']}")
        print(f"Sampling Time: {self.nmpc_config['control_params']['dt']}")
        print(f"State Weights: {self.nmpc_config['control_params']['Q']}")
        print(f"Control Weights: {self.nmpc_config['control_params']['R']}")


# 全局配置实例
config = ModelConfig()

# 便捷函数
def get_config(model_type: int = 3) -> ModelConfig:
    """获取配置实例"""
    return ModelConfig(model_type)

def set_model_type(model_type: int):
    """设置全局模型类型"""
    global config
    config.model_type = model_type

def get_model_config(model_type: Optional[int] = None) -> Dict[str, Any]:
    """获取模型配置"""
    return config.get_model_config(model_type)

def get_data_path(data_name: str = 'default') -> str:
    """获取数据路径"""
    return config.get_data_path(data_name)

def get_params_path(model_type: Optional[int] = None) -> str:
    """获取参数文件路径"""
    return config.get_params_path(model_type)


if __name__ == "__main__":
    # 测试配置管理
    print("=== Configuration Management Test ===")
    
    # 测试所有模型类型
    for model_type in [1, 2, 3]:
        print(f"\n--- Model {model_type} Configuration ---")
        test_config = get_config(model_type)
        test_config.print_config()
    
    # 测试路径获取
    print(f"\n--- Path Testing ---")
    print(f"Circle data path: {get_data_path('circle')}")
    print(f"Sin data path: {get_data_path('sin')}")
    print(f"Model 1 params: {get_params_path(1)}")
    print(f"Model 2 params: {get_params_path(2)}")
    print(f"Model 3 params: {get_params_path(3)}")