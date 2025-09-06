#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据格式验证和转换工具

提供数据格式验证、转换和质量检查功能，确保数据的一致性和完整性。

作者: 船舶动力学建模团队
日期: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
from pathlib import Path
import json

class DataValidator:
    """
    数据格式验证器
    
    提供数据格式验证、质量检查和转换功能。
    """
    
    def __init__(self):
        """初始化数据验证器"""
        self.required_columns = [
            'time', 'u', 'v', 'r', 'x', 'y', 'psi',
            'Ts', 'Tp', 'u_ref', 'v_ref', 'r_ref',
            'x_ref', 'y_ref', 'psi_ref',
            'error_x', 'error_y', 'error_psi'
        ]
        
        self.optional_columns = [
            'error_u', 'error_v', 'error_r',
            'control_effort', 'tracking_error_norm'
        ]
        
        self.data_types = {
            'time': 'float64',
            'u': 'float64', 'v': 'float64', 'r': 'float64',
            'x': 'float64', 'y': 'float64', 'psi': 'float64',
            'Ts': 'float64', 'Tp': 'float64',
            'u_ref': 'float64', 'v_ref': 'float64', 'r_ref': 'float64',
            'x_ref': 'float64', 'y_ref': 'float64', 'psi_ref': 'float64',
            'error_x': 'float64', 'error_y': 'float64', 'error_psi': 'float64'
        }
    
    def validate_dataframe(self, df: pd.DataFrame, strict: bool = True) -> Dict[str, Any]:
        """
        验证DataFrame格式
        
        参数:
            df: 待验证的DataFrame
            strict: 是否严格模式（要求所有必需列都存在）
            
        返回:
            验证结果字典
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'missing_columns': [],
            'extra_columns': [],
            'data_quality': {}
        }
        
        # 检查必需列
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            validation_result['missing_columns'] = missing_cols
            if strict:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"缺少必需列: {missing_cols}")
            else:
                validation_result['warnings'].append(f"缺少推荐列: {missing_cols}")
        
        # 检查额外列
        all_expected_cols = self.required_columns + self.optional_columns
        extra_cols = [col for col in df.columns if col not in all_expected_cols]
        if extra_cols:
            validation_result['extra_columns'] = extra_cols
            validation_result['warnings'].append(f"发现额外列: {extra_cols}")
        
        # 检查数据类型
        for col in df.columns:
            if col in self.data_types:
                expected_type = self.data_types[col]
                if df[col].dtype != expected_type:
                    try:
                        df[col] = df[col].astype(expected_type)
                        validation_result['warnings'].append(f"列 {col} 已转换为 {expected_type}")
                    except (ValueError, TypeError):
                        validation_result['errors'].append(f"列 {col} 无法转换为 {expected_type}")
                        validation_result['is_valid'] = False
        
        # 数据质量检查
        validation_result['data_quality'] = self._check_data_quality(df)
        
        return validation_result
    
    def _check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        检查数据质量
        
        参数:
            df: 待检查的DataFrame
            
        返回:
            数据质量报告
        """
        quality_report = {
            'total_rows': len(df),
            'missing_values': {},
            'infinite_values': {},
            'outliers': {},
            'time_consistency': {},
            'physical_constraints': {}
        }
        
        # 检查缺失值
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                quality_report['missing_values'][col] = {
                    'count': int(missing_count),
                    'percentage': float(missing_count / len(df) * 100)
                }
        
        # 检查无穷大值
        for col in df.select_dtypes(include=[np.number]).columns:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                quality_report['infinite_values'][col] = {
                    'count': int(inf_count),
                    'percentage': float(inf_count / len(df) * 100)
                }
        
        # 检查时间一致性
        if 'time' in df.columns:
            time_diff = df['time'].diff().dropna()
            quality_report['time_consistency'] = {
                'is_monotonic': bool(df['time'].is_monotonic_increasing),
                'mean_dt': float(time_diff.mean()),
                'std_dt': float(time_diff.std()),
                'min_dt': float(time_diff.min()),
                'max_dt': float(time_diff.max())
            }
        
        # 检查物理约束
        physical_checks = self._check_physical_constraints(df)
        quality_report['physical_constraints'] = physical_checks
        
        return quality_report
    
    def _check_physical_constraints(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        检查物理约束
        
        参数:
            df: 待检查的DataFrame
            
        返回:
            物理约束检查结果
        """
        constraints = {}
        
        # 速度合理性检查
        velocity_cols = ['u', 'v', 'r']
        for col in velocity_cols:
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > 0:
                    constraints[f'{col}_range'] = {
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'mean': float(values.mean()),
                        'std': float(values.std())
                    }
                    
                    # 检查异常值（3σ原则）
                    mean_val = values.mean()
                    std_val = values.std()
                    outliers = values[(values < mean_val - 3*std_val) | (values > mean_val + 3*std_val)]
                    if len(outliers) > 0:
                        constraints[f'{col}_outliers'] = {
                            'count': len(outliers),
                            'percentage': float(len(outliers) / len(values) * 100)
                        }
        
        # 控制输入合理性检查
        control_cols = ['Ts', 'Tp']
        for col in control_cols:
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > 0:
                    constraints[f'{col}_saturation'] = {
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'saturation_percentage': float((np.abs(values) > 0.95).sum() / len(values) * 100)
                    }
        
        return constraints
    
    def convert_legacy_format(self, data: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        转换旧格式数据为标准格式
        
        参数:
            data: 旧格式数据字典
            
        返回:
            标准格式DataFrame
        """
        # 创建空的标准格式DataFrame
        from .data_format import UnifiedDataFormat
        data_format = UnifiedDataFormat()
        
        # 确定数据长度
        data_length = 0
        for key, value in data.items():
            if isinstance(value, (list, np.ndarray)):
                data_length = max(data_length, len(value))
        
        df = data_format.create_empty_dataframe(data_length)
        
        # 映射旧格式到新格式
        column_mapping = {
            't': 'time', 'time': 'time',
            'u_actual': 'u', 'v_actual': 'v', 'r_actual': 'r',
            'x_actual': 'x', 'y_actual': 'y', 'psi_actual': 'psi',
            'u_reference': 'u_ref', 'v_reference': 'v_ref', 'r_reference': 'r_ref',
            'x_reference': 'x_ref', 'y_reference': 'y_ref', 'psi_reference': 'psi_ref',
            'thrust_port': 'Tp', 'thrust_starboard': 'Ts',
            'control_1': 'Ts', 'control_2': 'Tp'
        }
        
        # 填充数据
        for old_key, new_key in column_mapping.items():
            if old_key in data and new_key in df.columns:
                values = np.array(data[old_key])
                if len(values) == len(df):
                    df[new_key] = values
                elif len(values) < len(df):
                    # 如果数据长度不足，用NaN填充
                    df.loc[:len(values)-1, new_key] = values
        
        # 计算误差（如果参考值和实际值都存在）
        error_mappings = [
            ('x', 'x_ref', 'error_x'),
            ('y', 'y_ref', 'error_y'),
            ('psi', 'psi_ref', 'error_psi'),
            ('u', 'u_ref', 'error_u'),
            ('v', 'v_ref', 'error_v'),
            ('r', 'r_ref', 'error_r')
        ]
        
        for actual_col, ref_col, error_col in error_mappings:
            if actual_col in df.columns and ref_col in df.columns and error_col in df.columns:
                df[error_col] = df[actual_col] - df[ref_col]
        
        return df
    
    def repair_data(self, df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
        """
        修复数据问题
        
        参数:
            df: 待修复的DataFrame
            method: 修复方法 ('interpolate', 'forward_fill', 'drop')
            
        返回:
            修复后的DataFrame
        """
        df_repaired = df.copy()
        
        # 处理无穷大值
        for col in df_repaired.select_dtypes(include=[np.number]).columns:
            inf_mask = np.isinf(df_repaired[col])
            if inf_mask.any():
                warnings.warn(f"列 {col} 包含无穷大值，将被替换为NaN")
                df_repaired.loc[inf_mask, col] = np.nan
        
        # 处理缺失值
        if method == 'interpolate':
            # 线性插值
            df_repaired = df_repaired.interpolate(method='linear', limit_direction='both')
        elif method == 'forward_fill':
            # 前向填充
            df_repaired = df_repaired.fillna(method='ffill')
        elif method == 'drop':
            # 删除包含NaN的行
            df_repaired = df_repaired.dropna()
        
        return df_repaired
    
    def generate_validation_report(self, df: pd.DataFrame, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        生成详细的验证报告
        
        参数:
            df: 待验证的DataFrame
            output_path: 报告输出路径（可选）
            
        返回:
            验证报告字典
        """
        validation_result = self.validate_dataframe(df, strict=False)
        
        # 添加统计信息
        stats = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            values = df[col].dropna()
            if len(values) > 0:
                stats[col] = {
                    'count': len(values),
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'q25': float(values.quantile(0.25)),
                    'q50': float(values.quantile(0.50)),
                    'q75': float(values.quantile(0.75))
                }
        
        validation_result['statistics'] = stats
        
        # 保存报告
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(validation_result, f, indent=2, ensure_ascii=False)
        
        return validation_result

class DataConverter:
    """
    数据格式转换器
    
    提供不同数据格式之间的转换功能。
    """
    
    def __init__(self):
        """初始化数据转换器"""
        self.validator = DataValidator()
    
    def matlab_to_standard(self, matlab_data: Dict[str, Any]) -> pd.DataFrame:
        """
        将MATLAB格式数据转换为标准格式
        
        参数:
            matlab_data: MATLAB数据字典
            
        返回:
            标准格式DataFrame
        """
        # MATLAB数据通常以结构体形式存储
        converted_data = {}
        
        # 常见的MATLAB变量名映射
        matlab_mapping = {
            't': 'time',
            'u': 'u', 'v': 'v', 'r': 'r',
            'x': 'x', 'y': 'y', 'psi': 'psi',
            'u_ref': 'u_ref', 'v_ref': 'v_ref', 'r_ref': 'r_ref',
            'x_ref': 'x_ref', 'y_ref': 'y_ref', 'psi_ref': 'psi_ref',
            'Tp': 'Tp', 'Ts': 'Ts'
        }
        
        for matlab_key, std_key in matlab_mapping.items():
            if matlab_key in matlab_data:
                converted_data[std_key] = np.array(matlab_data[matlab_key]).flatten()
        
        return self.validator.convert_legacy_format(converted_data)
    
    def excel_to_standard(self, excel_path: str, sheet_name: str = None) -> pd.DataFrame:
        """
        将Excel文件转换为标准格式
        
        参数:
            excel_path: Excel文件路径
            sheet_name: 工作表名称（可选）
            
        返回:
            标准格式DataFrame
        """
        # 读取Excel文件
        df_excel = pd.read_excel(excel_path, sheet_name=sheet_name)
        
        # 常见的Excel列名映射
        excel_mapping = {
            'Time': 'time', 'time': 'time', 't': 'time',
            'U': 'u', 'u': 'u', 'surge': 'u',
            'V': 'v', 'v': 'v', 'sway': 'v',
            'R': 'r', 'r': 'r', 'yaw_rate': 'r',
            'X': 'x', 'x': 'x', 'position_x': 'x',
            'Y': 'y', 'y': 'y', 'position_y': 'y',
            'Psi': 'psi', 'psi': 'psi', 'heading': 'psi',
            'Thrust_Port': 'Tp', 'Tp': 'Tp', 'thrust_port': 'Tp',
            'Thrust_Starboard': 'Ts', 'Ts': 'Ts', 'thrust_starboard': 'Ts'
        }
        
        # 重命名列
        df_renamed = df_excel.rename(columns=excel_mapping)
        
        # 转换为标准格式
        data_dict = {col: df_renamed[col].values for col in df_renamed.columns}
        return self.validator.convert_legacy_format(data_dict)
    
    def csv_to_standard(self, csv_path: str, delimiter: str = ',') -> pd.DataFrame:
        """
        将CSV文件转换为标准格式
        
        参数:
            csv_path: CSV文件路径
            delimiter: 分隔符
            
        返回:
            标准格式DataFrame
        """
        # 读取CSV文件
        df_csv = pd.read_csv(csv_path, delimiter=delimiter)
        
        # 使用与Excel相同的映射规则
        return self.excel_to_standard(csv_path)

def validate_simulation_data(data_path: str, output_report: bool = True) -> Dict[str, Any]:
    """
    验证仿真数据的便捷函数
    
    参数:
        data_path: 数据文件路径
        output_report: 是否输出详细报告
        
    返回:
        验证结果
    """
    validator = DataValidator()
    
    # 根据文件扩展名选择读取方法
    file_path = Path(data_path)
    if file_path.suffix.lower() == '.csv':
        df = pd.read_csv(data_path)
    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(data_path)
    else:
        raise ValueError(f"不支持的文件格式: {file_path.suffix}")
    
    # 执行验证
    result = validator.validate_dataframe(df)
    
    # 输出报告
    if output_report:
        report_path = file_path.parent / f"{file_path.stem}_validation_report.json"
        validator.generate_validation_report(df, str(report_path))
        print(f"验证报告已保存到: {report_path}")
    
    return result

if __name__ == "__main__":
    # 示例用法
    validator = DataValidator()
    converter = DataConverter()
    
    print("数据格式验证和转换工具已加载")
    print("支持的功能:")
    print("- 数据格式验证")
    print("- 数据质量检查")
    print("- 格式转换 (Excel, CSV, MATLAB)")
    print("- 数据修复")
    print("- 验证报告生成")