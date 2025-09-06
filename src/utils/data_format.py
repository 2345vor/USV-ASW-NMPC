import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import json
from datetime import datetime

class UnifiedDataFormat:
    """
    统一数据格式类，定义船舶动力学建模与NMPC控制系统的标准数据格式
    """
    
    # 标准列名定义
    STANDARD_COLUMNS = {
        'time': 'Time',
        'position': ['X', 'Y'],
        'los_guidance': ['LosXF', 'LosYF'],
        'states': ['u', 'v', 'r'],
        'heading': 'psi',
        'controls': ['Tp', 'Ts'],
        'errors': ['Lateral_Error', 'Heading_Error']
    }
    
    # 数据类型定义
    DATA_TYPES = {
        'Time': float,
        'X': float,
        'Y': float,
        'LosXF': float,
        'LosYF': float,
        'u': float,
        'v': float,
        'r': float,
        'psi': float,
        'Tp': float,
        'Ts': float,
        'Lateral_Error': float,
        'Heading_Error': float
    }
    
    @staticmethod
    def create_standard_dataframe(time_steps: int, dt: float = 0.1) -> pd.DataFrame:
        """
        创建标准格式的空DataFrame
        
        参数:
            time_steps: 时间步数
            dt: 时间步长
            
        返回:
            pd.DataFrame: 标准格式的空DataFrame
        """
        # 生成时间序列
        time_array = np.arange(0, time_steps * dt, dt)
        
        # 创建所有列
        all_columns = [UnifiedDataFormat.STANDARD_COLUMNS['time']]
        all_columns.extend(UnifiedDataFormat.STANDARD_COLUMNS['position'])
        all_columns.extend(UnifiedDataFormat.STANDARD_COLUMNS['los_guidance'])
        all_columns.extend(UnifiedDataFormat.STANDARD_COLUMNS['states'])
        all_columns.append(UnifiedDataFormat.STANDARD_COLUMNS['heading'])
        all_columns.extend(UnifiedDataFormat.STANDARD_COLUMNS['controls'])
        all_columns.extend(UnifiedDataFormat.STANDARD_COLUMNS['errors'])
        
        # 创建DataFrame
        df = pd.DataFrame(index=range(len(time_array)), columns=all_columns)
        df[UnifiedDataFormat.STANDARD_COLUMNS['time']] = time_array
        
        # 初始化为0
        for col in all_columns[1:]:
            df[col] = 0.0
            
        return df
    
    @staticmethod
    def validate_data_format(df: pd.DataFrame) -> Dict[str, bool]:
        """
        验证数据格式是否符合标准
        
        参数:
            df: 待验证的DataFrame
            
        返回:
            Dict[str, bool]: 验证结果
        """
        validation_results = {
            'has_all_columns': True,
            'correct_data_types': True,
            'no_missing_values': True,
            'time_monotonic': True
        }
        
        # 检查列名
        required_columns = []
        required_columns.append(UnifiedDataFormat.STANDARD_COLUMNS['time'])
        required_columns.extend(UnifiedDataFormat.STANDARD_COLUMNS['position'])
        required_columns.extend(UnifiedDataFormat.STANDARD_COLUMNS['los_guidance'])
        required_columns.extend(UnifiedDataFormat.STANDARD_COLUMNS['states'])
        required_columns.append(UnifiedDataFormat.STANDARD_COLUMNS['heading'])
        required_columns.extend(UnifiedDataFormat.STANDARD_COLUMNS['controls'])
        required_columns.extend(UnifiedDataFormat.STANDARD_COLUMNS['errors'])
        
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            validation_results['has_all_columns'] = False
            print(f"缺少列: {missing_columns}")
        
        # 检查数据类型
        for col in df.columns:
            if col in UnifiedDataFormat.DATA_TYPES:
                try:
                    df[col].astype(UnifiedDataFormat.DATA_TYPES[col])
                except (ValueError, TypeError):
                    validation_results['correct_data_types'] = False
                    print(f"列 {col} 数据类型不正确")
        
        # 检查缺失值
        if df.isnull().any().any():
            validation_results['no_missing_values'] = False
            print("存在缺失值")
        
        # 检查时间序列单调性
        time_col = UnifiedDataFormat.STANDARD_COLUMNS['time']
        if time_col in df.columns:
            if not df[time_col].is_monotonic_increasing:
                validation_results['time_monotonic'] = False
                print("时间序列不是单调递增的")
        
        return validation_results
    
    @staticmethod
    def convert_to_standard_format(data: Union[Dict, pd.DataFrame, np.ndarray], 
                                 column_mapping: Optional[Dict] = None) -> pd.DataFrame:
        """
        将数据转换为标准格式
        
        参数:
            data: 输入数据
            column_mapping: 列名映射字典
            
        返回:
            pd.DataFrame: 标准格式的DataFrame
        """
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        elif isinstance(data, np.ndarray):
            # 假设数组按标准列顺序排列
            columns = []
            columns.append(UnifiedDataFormat.STANDARD_COLUMNS['time'])
            columns.extend(UnifiedDataFormat.STANDARD_COLUMNS['position'])
            columns.extend(UnifiedDataFormat.STANDARD_COLUMNS['los_guidance'])
            columns.extend(UnifiedDataFormat.STANDARD_COLUMNS['states'])
            columns.append(UnifiedDataFormat.STANDARD_COLUMNS['heading'])
            columns.extend(UnifiedDataFormat.STANDARD_COLUMNS['controls'])
            columns.extend(UnifiedDataFormat.STANDARD_COLUMNS['errors'])
            
            df = pd.DataFrame(data, columns=columns[:data.shape[1]])
        else:
            df = data.copy()
        
        # 应用列名映射
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        return df
    
    @staticmethod
    def save_to_csv(df: pd.DataFrame, filepath: str, validate: bool = True) -> bool:
        """
        保存数据到CSV文件
        
        参数:
            df: 要保存的DataFrame
            filepath: 文件路径
            validate: 是否验证数据格式
            
        返回:
            bool: 保存是否成功
        """
        try:
            if validate:
                validation_results = UnifiedDataFormat.validate_data_format(df)
                if not all(validation_results.values()):
                    print("警告: 数据格式验证失败，但仍将保存")
            
            df.to_csv(filepath, index=False)
            print(f"数据已保存到: {filepath}")
            return True
        except Exception as e:
            print(f"保存失败: {e}")
            return False
    
    @staticmethod
    def load_from_csv(filepath: str, validate: bool = True) -> Optional[pd.DataFrame]:
        """
        从CSV文件加载数据
        
        参数:
            filepath: 文件路径
            validate: 是否验证数据格式
            
        返回:
            Optional[pd.DataFrame]: 加载的DataFrame，失败时返回None
        """
        try:
            df = pd.read_csv(filepath)
            
            if validate:
                validation_results = UnifiedDataFormat.validate_data_format(df)
                if not all(validation_results.values()):
                    print("警告: 数据格式验证失败")
            
            print(f"数据已从 {filepath} 加载")
            return df
        except Exception as e:
            print(f"加载失败: {e}")
            return None
    
    @staticmethod
    def get_metadata(df: pd.DataFrame) -> Dict:
        """
        获取数据集的元数据信息
        
        参数:
            df: DataFrame
            
        返回:
            Dict: 元数据信息
        """
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'shape': df.shape,
            'columns': list(df.columns),
            'time_range': {
                'start': float(df[UnifiedDataFormat.STANDARD_COLUMNS['time']].min()) if UnifiedDataFormat.STANDARD_COLUMNS['time'] in df.columns else None,
                'end': float(df[UnifiedDataFormat.STANDARD_COLUMNS['time']].max()) if UnifiedDataFormat.STANDARD_COLUMNS['time'] in df.columns else None,
                'duration': float(df[UnifiedDataFormat.STANDARD_COLUMNS['time']].max() - df[UnifiedDataFormat.STANDARD_COLUMNS['time']].min()) if UnifiedDataFormat.STANDARD_COLUMNS['time'] in df.columns else None
            },
            'statistics': {
                col: {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                } for col in df.select_dtypes(include=[np.number]).columns
            }
        }
        
        return metadata
    
    @staticmethod
    def save_metadata(metadata: Dict, filepath: str) -> bool:
        """
        保存元数据到JSON文件
        
        参数:
            metadata: 元数据字典
            filepath: 文件路径
            
        返回:
            bool: 保存是否成功
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"元数据已保存到: {filepath}")
            return True
        except Exception as e:
            print(f"元数据保存失败: {e}")
            return False

class DataExporter:
    """
    数据导出器，用于将模型结果导出为标准格式
    """
    
    @staticmethod
    def export_simulation_results(time_array: np.ndarray,
                                states: np.ndarray,
                                controls: np.ndarray,
                                reference_trajectory: Optional[np.ndarray] = None,
                                errors: Optional[np.ndarray] = None,
                                filepath: Optional[str] = None) -> pd.DataFrame:
        """
        导出仿真结果为标准格式
        
        参数:
            time_array: 时间数组
            states: 状态数组 [u, v, r, x, y, psi]
            controls: 控制输入数组 [Tp, Ts]
            reference_trajectory: 参考轨迹 [x_ref, y_ref]
            errors: 误差数组 [lateral_error, heading_error]
            filepath: 保存路径
            
        返回:
            pd.DataFrame: 标准格式的DataFrame
        """
        # 创建标准DataFrame
        df = UnifiedDataFormat.create_standard_dataframe(len(time_array))
        
        # 填充时间
        df[UnifiedDataFormat.STANDARD_COLUMNS['time']] = time_array
        
        # 填充状态数据
        if states.shape[1] >= 3:
            df['u'] = states[:, 0]
            df['v'] = states[:, 1]
            df['r'] = states[:, 2]
        
        if states.shape[1] >= 6:
            df['X'] = states[:, 3]
            df['Y'] = states[:, 4]
            df['psi'] = states[:, 5]
        
        # 填充控制输入
        if controls.shape[1] >= 2:
            df['Tp'] = controls[:, 0]
            df['Ts'] = controls[:, 1]
        
        # 填充参考轨迹
        if reference_trajectory is not None:
            if reference_trajectory.shape[1] >= 2:
                df['LosXF'] = reference_trajectory[:, 0]
                df['LosYF'] = reference_trajectory[:, 1]
        
        # 填充误差
        if errors is not None:
            if errors.shape[1] >= 2:
                df['Lateral_Error'] = errors[:, 0]
                df['Heading_Error'] = errors[:, 1]
        
        # 保存文件
        if filepath:
            UnifiedDataFormat.save_to_csv(df, filepath)
            
            # 保存元数据
            metadata = UnifiedDataFormat.get_metadata(df)
            metadata_filepath = filepath.replace('.csv', '_metadata.json')
            UnifiedDataFormat.save_metadata(metadata, metadata_filepath)
        
        return df