import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
import seaborn as sns
from .data_format import UnifiedDataFormat

# Set font and style
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class SimplifiedVisualizer:
    """
    Simplified visualizer class for key parameter plots
    """
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Initialize the visualizer
        
        Args:
            figsize: Figure size
        """
        self.figsize = figsize
        self.colors = {
            'actual': '#2E86AB',
            'reference': '#A23B72',
            'error': '#F18F01',
            'control': '#C73E1D'
        }
    
    def plot_key_parameters(self, df: pd.DataFrame, model_name: str = "Model", 
                          save_path: Optional[str] = None) -> None:
        """
        Plot key parameter charts (4 subplots)
        
        Args:
            df: Standard format DataFrame
            model_name: Model name
            save_path: Save path
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(f'{model_name} - Key Parameter Validation Results', fontsize=16, fontweight='bold')
        
        time_col = UnifiedDataFormat.STANDARD_COLUMNS['time']
        
        # 1. State trajectory (u, v, r)
        ax1 = axes[0, 0]
        ax1.plot(df[time_col], df['u'], color=self.colors['actual'], linewidth=2, label='Surge velocity u')
        ax1.plot(df[time_col], df['v'], color=self.colors['reference'], linewidth=2, label='Sway velocity v')
        ax1.plot(df[time_col], df['r'], color=self.colors['error'], linewidth=2, label='Yaw rate r')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Velocity/Angular velocity')
        ax1.set_title('State Variable Trajectories')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Position trajectory (X-Y plane)
        ax2 = axes[0, 1]
        ax2.plot(df['X'], df['Y'], color=self.colors['actual'], linewidth=2, label='Actual trajectory')
        if 'LosXF' in df.columns and 'LosYF' in df.columns:
            ax2.plot(df['LosXF'], df['LosYF'], '--', color=self.colors['reference'], 
                    linewidth=2, label='Reference trajectory')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Position Trajectory')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        # 3. Control inputs
        ax3 = axes[1, 0]
        ax3.plot(df[time_col], df['Tp'], color=self.colors['control'], linewidth=2, label='Port thrust Tp')
        ax3.plot(df[time_col], df['Ts'], color=self.colors['error'], linewidth=2, label='Starboard thrust Ts')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Thrust (N)')
        ax3.set_title('Control Inputs')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Tracking errors
        ax4 = axes[1, 1]
        if 'Lateral_Error' in df.columns and 'Heading_Error' in df.columns:
            ax4.plot(df[time_col], np.abs(df['Lateral_Error']), 
                    color=self.colors['actual'], linewidth=2, label='Lateral error')
            ax4.plot(df[time_col], np.abs(df['Heading_Error']), 
                    color=self.colors['reference'], linewidth=2, label='Heading error')
            ax4.set_yscale('log')
        else:
            # If no error data, show heading angle
            ax4.plot(df[time_col], df['psi'], color=self.colors['actual'], linewidth=2, label='Heading angle ψ')
        
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Error/Angle')
        ax4.set_title('Tracking Errors/Heading Angle')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        # plt.show()
        return fig
    
    def plot_performance_summary(self, df: pd.DataFrame, model_name: str = "Model",
                               save_path: Optional[str] = None) -> Dict[str, float]:
        """
        Plot performance summary charts and return key metrics
        
        Args:
            df: Standard format DataFrame
            model_name: Model name
            save_path: Save path
            
        Returns:
            Dict[str, float]: Key performance metrics
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'{model_name} - Performance Summary', fontsize=16, fontweight='bold')
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(df)
        
        # 1. State variable statistics
        ax1 = axes[0]
        states = ['u', 'v', 'r']
        state_means = [df[state].mean() for state in states]
        state_stds = [df[state].std() for state in states]
        
        x_pos = np.arange(len(states))
        bars = ax1.bar(x_pos, state_means, yerr=state_stds, capsize=5, 
                      color=[self.colors['actual'], self.colors['reference'], self.colors['error']],
                      alpha=0.7)
        ax1.set_xlabel('State Variables')
        ax1.set_ylabel('Mean ± Std')
        ax1.set_title('State Variable Statistics')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(states)
        ax1.grid(True, alpha=0.3)
        
        # 2. Control input distribution
        ax2 = axes[1]
        control_data = [df['Tp'].values, df['Ts'].values]
        bp = ax2.boxplot(control_data, labels=['Tp', 'Ts'], patch_artist=True)
        bp['boxes'][0].set_facecolor(self.colors['control'])
        bp['boxes'][1].set_facecolor(self.colors['error'])
        ax2.set_xlabel('Control Inputs')
        ax2.set_ylabel('Thrust (N)')
        ax2.set_title('Control Input Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 3. Key performance indicators
        ax3 = axes[2]
        if len(metrics) > 0:
            # Normalized indicators for radar chart
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            # Simplified bar chart for key indicators
            bars = ax3.bar(range(len(metric_names)), metric_values, 
                          color=self.colors['actual'], alpha=0.7)
            ax3.set_xlabel('Performance Metrics')
            ax3.set_ylabel('Values')
            ax3.set_title('Key Performance Indicators')
            ax3.set_xticks(range(len(metric_names)))
            ax3.set_xticklabels(metric_names, rotation=45, ha='right')
            ax3.grid(True, alpha=0.3)
            
            # Display values on bars
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance summary plot saved to: {save_path}")
        
        plt.show()
        
        return fig, metrics
    
    def _calculate_performance_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate performance metrics
        
        Args:
            df: Standard format DataFrame
            
        Returns:
            Dict[str, float]: Performance metrics dictionary
        """
        metrics = {}
        
        try:
            # Control effort
            control_effort = np.sqrt(df['Tp']**2 + df['Ts']**2).mean()
            metrics['Control Effort'] = control_effort
            
            # State variation rate
            state_variation = (df['u'].std() + df['v'].std() + df['r'].std()) / 3
            metrics['State Variation'] = state_variation
            
            # If error data exists
            if 'Lateral_Error' in df.columns and 'Heading_Error' in df.columns:
                lateral_rmse = np.sqrt(np.mean(df['Lateral_Error']**2))
                heading_rmse = np.sqrt(np.mean(df['Heading_Error']**2))
                metrics['Lateral RMSE'] = lateral_rmse
                metrics['Heading RMSE'] = heading_rmse
            
            # Trajectory smoothness
            if len(df) > 1:
                dx = np.diff(df['X'])
                dy = np.diff(df['Y'])
                path_smoothness = np.mean(np.sqrt(dx**2 + dy**2))
                metrics['Path Smoothness'] = path_smoothness
        
        except Exception as e:
            print(f"Error calculating performance metrics: {e}")
        
        return metrics
    
    def plot_comparison(self, data_dict: Dict[str, pd.DataFrame], 
                       parameter: str = 'u', save_path: Optional[str] = None) -> None:
        """
        Compare specific parameters across multiple models
        
        Args:
            data_dict: Model data dictionary {model_name: dataframe}
            parameter: Parameter to compare
            save_path: Save path
        """
        plt.figure(figsize=(12, 6))
        
        time_col = UnifiedDataFormat.STANDARD_COLUMNS['time']
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        for i, (model_name, df) in enumerate(data_dict.items()):
            if parameter in df.columns:
                plt.plot(df[time_col], df[parameter], 
                        color=colors[i % len(colors)], linewidth=2, 
                        label=f'{model_name}')
        
        plt.xlabel('Time (s)')
        plt.ylabel(f'{parameter}')
        plt.title(f'Model Comparison - {parameter}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to: {save_path}")
        
        plt.show()
    
    @staticmethod
    def quick_plot(df: pd.DataFrame, model_name: str = "Model") -> None:
        """
        Quick plot of key parameters (single plot version)
        
        Args:
            df: Standard format DataFrame
            model_name: Model name
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        time_col = UnifiedDataFormat.STANDARD_COLUMNS['time']
        
        # Plot main state variables
        ax.plot(df[time_col], df['u'], 'b-', linewidth=2, label='Surge velocity u')
        ax.plot(df[time_col], df['v'], 'r-', linewidth=2, label='Sway velocity v')
        ax.plot(df[time_col], df['r'], 'g-', linewidth=2, label='Yaw rate r')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity/Angular velocity')
        ax.set_title(f'{model_name} - State Variables')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report_plots(self, df: pd.DataFrame, model_name: str,
                            output_dir: str = "./plots") -> List[str]:
        """
        Generate all plots for report
        
        Args:
            df: Standard format DataFrame
            model_name: Model name
            output_dir: Output directory
            
        Returns:
            List[str]: List of generated plot file paths
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        generated_files = []
        
        # Key parameters plot
        key_params_path = os.path.join(output_dir, f"{model_name}_key_parameters.png")
        self.plot_key_parameters(df, model_name, key_params_path)
        generated_files.append(key_params_path)
        
        # Performance summary plot
        performance_path = os.path.join(output_dir, f"{model_name}_performance_summary.png")
        self.plot_performance_summary(df, model_name, performance_path)
        generated_files.append(performance_path)
        
        return generated_files