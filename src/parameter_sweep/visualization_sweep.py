import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional, Union
import json
from matplotlib.gridspec import GridSpec

# For 3D plotting
from mpl_toolkits.mplot3d import Axes3D

def set_plot_style():
    """Set publication-quality plot style"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'figure.figsize': (10, 6),
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })

def plot_parameter_effect(df: pd.DataFrame, 
                         parameter: str, 
                         metric: str = 'QRC_final_test_acc',
                         title: Optional[str] = None,
                         output_path: Optional[str] = None) -> None:
    """
    Plot the effect of a parameter on a performance metric.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe
    parameter : str
        Parameter to plot on x-axis
    metric : str, optional
        Metric to plot on y-axis, by default 'QRC_final_test_acc'
    title : Optional[str], optional
        Plot title, by default None
    output_path : Optional[str], optional
        Path to save the figure, by default None
    """
    set_plot_style()
    
    if parameter not in df.columns or metric not in df.columns:
        print(f"Parameter {parameter} or metric {metric} not found in dataframe")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Convert parameter to string for consistent sorting
    df[f'{parameter}_str'] = df[parameter].astype(str)
    
    # Create box plot
    sns.boxplot(x=f'{parameter}_str', y=metric, data=df)
    
    if title:
        plt.title(title)
    else:
        plt.title(f'Effect of {parameter} on {metric}')
        
    plt.xlabel(parameter)
    plt.ylabel(metric.replace('_', ' '))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        
    plt.show()

def plot_parameter_comparison(df: pd.DataFrame, 
                             x_param: str, 
                             hue_param: str,
                             metric: str = 'QRC_final_test_acc',
                             title: Optional[str] = None,
                             output_path: Optional[str] = None) -> None:
    """
    Compare the effect of two parameters on a performance metric.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe
    x_param : str
        Parameter to plot on x-axis
    hue_param : str
        Parameter to use for grouping (color)
    metric : str, optional
        Metric to plot on y-axis, by default 'QRC_final_test_acc'
    title : Optional[str], optional
        Plot title, by default None
    output_path : Optional[str], optional
        Path to save the figure, by default None
    """
    set_plot_style()
    
    for param in [x_param, hue_param]:
        if param not in df.columns:
            print(f"Parameter {param} not found in dataframe")
            return
            
    if metric not in df.columns:
        print(f"Metric {metric} not found in dataframe")
        return
    
    plt.figure(figsize=(12, 7))
    
    # Convert parameters to string for consistent sorting
    df[f'{x_param}_str'] = df[x_param].astype(str)
    df[f'{hue_param}_str'] = df[hue_param].astype(str)
    
    # Create grouped box plot
    sns.boxplot(x=f'{x_param}_str', y=metric, hue=f'{hue_param}_str', data=df)
    
    if title:
        plt.title(title)
    else:
        plt.title(f'Effect of {x_param} and {hue_param} on {metric}')
        
    plt.xlabel(x_param)
    plt.ylabel(metric.replace('_', ' '))
    plt.xticks(rotation=45)
    plt.legend(title=hue_param)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        
    plt.show()

def plot_heatmap(df: pd.DataFrame, 
                x_param: str, 
                y_param: str,
                metric: str = 'QRC_final_test_acc',
                title: Optional[str] = None,
                output_path: Optional[str] = None) -> None:
    """
    Create a heatmap showing the interaction between two parameters.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe
    x_param : str
        Parameter for x-axis
    y_param : str
        Parameter for y-axis
    metric : str, optional
        Metric to plot, by default 'QRC_final_test_acc'
    title : Optional[str], optional
        Plot title, by default None
    output_path : Optional[str], optional
        Path to save the figure, by default None
    """
    set_plot_style()
    
    for param in [x_param, y_param]:
        if param not in df.columns:
            print(f"Parameter {param} not found in dataframe")
            return
            
    if metric not in df.columns:
        print(f"Metric {metric} not found in dataframe")
        return
    
    # Create pivot table
    pivot = df.pivot_table(index=y_param, columns=x_param, values=metric, aggfunc='mean')
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", cbar_kws={'label': metric.replace('_', ' ')})
    
    if title:
        plt.title(title)
    else:
        plt.title(f'Interaction between {x_param} and {y_param} on {metric}')
        
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        
    plt.show()

def plot_3d_surface(df: pd.DataFrame, 
                   x_param: str, 
                   y_param: str,
                   metric: str = 'QRC_final_test_acc',
                   title: Optional[str] = None,
                   output_path: Optional[str] = None) -> None:
    """
    Create a 3D surface plot showing the interaction between two parameters.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe
    x_param : str
        Parameter for x-axis
    y_param : str
        Parameter for y-axis
    metric : str, optional
        Metric to plot, by default 'QRC_final_test_acc'
    title : Optional[str], optional
        Plot title, by default None
    output_path : Optional[str], optional
        Path to save the figure, by default None
    """
    set_plot_style()
    
    for param in [x_param, y_param]:
        if param not in df.columns:
            print(f"Parameter {param} not found in dataframe")
            return
            
    if metric not in df.columns:
        print(f"Metric {metric} not found in dataframe")
        return
    
    # Create pivot table
    pivot = df.pivot_table(index=y_param, columns=x_param, values=metric, aggfunc='mean')
    
    # Create meshgrid
    x = pivot.columns
    y = pivot.index
    X, Y = np.meshgrid(x, y)
    Z = pivot.values
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create surface plot
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_zlabel(metric.replace('_', ' '))
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'3D Surface: {x_param} vs {y_param} on {metric}')
        
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        
    plt.show()

def plot_metric_comparison(df: pd.DataFrame, 
                         group_by: str, 
                         metrics: List[str] = None,
                         title: Optional[str] = None,
                         output_path: Optional[str] = None) -> None:
    """
    Compare different metrics grouped by a parameter.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe
    group_by : str
        Parameter to group by
    metrics : List[str], optional
        List of metrics to compare, by default None (will use all test_acc metrics)
    title : Optional[str], optional
        Plot title, by default None
    output_path : Optional[str], optional
        Path to save the figure, by default None
    """
    set_plot_style()
    
    if group_by not in df.columns:
        print(f"Parameter {group_by} not found in dataframe")
        return
    
    # If metrics not specified, use all test accuracy metrics
    if metrics is None:
        metrics = [col for col in df.columns if '_test_acc' in col]
    
    # Check metrics exist
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        print("No specified metrics found in dataframe")
        return
    
    # Convert to long format
    df_long = df.melt(id_vars=[group_by], 
                     value_vars=metrics, 
                     var_name='Metric', 
                     value_name='Accuracy')
    
    # Clean up metric names for legend
    df_long['Metric'] = df_long['Metric'].str.replace('_final_test_acc', '')
    
    plt.figure(figsize=(12, 7))
    
    # Create grouped bar plot
    sns.barplot(x=group_by, y='Accuracy', hue='Metric', data=df_long)
    
    if title:
        plt.title(title)
    else:
        plt.title(f'Comparison of Models Grouped by {group_by}')
        
    plt.xlabel(group_by)
    plt.ylabel('Test Accuracy')
    plt.xticks(rotation=45)
    plt.legend(title='Model')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        
    plt.show()

def create_summary_dashboard(df: pd.DataFrame, 
                           output_dir: str,
                           metric: str = 'QRC_final_test_acc') -> None:
    """
    Create a comprehensive dashboard of plots for a parameter sweep.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe
    output_dir : str
        Directory to save the plots
    metric : str, optional
        Main metric to analyze, by default 'QRC_final_test_acc'
    """
    set_plot_style()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Find the most variable parameters
    numeric_params = df.select_dtypes(include=['number']).columns
    params_to_analyze = []
    
    for param in numeric_params:
        # Skip metrics and parameters with only one value
        if ('_acc' in param or '_loss' in param or 
            df[param].nunique() <= 1 or param == 'seed'):
            continue
        params_to_analyze.append(param)
    
    # 2. Plot individual parameter effects
    for param in params_to_analyze:
        output_path = os.path.join(output_dir, f'param_effect_{param}.png')
        plot_parameter_effect(df, param, metric, output_path=output_path)
    
    # 3. Plot parameter interactions (pairs)
    if len(params_to_analyze) >= 2:
        for i, param1 in enumerate(params_to_analyze[:-1]):
            for param2 in params_to_analyze[i+1:]:
                output_path = os.path.join(output_dir, f'heatmap_{param1}_vs_{param2}.png')
                plot_heatmap(df, param1, param2, metric, output_path=output_path)
    
    # 4. Compare metrics for different models
    test_acc_metrics = [col for col in df.columns if '_test_acc' in col]
    if len(test_acc_metrics) > 1 and 'reduction_method' in df.columns:
        output_path = os.path.join(output_dir, 'model_comparison.png')
        plot_metric_comparison(df, 'reduction_method', test_acc_metrics, output_path=output_path)
    
    # 5. Create correlation matrix of parameters and metrics
    output_path = os.path.join(output_dir, 'correlation_matrix.png')
    plt.figure(figsize=(12, 10))
    
    # Select columns for correlation
    corr_cols = params_to_analyze + test_acc_metrics
    corr = df[corr_cols].corr()
    
    # Plot correlation matrix
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', 
               vmin=-1, vmax=1, center=0, square=True)
    
    plt.title('Parameter and Performance Correlation Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    # 6. Create a summary report
    summary = {
        'parameters_analyzed': params_to_analyze,
        'metrics_analyzed': test_acc_metrics,
        'best_configurations': {}
    }
    
    # Find best configuration for each metric
    for metric in test_acc_metrics:
        best_idx = df[metric].idxmax()
        best_config = df.loc[best_idx].to_dict()
        
        # Filter out non-parameter values
        best_params = {k: v for k, v in best_config.items() 
                     if k in params_to_analyze or k == 'reduction_method'}
        best_params['value'] = best_config[metric]
        
        summary['best_configurations'][metric] = best_params
    
    # Save summary as JSON
    with open(os.path.join(output_dir, 'summary_report.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"Dashboard created in {output_dir}")

def create_publication_figure(df: pd.DataFrame, 
                             output_path: str,
                             main_params: List[str],
                             main_metric: str = 'QRC_final_test_acc') -> None:
    """
    Create a publication-quality figure summarizing key results.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe
    output_path : str
        Path to save the figure
    main_params : List[str]
        List of main parameters to highlight
    main_metric : str, optional
        Main metric to analyze, by default 'QRC_final_test_acc'
    """
    set_plot_style()
    
    # Verify parameters and metrics exist
    for param in main_params:
        if param not in df.columns:
            print(f"Parameter {param} not found in dataframe")
            return
    
    if main_metric not in df.columns:
        print(f"Metric {main_metric} not found in dataframe")
        return
    
    # Find all test accuracy metrics
    test_acc_metrics = [col for col in df.columns if '_test_acc' in col]
    
    # Create a multi-panel figure
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig)
    
    # Panel 1: Parameter Effect (Top-left)
    if len(main_params) > 0:
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Convert parameter to string for consistent sorting
        df[f'{main_params[0]}_str'] = df[main_params[0]].astype(str)
        
        # Create box plot
        sns.boxplot(x=f'{main_params[0]}_str', y=main_metric, data=df, ax=ax1)
        
        ax1.set_title(f'Effect of {main_params[0]}')
        ax1.set_xlabel(main_params[0])
        ax1.set_ylabel(main_metric.replace('_', ' '))
        ax1.tick_params(axis='x', rotation=45)
    
    # Panel 2: Parameter Interaction (Top-middle)
    if len(main_params) >= 2:
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Create pivot table
        pivot = df.pivot_table(
            index=main_params[1], 
            columns=main_params[0], 
            values=main_metric, 
            aggfunc='mean'
        )
        
        # Create heatmap
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", 
                   cbar_kws={'label': main_metric.replace('_', ' ')},
                   ax=ax2)
        
        ax2.set_title(f'Interaction: {main_params[0]} vs {main_params[1]}')
    
    # Panel 3: Model Comparison (Top-right)
    if 'reduction_method' in df.columns and len(test_acc_metrics) > 1:
        ax3 = fig.add_subplot(gs[0, 2])
        
        # Convert to long format for model comparison
        df_long = df.melt(id_vars=['reduction_method'], 
                         value_vars=test_acc_metrics, 
                         var_name='Metric', 
                         value_name='Accuracy')
        
        # Clean up metric names for legend
        df_long['Metric'] = df_long['Metric'].str.replace('_final_test_acc', '')
        
        # Create grouped bar plot
        sns.barplot(x='reduction_method', y='Accuracy', hue='Metric', data=df_long, ax=ax3)
        
        ax3.set_title('Model Performance by Reduction Method')
        ax3.set_xlabel('Reduction Method')
        ax3.set_ylabel('Test Accuracy')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend(title='Model')
    
    # Panel 4: 3D Surface Plot (Bottom-left)
    if len(main_params) >= 2:
        ax4 = fig.add_subplot(gs[1, 0], projection='3d')
        
        # Create pivot table
        pivot = df.pivot_table(
            index=main_params[1], 
            columns=main_params[0], 
            values=main_metric, 
            aggfunc='mean'
        )
        
        # Create meshgrid
        x = pivot.columns
        y = pivot.index
        X, Y = np.meshgrid(x, y)
        Z = pivot.values
        
        # Create surface plot
        surf = ax4.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
        
        ax4.set_xlabel(main_params[0])
        ax4.set_ylabel(main_params[1])
        ax4.set_zlabel(main_metric.replace('_', ' '))
        
        ax4.set_title(f'3D Surface: Parameter Interaction')
    
    # Panel 5: Line Plot for Another Parameter (Bottom-middle)
    if len(main_params) >= 3:
        ax5 = fig.add_subplot(gs[1, 1])
        
        # Group by the third parameter and plot lines for different reduction methods
        if 'reduction_method' in df.columns:
            grouped = df.groupby([main_params[2], 'reduction_method'])[main_metric].mean().reset_index()
            
            sns.lineplot(x=main_params[2], y=main_metric, hue='reduction_method', 
                        data=grouped, markers=True, dashes=False, ax=ax5)
            
            ax5.set_title(f'Effect of {main_params[2]} by Method')
            ax5.set_xlabel(main_params[2])
            ax5.set_ylabel(main_metric.replace('_', ' '))
        else:
            grouped = df.groupby(main_params[2])[main_metric].mean().reset_index()
            
            sns.lineplot(x=main_params[2], y=main_metric, 
                        data=grouped, markers=True, ax=ax5)
            
            ax5.set_title(f'Effect of {main_params[2]}')
            ax5.set_xlabel(main_params[2])
            ax5.set_ylabel(main_metric.replace('_', ' '))
    
    # Panel 6: Best Performance Table (Bottom-right)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Create best performance table
    best_configs = {}
    for metric in test_acc_metrics:
        model_name = metric.replace('_final_test_acc', '')
        best_idx = df[metric].idxmax()
        best_val = df.loc[best_idx, metric]
        
        # Get key parameters for this configuration
        if 'reduction_method' in df.columns:
            method = df.loc[best_idx, 'reduction_method']
        else:
            method = 'N/A'
            
        param_values = []
        for param in main_params:
            if param in df.columns:
                param_values.append(f"{param}: {df.loc[best_idx, param]}")
        
        best_configs[model_name] = {
            'accuracy': best_val,
            'method': method,
            'parameters': ', '.join(param_values)
        }
    
    # Convert to table
    table_data = []
    for model, config in best_configs.items():
        table_data.append([
            model, 
            f"{config['accuracy']:.4f}", 
            config['method'], 
            config['parameters']
        ])
    
    # Create table
    table = ax6.table(
        cellText=table_data,
        colLabels=['Model', 'Best Accuracy', 'Method', 'Parameters'],
        loc='center',
        cellLoc='center'
    )
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    ax6.set_title('Best Configurations')
    
    # Add super title
    plt.suptitle('Parameter Sweep Analysis for Quantum Reservoir Computing', fontsize=20)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Publication figure saved to {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize parameter sweep results")
    parser.add_argument("--results-csv", type=str, required=True,
                       help="Path to CSV file with sweep results")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Directory to save visualization outputs")
    parser.add_argument("--main-params", type=str, nargs='+',
                       help="Main parameters to highlight in publication figure")
    parser.add_argument("--main-metric", type=str, default="QRC_final_test_acc",
                       help="Main metric to analyze")
    
    args = parser.parse_args()
    
    # Load results
    df = pd.read_csv(args.results_csv)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dashboard
    create_summary_dashboard(df, args.output_dir, args.main_metric)
    
    # Create publication figure if main parameters specified
    if args.main_params:
        pub_fig_path = os.path.join(args.output_dir, "publication_figure.png")
        create_publication_figure(df, pub_fig_path, args.main_params, args.main_metric)
