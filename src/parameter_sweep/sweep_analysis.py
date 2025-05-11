import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
import json
from argparse import ArgumentParser

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from results_manager import combine_sweep_results
from visualization_sweep import set_plot_style, plot_parameter_effect, plot_heatmap

def identify_significant_parameters(df: pd.DataFrame, 
                                  target_metric: str = 'QRC_final_test_acc',
                                  threshold: float = 0.1) -> List[str]:
    """
    Identify parameters that significantly affect the target metric.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe
    target_metric : str, optional
        Target metric to analyze, by default 'QRC_final_test_acc'
    threshold : float, optional
        Correlation threshold, by default 0.1
        
    Returns
    -------
    List[str]
        List of significant parameters
    """
    # Calculate correlations with target metric
    corr = df.corr()[target_metric].abs()
    
    # Exclude metrics and non-parameters
    exclude_patterns = ['_acc', '_loss', 'epochs_to_', 'stability', 'timestamp']
    param_corrs = corr[~corr.index.str.contains('|'.join(exclude_patterns))]
    
    # Filter by threshold
    significant_params = param_corrs[param_corrs > threshold].index.tolist()
    
    # Remove the target metric itself if present
    if target_metric in significant_params:
        significant_params.remove(target_metric)
    
    return significant_params

def generate_best_configuration_table(df: pd.DataFrame, 
                                    metrics: List[str] = None, 
                                    output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Generate a table of best configurations for each metric.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe
    metrics : List[str], optional
        List of metrics to analyze, by default None (all test accuracy metrics)
    output_path : Optional[str], optional
        Path to save table as CSV, by default None
        
    Returns
    -------
    pd.DataFrame
        Table of best configurations
    """
    # If metrics not specified, use all test accuracy metrics
    if metrics is None:
        metrics = [col for col in df.columns if '_test_acc' in col]
    
    # Get best configuration for each metric
    best_configs = []
    for metric in metrics:
        # Skip columns that aren't metrics
        if not any(pattern in metric for pattern in ['_test_acc', '_train_acc']):
            continue
            
        best_idx = df[metric].idxmax()
        best_config = df.loc[best_idx].to_dict()
        
        # Add metric name and value
        best_config['metric'] = metric
        best_config['value'] = best_config[metric]
        
        best_configs.append(best_config)
    
    # Convert to dataframe
    best_df = pd.DataFrame(best_configs)
    
    # Select relevant columns
    important_cols = ['metric', 'value', 'reduction_method', 'dim_reduction', 
                     'readout_type', 'n_shots', 'rabi_freq', 
                     'evolution_time', 'time_steps', 'seed']
    
    # Keep only columns that exist in the dataframe
    cols_to_keep = ['metric', 'value'] + [col for col in important_cols if col in best_df.columns]
    
    # Save to CSV if output path provided
    if output_path:
        best_df[cols_to_keep].to_csv(output_path, index=False)
    
    return best_df[cols_to_keep]

def create_pairwise_interaction_plots(df: pd.DataFrame, 
                                    params: List[str], 
                                    target_metric: str = 'QRC_final_test_acc',
                                    output_dir: Optional[str] = None) -> None:
    """
    Create pairwise interaction plots for parameters.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe
    params : List[str]
        List of parameters to analyze
    target_metric : str, optional
        Target metric to analyze, by default 'QRC_final_test_acc'
    output_dir : Optional[str], optional
        Directory to save plots, by default None
    """
    set_plot_style()
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate all pairs of parameters
    for i, param1 in enumerate(params):
        for param2 in params[i+1:]:
            # Check if parameters exist and have multiple values
            if (param1 in df.columns and param2 in df.columns and
                df[param1].nunique() > 1 and df[param2].nunique() > 1):
                
                # Create heatmap
                plt.figure(figsize=(10, 8))
                
                # Create pivot table
                pivot = df.pivot_table(
                    index=param2, 
                    columns=param1, 
                    values=target_metric,
                    aggfunc='mean'
                )
                
                # Plot heatmap
                sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
                
                plt.title(f'Interaction between {param1} and {param2}')
                plt.tight_layout()
                
                # Save plot if output directory provided
                if output_dir:
                    plt.savefig(os.path.join(output_dir, f'interaction_{param1}_{param2}.png'))
                    plt.close()
                else:
                    plt.show()

def analyze_statistical_significance(df: pd.DataFrame,
                                   param: str,
                                   target_metric: str = 'QRC_final_test_acc',
                                   output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Analyze statistical significance of a parameter's effect.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe
    param : str
        Parameter to analyze
    target_metric : str, optional
        Target metric to analyze, by default 'QRC_final_test_acc'
    output_path : Optional[str], optional
        Path to save results as CSV, by default None
        
    Returns
    -------
    pd.DataFrame
        Statistical summary
    """
    if param not in df.columns or target_metric not in df.columns:
        print(f"Error: Parameter {param} or metric {target_metric} not found in dataframe")
        return None
    
    # Group by parameter and calculate statistics
    stats = df.groupby(param)[target_metric].agg(['mean', 'std', 'count'])
    
    # Calculate 95% confidence interval
    stats['conf_int'] = 1.96 * stats['std'] / np.sqrt(stats['count'])
    stats['lower_ci'] = stats['mean'] - stats['conf_int']
    stats['upper_ci'] = stats['mean'] + stats['conf_int']
    
    # Save to CSV if output path provided
    if output_path:
        stats.to_csv(output_path)
    
    return stats

def create_scientific_report(df: pd.DataFrame, 
                           output_dir: str,
                           target_metric: str = 'QRC_final_test_acc',
                           experiment_name: str = "Parameter Sweep") -> None:
    """
    Create a comprehensive scientific report of parameter sweep results.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe
    output_dir : str
        Directory to save report files
    target_metric : str, optional
        Target metric to analyze, by default 'QRC_final_test_acc'
    experiment_name : str, optional
        Name of the experiment for the report title, by default "Parameter Sweep"
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Identify significant parameters
    significant_params = identify_significant_parameters(df, target_metric)
    
    # Create report dictionary
    report = {
        "experiment_name": experiment_name,
        "timestamp": pd.Timestamp.now().isoformat(),
        "parameters_analyzed": list(df.columns),
        "number_of_experiments": len(df),
        "significant_parameters": significant_params,
        "best_configurations": {}
    }
    
    # Get best configurations
    best_configs_df = generate_best_configuration_table(
        df, 
        output_path=os.path.join(output_dir, "best_configurations.csv")
    )
    
    # Generate parameter effect plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    for param in significant_params:
        if df[param].nunique() > 1:
            # Create parameter effect plot
            plot_parameter_effect(
                df, 
                param, 
                target_metric, 
                output_path=os.path.join(plots_dir, f'effect_{param}.png')
            )
            
            # Analyze statistical significance
            stats = analyze_statistical_significance(
                df, 
                param, 
                target_metric, 
                output_path=os.path.join(output_dir, f'stats_{param}.csv')
            )
            
            # Add to report
            report["parameter_statistics"] = report.get("parameter_statistics", {})
            report["parameter_statistics"][param] = stats.to_dict()
    
    # Create pairwise interaction plots
    if len(significant_params) >= 2:
        create_pairwise_interaction_plots(
            df, 
            significant_params, 
            target_metric, 
            output_dir=os.path.join(plots_dir, "interactions")
        )
    
    # Save report as JSON
    with open(os.path.join(output_dir, "report.json"), 'w') as f:
        # Convert complex objects to strings for JSON serialization
        report_serializable = json.loads(pd.json.dumps(report))
        json.dump(report_serializable, f, indent=4)
    
    # Generate HTML report
    html_report = generate_html_report(df, report, significant_params, target_metric)
    
    with open(os.path.join(output_dir, "report.html"), 'w') as f:
        f.write(html_report)
    
    print(f"Scientific report created in {output_dir}")

def generate_html_report(df: pd.DataFrame, 
                       report: Dict, 
                       significant_params: List[str],
                       target_metric: str) -> str:
    """
    Generate an HTML report from the analysis results.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe
    report : Dict
        Report dictionary
    significant_params : List[str]
        List of significant parameters
    target_metric : str
        Target metric analyzed
        
    Returns
    -------
    str
        HTML report content
    """
    # Generate basic HTML report
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{report['experiment_name']} - Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ text-align: left; padding: 8px; border: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .significant {{ font-weight: bold; color: #cc0000; }}
            .container {{ margin-bottom: 30px; }}
            img {{ max-width: 90%; margin: 15px 0; }}
        </style>
    </head>
    <body>
        <h1>{report['experiment_name']} - Analysis Report</h1>
        <p>Generated on: {report['timestamp']}</p>
        
        <div class="container">
            <h2>Experiment Summary</h2>
            <p>Number of experiments: {report['number_of_experiments']}</p>
            <p>Target metric: {target_metric}</p>
        </div>
        
        <div class="container">
            <h2>Significant Parameters</h2>
            <ul>
    """
    
    # Add significant parameters
    for param in significant_params:
        html += f"<li>{param}</li>\n"
    
    html += """
            </ul>
        </div>
        
        <div class="container">
            <h2>Parameter Effects</h2>
    """
    
    # Add parameter effect plots
    for param in significant_params:
        if df[param].nunique() > 1:
            html += f"""
            <h3>Effect of {param}</h3>
            <img src="plots/effect_{param}.png" alt="Effect of {param}">
            """
    
    html += """
        </div>
        
        <div class="container">
            <h2>Best Configurations</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
    """
    
    # Add columns for significant parameters
    for param in significant_params:
        if param in df.columns:
            html += f"<th>{param}</th>\n"
    
    html += """
                </tr>
    """
    
    # Add best configurations
    best_configs_df = generate_best_configuration_table(df)
    for _, row in best_configs_df.iterrows():
        html += f"""
                <tr>
                    <td>{row['metric']}</td>
                    <td>{row['value']:.4f}</td>
        """
        
        # Add values for significant parameters
        for param in significant_params:
            if param in row:
                html += f"<td>{row[param]}</td>\n"
            else:
                html += "<td>N/A</td>\n"
                
        html += "</tr>\n"
    
    html += """
            </table>
        </div>
        
        <div class="container">
            <h2>Conclusion</h2>
            <p>
                Based on the parameter sweep analysis, the following recommendations can be made:
            </p>
            <ul>
    """
    
    # Add recommendations
    for param in significant_params:
        if param in df.columns and df[param].nunique() > 1:
            # Get best value for this parameter
            param_avg = df.groupby(param)[target_metric].mean()
            best_value = param_avg.idxmax()
            html += f"<li>Use {param} = {best_value} for optimal performance</li>\n"
    
    html += """
            </ul>
        </div>
    </body>
    </html>
    """
    
    return html

if __name__ == "__main__":
    parser = ArgumentParser(description="Analyze parameter sweep results")
    parser.add_argument("results_file", type=str, help="Path to CSV file with sweep results")
    parser.add_argument("output_dir", type=str, help="Directory to save analysis outputs")
    parser.add_argument("--target-metric", type=str, default="QRC_final_test_acc",
                       help="Target metric to analyze")
    parser.add_argument("--experiment-name", type=str, default="QRC Parameter Analysis",
                       help="Name of the experiment for the report title")
    
    args = parser.parse_args()
    
    # Load results
    df = pd.read_csv(args.results_file)
    
    # Create scientific report
    create_scientific_report(
        df, 
        args.output_dir,
        args.target_metric,
        args.experiment_name
    )
