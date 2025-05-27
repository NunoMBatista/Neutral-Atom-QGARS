import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from typing import Optional, Tuple

def load_data(results_file: str, param_grid_file: Optional[str] = None) -> Tuple[pd.DataFrame, dict]:
    """
    Load experiment results and parameter grid information.
    
    Parameters
    ----------
    results_file : str
        Path to the CSV file with experiment results
    param_grid_file : Optional[str]
        Path to the parameter grid JSON file. If None, inferred from results_file.
        
    Returns
    -------
    Tuple[pd.DataFrame, dict]
        DataFrame with results and parameter grid dictionary
    """
    # Load results CSV
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} experiment results from {results_file}")
    
    # Filter only successful runs
    df = df[df['status'] == 'success']
    print(f"After filtering, {len(df)} successful experiments remain")
    
    # Load parameter grid if provided, otherwise infer from results directory
    if param_grid_file is None:
        results_dir = os.path.dirname(results_file)
        param_grid_file = os.path.join(results_dir, "param_grid.json")
    
    if os.path.exists(param_grid_file):
        with open(param_grid_file, 'r') as f:
            param_grid = json.load(f)
        print(f"Loaded parameter grid from {param_grid_file}")
    else:
        # If param_grid.json doesn't exist, create a simple grid from unique values in results
        param_grid = {
            'time_steps': sorted(df['time_steps'].unique().tolist()),
            'evolution_time': sorted(df['evolution_time'].unique().tolist())
        }
        print("Parameter grid file not found, inferred from results")
    
    return df, param_grid

def create_heatmap(df: pd.DataFrame, param_grid: dict, output_path: Optional[str] = None):
    """
    Create a heatmap of QRC test accuracy for different time_steps and evolution_time values.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with experiment results
    param_grid : dict
        Parameter grid dictionary
    output_path : Optional[str]
        Path to save the output figure
    """
    # Pivot data to create a grid of evolution_time × time_steps with test accuracy values
    pivot_df = df.pivot_table(
        index='evolution_time',
        columns='time_steps',
        values='QRC_final_test_acc',
        aggfunc='mean'  # In case of duplicates
    )
    
    # Convert to percentage for better visualization
    pivot_df = pivot_df * 100
    
    # Set up plot with appropriate size
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with seaborn
    ax = sns.heatmap(
        pivot_df,
        annot=True,  # Show values in cells
        fmt='.1f',   # Format as 1 decimal place
        cmap='viridis',  # Color map (blues are better for accuracy visualization)
        vmin=80,     # Minimum value for color scale (adjust based on your data)
        vmax=90,     # Maximum value for color scale (adjust based on your data)
        linewidths=0.5,
        cbar_kws={'label': 'QRC Test Accuracy (%)'}
    )
    
    # Set axis labels and title
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Evolution Time', fontsize=12)
    #plt.title('Effect of Time Steps and Evolution Time on QRC Test Accuracy', fontsize=14)
    
    # Ensure all labels are visible
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    
    plt.show()

def analyze_optimal_parameters(df: pd.DataFrame):
    """
    Analyze and print information about optimal parameter settings.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with experiment results
    """
    print("\nAnalysis of Optimal Parameters:")
    print("-" * 60)
    
    # Find settings with highest test accuracy
    best_idx = df['QRC_final_test_acc'].idxmax()
    best_row = df.loc[best_idx]
    
    print(f"Best QRC test accuracy: {best_row['QRC_final_test_acc']*100:.2f}%")
    print(f"Optimal time steps: {best_row['time_steps']}")
    print(f"Optimal evolution time: {best_row['evolution_time']}")
    
    # Analyze average effect of each parameter
    print("\nAverage Effect of Parameters:")
    print("-" * 60)
    
    # Group by time_steps
    time_steps_effect = df.groupby('time_steps')['QRC_final_test_acc'].mean().sort_index() * 100
    print("Effect of Time Steps:")
    for steps, acc in time_steps_effect.items():
        print(f"  {steps} steps: {acc:.2f}%")
    
    # Group by evolution_time
    evol_time_effect = df.groupby('evolution_time')['QRC_final_test_acc'].mean().sort_index() * 100
    print("\nEffect of Evolution Time:")
    for time, acc in evol_time_effect.items():
        print(f"  {time} time units: {acc:.2f}%")
    
    # Print information about variability
    print("\nParameter Stability Analysis:")
    print("-" * 60)
    time_steps_std = df.groupby('time_steps')['QRC_final_test_acc'].std() * 100
    evol_time_std = df.groupby('evolution_time')['QRC_final_test_acc'].std() * 100
    
    print(f"Time steps variability (std dev): {time_steps_std.mean():.2f}%")
    print(f"Evolution time variability (std dev): {evol_time_std.mean():.2f}%")
    
    # Determine which parameter has more influence
    time_steps_range = time_steps_effect.max() - time_steps_effect.min()
    evol_time_range = evol_time_effect.max() - evol_time_effect.min()
    
    print(f"\nAccuracy range for Time Steps: {time_steps_range:.2f}%")
    print(f"Accuracy range for Evolution Time: {evol_time_range:.2f}%")
    
    if time_steps_range > evol_time_range:
        print("\nConclusion: Time Steps has a stronger effect on performance")
    elif evol_time_range > time_steps_range:
        print("\nConclusion: Evolution Time has a stronger effect on performance")
    else:
        print("\nConclusion: Both parameters have similar effects on performance")

def main():
    """Main function to execute analysis"""
    parser = argparse.ArgumentParser(description='Analyze time steps and evolution time effects on QRC performance')
    parser.add_argument('--results', type=str, 
                      default="/home/nbatista/GIC-quAI-QRC/results/generated/time_steps/evolution_time/time_steps_evolution_time_20250522_012322/all_results.csv",
                      help='Path to the results CSV file')
    parser.add_argument('--param-grid', type=str, default="/home/nbatista/GIC-quAI-QRC/results/generated/time_steps/evolution_time/time_steps_evolution_time_20250522_012322/param_grid.json",
                      help='Path to the parameter grid JSON file')
    parser.add_argument('--output', type=str, 
                      default="/home/nbatista/GIC-quAI-QRC/results/figures/generated/time_steps_evolution_time_heatmap.pdf",
                      help='Path to save the output figure (PDF)')
    
    args = parser.parse_args()
    
    # Load data
    df, param_grid = load_data(args.results, args.param_grid)
    
    # Create heatmap
    create_heatmap(df, param_grid, args.output)
    
    # Analyze optimal parameters
    analyze_optimal_parameters(df)

if __name__ == "__main__":
    main()
