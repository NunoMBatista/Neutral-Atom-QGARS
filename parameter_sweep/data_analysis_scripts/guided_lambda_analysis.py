import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

def load_sweep_results(filepath: str) -> pd.DataFrame:
    """Load parameter sweep results from CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} experiment results from {filepath}")
    except Exception as e:
        print(f"Error loading results: {e}")
        df = pd.DataFrame()
    
    return df

# def plot_guided_lambda_effect(df: pd.DataFrame, output_path: Optional[str] = None):
    # """Create a plot showing guided_lambda vs QRC test accuracy."""
    # # Set up aesthetics
    # sns.set_style("whitegrid")
    
    # # Sort by guided_lambda value
    # df_sorted = df.sort_values(by='guided_lambda')
    
    # # Convert accuracy to percentage
    # y_values = df_sorted['QRC_final_test_acc'].values * 100
    # x_values = df_sorted['guided_lambda'].values
    
    # # Create a figure with appropriate size
    # plt.figure(figsize=(10, 6))
    
    # # Plot line connecting points
    # plt.plot(x_values, y_values, '-', linewidth=2, color='#1f77b4')
    
    # # Plot individual points
    # plt.scatter(x_values, y_values, s=80, color='#1f77b4', zorder=5)
    
    # # Add value labels
    # for i, (x, y) in enumerate(zip(x_values, y_values)):
    #     plt.annotate(f"{y:.1f}%", (x, y), xytext=(0, 10), 
    #                 textcoords="offset points", ha='center')
    
    # # Labels and title
    # plt.xlabel('Guided Lambda', fontsize=12)
    # plt.ylabel('QRC Test Accuracy (%)', fontsize=12)
    # plt.title("Effect of Guided Lambda on QRC Accuracy", fontsize=14)
    
    # # Add grid for better readability
    # plt.grid(True, linestyle='--', alpha=0.7)
    
    # plt.tight_layout()
    
    # # Save figure if output path is provided
    # if output_path:
    #     os.makedirs(os.path.dirname(output_path), exist_ok=True)
    #     plt.savefig(output_path, dpi=300, bbox_inches='tight')
    #     print(f"Figure saved to {output_path}")
    
    # plt.show()

def plot_guided_lambda_effect(df: pd.DataFrame, output_path: Optional[str] = None):
    """Create a plot showing guided_lambda vs QRC test accuracy."""
    # Set up aesthetics
    sns.set_style("whitegrid")
    
    # Sort by guided_lambda value
    df_sorted = df.sort_values(by='guided_lambda')
    
    # Convert accuracy to percentage
    y_values = df_sorted['QRC_final_test_acc'].values * 100
    x_values = df_sorted['guided_lambda'].values
    
    # Create a figure with appropriate size
    plt.figure(figsize=(10, 6))
    
    # Plot line connecting points
    plt.plot(x_values, y_values, '-', linewidth=2, color='#1f77b4')
    
    # Plot individual points
    plt.scatter(x_values, y_values, s=80, color='#1f77b4', zorder=5)
    
    # Add value labels
    for i, (x, y) in enumerate(zip(x_values, y_values)):
        plt.annotate(f"{y:.1f}%", (x, y), xytext=(0, 10), 
                    textcoords="offset points", ha='center')
    
    # Set x-axis to log scale
    plt.xscale('log')
    
    # Labels and title
    plt.xlabel('Guided Lambda (log scale)', fontsize=12)
    plt.ylabel('QRC Test Accuracy (%)', fontsize=12)
    plt.title("Effect of Guided Lambda on QRC Accuracy", fontsize=14)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    
    plt.show()


def main():
    """Main function to analyze guided_lambda results"""
    # Define file path
    results_file = "../results/generated/guided_lambda/log_scale_lambda/guided_autoencoder_lambda_20250523_181956/all_results.csv"
    results_file = "/home/nbatista/GIC-quAI-QRC/results/generated/guided_lambda/log_scale_lambda/guided_autoencoder_lambda_20250523_181956/all_results.csv"
    # Load results
    df = load_sweep_results(results_file)
    
    # Print data overview
    print("\nData Overview:")
    print("-" * 40)
    print(df[['guided_lambda', 'QRC_final_test_acc']].describe())
    
    # Plot results
    output_path = "/home/nbatista/GIC-quAI-QRC/results/figures/guided_lambda_log_effect.png"
    
    # Create plot
    plot_guided_lambda_effect(df, output_path)
    
    # Print optimal guided_lambda value
    optimal_idx = df['QRC_final_test_acc'].idxmax()
    optimal_lambda = df.loc[optimal_idx, 'guided_lambda']
    optimal_acc = df.loc[optimal_idx, 'QRC_final_test_acc'] * 100
    
    print(f"\nOptimal guided_lambda value: {optimal_lambda}")
    print(f"Highest QRC test accuracy: {optimal_acc:.2f}%")

if __name__ == "__main__":
    main()