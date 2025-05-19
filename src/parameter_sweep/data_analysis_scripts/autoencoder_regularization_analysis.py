import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

def plot_autoencoder_regularization_effect(df: pd.DataFrame, output_path: Optional[str] = None):
    """Create a simple plot showing regularization vs accuracy."""
    # Set up aesthetics
    sns.set_style("whitegrid")
    
    # Sort by regularization value
    df_sorted = df.sort_values(by='autoencoder_regularization')
    
    # Convert accuracy to percentage
    y_values = df_sorted['QRC_final_test_acc'].values * 100
    x_values = df_sorted['autoencoder_regularization'].values
    
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
    
    # Format x-axis to show values properly
    if 0 in x_values:
        # Use a semi-log scale with special handling for zero
        plt.xscale('symlog', linthresh=1e-6)  # symlog handles zero values
        
        # Custom x-ticks for all values including zero
        plt.xticks(x_values)
        
        # Set custom formatter for x-axis labels
        from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        plt.gca().xaxis.set_major_formatter(formatter)
    else:
        # If no zeros, use regular log scale
        plt.xscale('log')
        plt.xticks(x_values)  # Set ticks at actual data points
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%g'))  # Format without unnecessary zeros
    
    # Ensure tick labels are visible by adjusting them
    plt.gcf().canvas.draw()
    labels = [item.get_text() for item in plt.gca().get_xticklabels()]
    
    # Add explicit labels for each point
    for i, (x_val, label) in enumerate(zip(x_values, labels)):
        if x_val == 0:
            plt.gca().get_xticklabels()[i].set_text('0')
        elif x_val == 1e-5:
            plt.gca().get_xticklabels()[i].set_text('0.00001')
        elif x_val == 1e-4:
            plt.gca().get_xticklabels()[i].set_text('0.0001')
        elif x_val == 1e-3:
            plt.gca().get_xticklabels()[i].set_text('0.001')
        elif x_val == 1e-2:
            plt.gca().get_xticklabels()[i].set_text('0.01')
    
    # Adjust y-axis to start slightly below minimum value
    y_min = max(75, np.floor(y_values.min() - 3))
    y_max = min(100, np.ceil(y_values.max() + 3))
    plt.ylim(y_min, y_max)
    
    # Labels and title
    plt.xlabel('Autoencoder Regularization (Weight Decay)', fontsize=12)
    plt.ylabel('QRC Test Accuracy (%)', fontsize=12)
    plt.title("Effect of Autoencoder Regularization on QRC Accuracy for the Generated Polyp Dataset", fontsize=14)
    
    # Add grid for better readability
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Rotate x-tick labels for better readability
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    
    plt.show()

def main():
    """Main function to analyze autoencoder regularization results"""
    # Define file path
    results_file = "../results/generated/ae_regularization/autoencoder_regularization_20250519_032530_20250519_032530/all_results.csv"
    
    # Load results
    df = load_sweep_results(results_file)
    
    # Print data overview
    print("\nData Overview:")
    print("-" * 40)
    print(df[['autoencoder_regularization', 'QRC_final_test_acc']].describe())
    
    # Plot results
    output_path = "../results/figures/autoencoder_regularization_effect.png"
    
    # Create plot
    plot_autoencoder_regularization_effect(df, output_path)
    
    # Print optimal regularization value
    optimal_idx = df['QRC_final_test_acc'].idxmax()
    optimal_reg = df.loc[optimal_idx, 'autoencoder_regularization']
    optimal_acc = df.loc[optimal_idx, 'QRC_final_test_acc'] * 100
    
    print(f"\nOptimal autoencoder regularization value: {optimal_reg}")
    print(f"Highest QRC test accuracy: {optimal_acc:.2f}%")

if __name__ == "__main__":
    main()
