import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Optional
import argparse

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

def plot_quantum_update_frequency_effect(df: pd.DataFrame, output_path: Optional[str] = None):
    """Create a plot showing quantum update frequency vs QRC accuracies."""
    # Set up aesthetics
    sns.set_style("whitegrid")
    
    # Sort by update frequency value
    df_sorted = df.sort_values(by='quantum_update_frequency')
    
    # Convert accuracies to percentage
    train_acc = df_sorted['QRC_final_train_acc'].values * 100
    test_acc = df_sorted['QRC_final_test_acc'].values * 100
    x_values = df_sorted['quantum_update_frequency'].values
    
    # Create a figure with appropriate size
    plt.figure(figsize=(10, 6))
    
    # Plot test accuracy (solid line)
    plt.plot(x_values, test_acc, '-', linewidth=2, color='#1f77b4', label='Test Accuracy')
    plt.scatter(x_values, test_acc, s=80, color='#1f77b4', zorder=5)
    
    # Plot train accuracy (dashed line)
    plt.plot(x_values, train_acc, '--', linewidth=2, color='#ff7f0e', label='Training Accuracy')
    plt.scatter(x_values, train_acc, s=80, color='#ff7f0e', zorder=5, marker='s')
    
    # Add value labels for test accuracy
    for i, (x, y) in enumerate(zip(x_values, test_acc)):
        plt.annotate(f"{y:.1f}%", (x, y), xytext=(0, 10), 
                    textcoords="offset points", ha='center')
    
    # Special handling for x=0 (no quantum update)
    if 0 in x_values:
        plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
        plt.text(0, plt.ylim()[0] + 5, 'No Quantum\nUpdate', 
                ha='center', va='bottom', fontsize=8, bbox=dict(facecolor='white', alpha=0.8))
    
    # Labels and title
    plt.xlabel('Quantum Update Frequency', fontsize=12)
    plt.ylabel('QRC Accuracy (%)', fontsize=12)
    plt.title("Effect of Quantum Update Frequency on QRC Performance", fontsize=14)
    
    # Add legend
    plt.legend(loc='lower right')
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set x-axis ticks to match actual data points
    plt.xticks(x_values)
    
    # Set y-axis limits to focus on the relevant range
    y_min = max(70, np.floor(min(train_acc.min(), test_acc.min()) - 5))
    y_max = min(101, np.ceil(max(train_acc.max(), test_acc.max()) + 3))
    plt.ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    
    plt.show()

def analyze_quantum_update_effect(df: pd.DataFrame):
    """Analyze the effect of quantum update frequency on performance."""
    print("\nAnalysis of Quantum Update Frequency Effects:")
    print("-" * 70)
    
    # Sort by update frequency
    df_sorted = df.sort_values(by='quantum_update_frequency')
    
    # Print header
    print(f"{'Update Freq':<12} | {'Test Acc':<10} | {'Train Acc':<10} | {'Gap':<10} | {'Observation'}")
    print("-" * 70)
    
    # Calculate gap between train and test accuracy
    for _, row in df_sorted.iterrows():
        freq = row['quantum_update_frequency']
        test_acc = row['QRC_final_test_acc'] * 100
        train_acc = row['QRC_final_train_acc'] * 100
        gap = train_acc - test_acc
        
        # Generate observation
        if freq == 0:
            observation = "No quantum updates - baseline"
        elif gap > 15:
            observation = "Significant overfitting"
        elif gap < 5 and test_acc > 90:
            observation = "Good generalization"
        else:
            observation = ""
        
        print(f"{freq:<12} | {test_acc:<10.1f} | {train_acc:<10.1f} | {gap:<10.1f} | {observation}")
    
    # Find optimal frequency
    optimal_idx = df['QRC_final_test_acc'].idxmax()
    optimal_freq = df.loc[optimal_idx, 'quantum_update_frequency']
    optimal_acc = df.loc[optimal_idx, 'QRC_final_test_acc'] * 100
    
    print("\nOptimal Quantum Update Frequency Analysis:")
    print("-" * 70)
    print(f"Optimal frequency: {optimal_freq}")
    print(f"Highest test accuracy: {optimal_acc:.1f}%")
    
    # Find zero frequency row for comparison
    if 0 in df['quantum_update_frequency'].values:
        zero_freq_row = df[df['quantum_update_frequency'] == 0].iloc[0]
        zero_freq_acc = zero_freq_row['QRC_final_test_acc'] * 100
        improvement = optimal_acc - zero_freq_acc
        print(f"Improvement over no quantum updates: {improvement:.1f}%")

def main():
    """Main function to analyze quantum update frequency results"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze quantum update frequency effect on QRC performance')
    parser.add_argument('--results', type=str, 
                      default="/home/nbatista/GIC-quAI-QRC/results/generated/guided_autoencoder_update_frequency/guided_autoencoder_update_frequency_20250520_160624_20250520_160624/all_results.csv",
                      help='Path to the results CSV file')
    parser.add_argument('--output', type=str, 
                      default="/home/nbatista/GIC-quAI-QRC/results/figures/generated/quantum_update_frequency_effect.png",
                      help='Path to save the output figure')
    
    args = parser.parse_args()
    
    # Load results
    df = load_sweep_results(args.results)
    
    # Print data overview
    print("\nData Overview:")
    print("-" * 40)
    print(df[['quantum_update_frequency', 'QRC_final_test_acc', 'QRC_final_train_acc']].describe())
    
    # Plot results
    plot_quantum_update_frequency_effect(df, args.output)
    
    # Analyze effect
    analyze_quantum_update_effect(df)

if __name__ == "__main__":
    main()
