import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
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

def transform_lambda_to_mu(lambda_val):
    """Transform lambda to mu using mu = sqrt(lambda/(1-lambda))."""
    # Handle edge cases
    # if lambda_val >= 0.999999:  # Treat values very close to 1 as a large finite number
        # return 1000.0  # Use: a large value instead of infinity
    if lambda_val == 0.999:
        return 15
    if lambda_val == 0.9999:
        return 30
    if lambda_val == 0.99999:
        return 50
    if lambda_val == 0.999999:
        return 75
    elif lambda_val >= 1.0:
        return 100
    
    elif lambda_val <= 0.0:
        return 0.0
    else:
        return np.sqrt(lambda_val / (1 - lambda_val))

def plot_guided_lambda_effect(df: pd.DataFrame, output_path: Optional[str] = None):
    """Create a plot showing guided_lambda vs QRC test accuracy with three separate views."""
    # Set up aesthetics
    sns.set_style("whitegrid")
    
    # Sort by guided_lambda value
    df_sorted = df.sort_values(by='guided_lambda')
    
    # Create a figure with 3 subplots horizontally arranged
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Filter data for each range
    df_small = df_sorted[(df_sorted['guided_lambda'] > 0) & (df_sorted['guided_lambda'] <= 0.1)]
    df_mid = df_sorted[(df_sorted['guided_lambda'] >= 0.1) & (df_sorted['guided_lambda'] <= 0.9)]
    df_large = df_sorted[(df_sorted['guided_lambda'] >= 0.9) & (df_sorted['guided_lambda'] <= 1.0)]
    df_zero = df_sorted[df_sorted['guided_lambda'] == 0]
    
    # 1. Plot small values (0 to 10^-1) on log scale
    ax = axes[0]
    if not df_small.empty:
        ax.plot(df_small['guided_lambda'], df_small['QRC_final_test_acc'] * 100, 
                '-o', linewidth=2, markersize=6, color='#1f77b4')
        
        # Add value labels
        for x, y in zip(df_small['guided_lambda'], df_small['QRC_final_test_acc'] * 100):
            ax.annotate(f"{y:.1f}%", (x, y), xytext=(0, 5), 
                        textcoords="offset points", ha='center', fontsize=8)
    
    # Handle zero value specially
    if not df_zero.empty:
        for _, row in df_zero.iterrows():
            if not df_small.empty:
                min_val = df_small['guided_lambda'].min() * 0.5
            else:
                min_val = 1e-8
            ax.scatter(min_val, row['QRC_final_test_acc'] * 100, 
                      s=80, color='red', marker='x', zorder=6)
            ax.annotate(f"λ=0: {row['QRC_final_test_acc'] * 100:.1f}%", 
                      (min_val, row['QRC_final_test_acc'] * 100), 
                      xytext=(5, 0), textcoords="offset points", fontsize=8)
    
    ax.set_xscale('log')
    ax.set_xlabel('λ: 0 to 0.1 (log scale)', fontsize=11)
    ax.set_ylabel('QRC Test Accuracy (%)', fontsize=11)
    ax.set_title('Small λ Values', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Plot mid-range values (0.1 to 0.9) on linear scale
    ax = axes[1]
    if not df_mid.empty:
        ax.plot(df_mid['guided_lambda'], df_mid['QRC_final_test_acc'] * 100, 
                '-o', linewidth=2, markersize=6, color='#2ca02c')
        
        # Add value labels
        for x, y in zip(df_mid['guided_lambda'], df_mid['QRC_final_test_acc'] * 100):
            ax.annotate(f"{y:.1f}%", (x, y), xytext=(0, 5), 
                        textcoords="offset points", ha='center', fontsize=8)
    
    ax.set_xlabel('λ: 0.1 to 0.9 (linear scale)', fontsize=11)
    ax.set_title('Mid-range λ Values', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 3. Plot values close to 1 (0.9 to 0.999999) on transformed scale
    ax = axes[2]
    if not df_large.empty:
        # Transform x values to show small differences near 1
        # Using (1-x) and log scale to spread out values near 1
        transformed_x = 1 - df_large['guided_lambda']
        
        ax.plot(transformed_x, df_large['QRC_final_test_acc'] * 100, 
                '-o', linewidth=2, markersize=6, color='#d62728')
        
        # Add value labels with original lambda values
        for i, (tx, y) in enumerate(zip(transformed_x, df_large['QRC_final_test_acc'] * 100)):
            orig_x = df_large['guided_lambda'].iloc[i]
            ax.annotate(f"{orig_x:.6f}\n{y:.1f}%", (tx, y), xytext=(0, 5), 
                        textcoords="offset points", ha='center', fontsize=8)
    
    ax.set_xscale('log')
    ax.set_xlabel('1-λ: Values from 0.9 to 1.0 (reverse log)', fontsize=11)
    ax.set_title('λ Values Close to 1', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    # Invert x-axis to show values increasing toward 1.0
    ax.invert_xaxis()
    
    # Adjust layout
    plt.tight_layout()
    
    # Add overall title
    fig.suptitle('Effect of Guided Lambda (λ) on QRC Accuracy', 
                fontsize=14, y=1.05)
    
    # Save figure if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    
    plt.show()
    return fig

def plot_guided_mu_effect(df: pd.DataFrame, output_path: Optional[str] = None):
    """Create a plot showing mu (transformed lambda) vs QRC test accuracy."""
    # Set up aesthetics
    sns.set_style("whitegrid")
    
    # Sort by guided_lambda value
    df_sorted = df.sort_values(by='guided_lambda')
    
    # Create mu values
    df_sorted['mu'] = df_sorted['guided_lambda'].apply(transform_lambda_to_mu)
    
    # Create a figure with appropriate size
    plt.figure(figsize=(12, 7))
    
    # Handle zero values separately since they won't appear on log scale
    df_nonzero = df_sorted[df_sorted['mu'] > 0]
    df_zero = df_sorted[df_sorted['mu'] == 0]
    
    # Plot line for non-zero values
    if not df_nonzero.empty:
        plt.plot(df_nonzero['mu'], df_nonzero['QRC_final_test_acc'] * 100, 
                '-o', linewidth=2, markersize=8, color='#1f77b4')
        
        # Anti-collision system for labels
        # Keep track of used label positions
        used_positions = []
        min_y_distance = 3.0  # Minimum vertical distance between labels in data units
        min_x_distance = 0.2  # Minimum horizontal distance factor (relative to x value)
        
        # First, identify dense regions to alternate labels
        # Process points from left to right (easier to avoid collisions)
        df_labeled = df_nonzero.sort_values(by='mu')
        
        # Add value labels for lambda with cleaned formatting (no trailing zeros)
        for i, (x, y) in enumerate(zip(df_labeled['mu'], df_labeled['QRC_final_test_acc'] * 100)):
            lambda_val = df_labeled['guided_lambda'].iloc[i]
            # Format lambda value to remove trailing zeros
            lambda_str = str(lambda_val)
            if '.' in lambda_str:
                lambda_str = lambda_str.rstrip('0').rstrip('.' if lambda_str.endswith('.') else '')
            
            # Default position (above the point)
            y_offset = 10
            x_offset = 0
            
            # For points very close to y-axis, offset to the right
            if x < 0.05:  # Adjust threshold as needed
                x_offset = 15
                y_offset = 0
            
            # Check for potential overlap with existing labels
            # Try alternating above/below placement if there are nearby labels
            position_found = False
            above_attempts = 0
            below_attempts = 0
            
            # Try positions until a good one is found
            while not position_found and (above_attempts < 3 or below_attempts < 3):
                current_y = y
                # Try positions above the point first
                if above_attempts <= below_attempts:
                    current_y = y + above_attempts * min_y_distance
                    y_offset = 10 + above_attempts * 5
                    above_attempts += 1
                else:
                    # Then try below if above is crowded
                    current_y = y - below_attempts * min_y_distance
                    y_offset = -20 - below_attempts * 5  # Negative offset to place below
                    below_attempts += 1
                
                # Check for collision
                collision = False
                for pos_x, pos_y in used_positions:
                    # Consider both x and y distances for collision
                    # For x-distance, use logarithmic scaling since we're on a log plot
                    log_x_distance = abs(np.log10(pos_x) - np.log10(x)) if x > 0 and pos_x > 0 else 1
                    y_distance = abs(pos_y - current_y)
                    
                    # Collision occurs if both x and y are too close
                    if log_x_distance < 0.2 and y_distance < min_y_distance:
                        collision = True
                        break
                
                if not collision:
                    position_found = True
            
            # Set horizontal alignment based on position
            ha = 'center'
            if x_offset > 0:
                ha = 'left'
            elif y_offset < 0:  # If label is below, center it
                ha = 'center'
            
            # Add the label at adjusted position
            plt.annotate(f"λ={lambda_str}\n{y:.1f}%", 
                       (x, y), xytext=(x_offset, y_offset), 
                       textcoords="offset points", 
                       ha=ha, va='bottom' if y_offset > 0 else 'top',
                       fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7),
                       zorder=10)  # Higher zorder to appear on top of axes
            
            # Remember this position
            used_positions.append((x, current_y))
    
    # Handle zero value specially with an offset to avoid axis overlap
    if not df_zero.empty:
        for _, row in df_zero.iterrows():
            min_mu = 0.015  # Increased minimum to avoid axis
            if not df_nonzero.empty:
                min_mu = max(0.015, df_nonzero['mu'].min() * 0.15)
                
            plt.scatter(min_mu, row['QRC_final_test_acc'] * 100, 
                      s=100, color='red', marker='x', zorder=6)
            plt.annotate(f"λ=0: {row['QRC_final_test_acc'] * 100:.1f}%", 
                       (min_mu, row['QRC_final_test_acc'] * 100), 
                       xytext=(20, 0), # Further increased offset to avoid axis
                       textcoords="offset points", fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.7),
                       zorder=10)  # Higher zorder to appear on top of axes
                       
    # Set x-axis to log scale
    plt.xscale('log')
    
    # Set safe x-axis limits with more margin on the left
    if not df_nonzero.empty:
        # Filter out any potential infinite values
        finite_mu = df_nonzero[np.isfinite(df_nonzero['mu'])]['mu']
        if not finite_mu.empty:
            x_min = max(0.005, finite_mu.min() * 0.2)  # More left margin
            
            # Extend the upper limit to 10^2 (100) as requested
            x_max = 100  # Set fixed upper limit to 10^2
                
            plt.xlim(x_min, x_max)
    else:
        # Default limits if no non-zero data
        plt.xlim(0.001, 100)  # Upper limit to 10^2
    
    # Set y-axis limits to 0-100%
    plt.ylim(70, 95)
    
    # Add secondary x-axis with lambda values
    ax1 = plt.gca()
    ax2 = ax1.twiny()
    
    # Create lambda ticks at meaningful points with more detail for high lambda values
    # Include values that map to mu values close to 100
    lambda_ticks = [0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999]
    mu_ticks = [transform_lambda_to_mu(l) for l in lambda_ticks]
    
    # Add explicit mu ticks to ensure we have tick marks at powers of 10
    mu_major_ticks = [0.01, 0.1, 1, 10, 100]
    ax1.set_xticks(mu_major_ticks)
    
    # Custom formatter to show infinity symbol for the last tick (100)
    from matplotlib.ticker import FuncFormatter
    def mu_formatter(x, pos):
        if x == 100:
            return r'$\infty$'  # Infinity symbol in LaTeX format
        return f"{x:.0f}" if x >= 1 else f"{x:.2f}".rstrip('0').rstrip('.')
    
    ax1.xaxis.set_major_formatter(FuncFormatter(mu_formatter))
    
    # Create custom tick labels with formatting based on value
    tick_labels = []
    for l in lambda_ticks:
        if l >= 0.999:
            # For very high values, show number of 9s
            num_9s = str(l).count('9')
            tick_labels.append(f"λ=0.{'9'*num_9s}")
        elif l == 0.999999:  # The last lambda value
            tick_labels.append("λ=1")  # Label it as λ=1
        else:
            # Remove trailing zeros for cleaner display
            l_str = str(l)
            if '.' in l_str:
                l_str = l_str.rstrip('0').rstrip('.' if l_str.endswith('.') else '')
            tick_labels.append(f"λ={l_str}")
    
    ax2.set_xscale('log')
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(mu_ticks)
    ax2.set_xticklabels(tick_labels, rotation=45 if any(l >= 0.999 for l in lambda_ticks) else 0)
    
    # Highlight low lambda region (0 to 0.1) with a subtle background color
    if any(l <= 0.1 for l in df_sorted['guided_lambda']):
        low_lambda_mu_min = ax1.get_xlim()[0]
        low_lambda_mu_max = transform_lambda_to_mu(0.1)
        ax1.axvspan(low_lambda_mu_min, low_lambda_mu_max, alpha=0.1, color='blue', zorder=0)
        
        # Add annotation explaining the low lambda region
        ax1.text(low_lambda_mu_min * 1.5, ax1.get_ylim()[0] + (ax1.get_ylim()[1]-ax1.get_ylim()[0])*0.05, 
                "Low λ region\n(0-0.1)", 
                fontsize=8, color='darkblue', alpha=0.8, ha='left', va='bottom')
    
    # Highlight high lambda region (0.9 to 1.0) with a subtle background color
    if any(l >= 0.9 for l in df_sorted['guided_lambda']):
        high_lambda_mu_min = transform_lambda_to_mu(0.9)  # Changed from 0.99 to 0.9
        high_lambda_mu_max = ax1.get_xlim()[1]
        ax1.axvspan(high_lambda_mu_min, high_lambda_mu_max, alpha=0.1, color='red', zorder=0)
        
        # Add annotation explaining the high lambda region
        ax1.text(high_lambda_mu_min * 1.1, ax1.get_ylim()[0] + (ax1.get_ylim()[1]-ax1.get_ylim()[0])*0.15, 
                "High λ region\n(0.9-1.0)",  # Changed range description
                fontsize=8, color='darkred', alpha=0.8, ha='left', va='bottom')
    
    # Labels and title
    plt.xlabel('μ = sqrt(λ/(1-λ)) [log scale]', fontsize=12)
    plt.ylabel('QRC Test Accuracy (%)', fontsize=12)
    plt.title('Effect of Guided Lambda on QRC Accuracy (μ-space)', fontsize=14)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        base_path, ext = os.path.splitext(output_path)
        mu_output_path = f"{base_path}_mu{ext}"
        os.makedirs(os.path.dirname(mu_output_path), exist_ok=True)
        plt.savefig(mu_output_path, format='pdf', bbox_inches='tight')
        print(f"Figure saved to {mu_output_path}")
    
    plt.show()
    return plt.gcf()

def main():
    """Main function to analyze guided_lambda results"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze guided lambda parameter sweep results')
    parser.add_argument('--use-mu', action='store_true', 
                        help='Use mu transformation (sqrt(lambda/(1-lambda)) instead of lambda')
    args = parser.parse_args()
    
    # Define file path
    results_file = "/home/nbatista/GIC-quAI-QRC/results/generated/guided_lambda/log_scale_lambda_all/guided_autoencoder_lambda_20250527_135753/all_results.csv"
    
    # Load results
    df = load_sweep_results(results_file)
    
    # Print data overview
    print("\nData Overview:")
    print("-" * 40)
    print(df[['guided_lambda', 'QRC_final_test_acc']].describe())
    
    # Plot results
    base_output_path = "/home/nbatista/GIC-quAI-QRC/results/figures/guided_lambda_log_all_effect.pdf"
    
    # Choose plot function based on command line argument
    if args.use_mu:
        fig = plot_guided_mu_effect(df, base_output_path)
    else:
        fig = plot_guided_lambda_effect(df, base_output_path)
    
    # Print optimal guided_lambda value
    optimal_idx = df['QRC_final_test_acc'].idxmax()
    optimal_lambda = df.loc[optimal_idx, 'guided_lambda']
    optimal_acc = df.loc[optimal_idx, 'QRC_final_test_acc'] * 100
    
    print(f"\nOptimal guided_lambda value: {optimal_lambda}")
    print(f"Highest QRC test accuracy: {optimal_acc:.2f}%")
    
    # If using mu, also print the optimal mu value
    if args.use_mu:
        optimal_mu = transform_lambda_to_mu(optimal_lambda)
        print(f"Optimal mu value: {optimal_mu:.6f}")

if __name__ == "__main__":
    main()