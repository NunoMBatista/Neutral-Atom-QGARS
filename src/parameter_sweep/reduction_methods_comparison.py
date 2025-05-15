import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def plot_qrc_accuracy_comparison(csv_path):
    """
    Plot QRC accuracy comparison for different dimension reduction methods.
    
    Parameters:
    -----------
    csv_path : str
        Path to the all_results.csv file
    """
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Filter successful experiments
    df = df[df['status'] == 'success']
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Define colors and markers for each method
    styles = {
        'pca': {'color': 'blue', 'marker': 'o', 'label': 'PCA'},
        'autoencoder': {'color': 'green', 'marker': 's', 'label': 'Autoencoder'},
        'guided_autoencoder': {'color': 'red', 'marker': '^', 'label': 'Guided Autoencoder'}
    }
    
    # For each reduction method, plot a line
    for method, style in styles.items():
        # Filter data for this method
        method_data = df[df['reduction_method'] == method]
        
        if not method_data.empty:
            # Sort by dimension
            method_data = method_data.sort_values('dim_reduction')
            
            # Plot the line
            plt.plot(method_data['dim_reduction'], method_data['QRC_final_test_acc'], 
                     marker=style['marker'], color=style['color'], 
                     label=style['label'], linewidth=2, markersize=8)
    
    # Add labels and title
    plt.xlabel('Dimension Reduction', fontsize=12)
    plt.ylabel('QRC Accuracy', fontsize=12)
    plt.title('QRC Accuracy vs Dimension Reduction by Method', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set axes limits
    plt.ylim(0, 1.0)
    plt.xticks(np.arange(4, 14, 2))  # 4, 6, 8, 10, 12
    
    # Add a note about PCA limitations
    plt.annotate('Note: PCA failed for dimensions > 4\ndue to sample size limitations', 
                 xy=(0.5, 0.02), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8),
                 ha='center', fontsize=10)
    
    # Save figure
    output_dir = os.path.dirname(csv_path)
    plt.savefig(os.path.join(output_dir, 'qrc_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Find the most recent results file
    results_dir = "./results"
    subdirs = glob.glob(os.path.join(results_dir, "*"))
    if subdirs:
        latest_subdir = max(subdirs, key=os.path.getmtime)
        csv_path = os.path.join(latest_subdir, "all_results.csv")
        
        if os.path.exists(csv_path):
            print(f"Plotting results from: {csv_path}")
            plot_qrc_accuracy_comparison(csv_path)
        else:
            print(f"No results CSV found in {latest_subdir}")
    else:
        print(f"No result directories found in {results_dir}")