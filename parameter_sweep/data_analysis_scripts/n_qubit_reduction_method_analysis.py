import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_qrc_performance_by_method(results_file, output_dir):
    """
    Plot QRC performance (train and test accuracy) for different reduction methods
    across varying number of qubits.
    
    Parameters
    ----------
    results_file : str
        Path to the CSV file containing the results
    output_dir : str
        Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV file
    print(f"Reading data from {results_file}")
    df = pd.read_csv(results_file)
    
    # Filter only successful runs
    df = df[df['status'] == 'success']
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Process each reduction method with distinct colors
    methods = {
        'pca': {'color': 'blue', 'label': 'PCA'},
        'autoencoder': {'color': 'red', 'label': 'Autoencoder'},
        'guided_autoencoder': {'color': 'green', 'label': 'Guided Autoencoder'}
    }
    
    for method, properties in methods.items():
        # Filter for this reduction method
        method_df = df[df['reduction_method'] == method]
        
        if method_df.empty:
            print(f"No data for {method}")
            continue
            
        # Group by number of qubits and compute mean if there are multiple entries
        grouped = method_df.groupby('dim_reduction').agg({
            'QRC_final_test_acc': 'mean',
            'QRC_final_train_acc': 'mean'
        }).reset_index()
        
        # Sort by number of qubits for proper line plotting
        grouped = grouped.sort_values('dim_reduction')
        
        # Plot test accuracy (solid line)
        plt.plot(
            grouped['dim_reduction'], 
            grouped['QRC_final_test_acc'], 
            color=properties['color'], 
            marker='o', 
            linestyle='-',
            label=f"{properties['label']} Test"
        )
        
        # Plot train accuracy (dashed line)
        plt.plot(
            grouped['dim_reduction'], 
            grouped['QRC_final_train_acc'], 
            color=properties['color'], 
            marker='s', 
            linestyle='--',
            label=f"{properties['label']} Train"
        )
    
    # Add grid and labels
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Number of Qubits (dim_reduction)', fontsize=12)
    plt.ylabel('QRC Accuracy', fontsize=12)
    plt.title('QRC Performance by Reduction Method and Qubit Count', fontsize=14)
    
    # Set y-axis limits between 0.5 and 1.0 for better visualization
    plt.ylim(0.5, 1.0)
    
    # Create integer ticks for x-axis based on available data
    all_dims = sorted(df['dim_reduction'].unique())
    plt.xticks(all_dims)
    
    # Add legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    
    # Save the figure
    output_file = os.path.join(output_dir, 'qrc_performance_by_reduction_method.pdf')
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    print(f"Figure saved to {output_file}")
    
    # Show the plot
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze QRC performance by reduction method and qubit count')
    parser.add_argument('results', help='all_results.csv written by the encoding_dimensions profile')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save the output figure (default: next to the results file)')
    
    args = parser.parse_args()
    output_dir = os.path.abspath(args.output_dir or os.path.dirname(args.results))
    
    plot_qrc_performance_by_method(args.results, output_dir)


if __name__ == "__main__":
    main()
