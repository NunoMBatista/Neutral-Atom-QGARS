import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, Any, Optional, List, Tuple
from matplotlib.ticker import MaxNLocator

def load_metrics(filepath: str) -> Dict[str, Any]:
    """Load metrics from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metrics file: {e}")
        return {}

def plot_losses(metrics: Dict[str, Any], 
                output_dir: str, 
                filename_prefix: str = "classifier", 
                log_scale: bool = True,
                smooth_window: int = 0,
                dpi: int = 300) -> Tuple[str, str]:
    """
    Plot the loss evolution for linear, QRC, and NN classifiers.
    
    Args:
        metrics: Dictionary containing classifier metrics
        output_dir: Directory to save the plots
        filename_prefix: Prefix for the output filenames
        log_scale: Whether to use log scale for y-axis
        smooth_window: Window size for moving average smoothing (0 for no smoothing)
        dpi: DPI for PNG output
        
    Returns:
        Tuple of (pdf_path, png_path) where the plots were saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the style for a professional look
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    
    # Create figure and axis with appropriate size
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors and line styles for different classifiers
    styles = {
        "linear": {"color": "#1f77b4", "linestyle": "-", "marker": "o", "markersize": 3, "markevery": 10},
        "QRC": {"color": "#d62728", "linestyle": "-", "marker": "s", "markersize": 3, "markevery": 10},
        "NN": {"color": "#2ca02c", "linestyle": "-", "marker": "^", "markersize": 3, "markevery": 10}
    }
    
    # Dictionary to store min loss values and epochs for annotation
    min_loss_data = {}
    
    # Apply smoothing function if window size > 0
    def smooth(y, window_size):
        if window_size <= 1:
            return y
        box = np.ones(window_size) / window_size
        return np.convolve(y, box, mode='same')
    
    # Plot losses for each classifier
    for classifier, style in styles.items():
        if classifier in metrics and "losses" in metrics[classifier]:
            losses = metrics[classifier]["losses"]
            epochs = np.arange(1, len(losses) + 1)
            
            # Apply smoothing if requested
            if smooth_window > 0:
                smoothed_losses = smooth(losses, smooth_window)
                # Plot original data with lower alpha
                ax.plot(epochs, losses, alpha=0.2, color=style["color"])
                # Plot smoothed data
                line, = ax.plot(epochs, smoothed_losses, 
                             color=style["color"], 
                             linestyle=style["linestyle"],
                             linewidth=2, 
                             label=f"{classifier.upper()}",
                             marker=style["marker"],
                             markersize=style["markersize"],
                             markevery=style["markevery"])
                # Find minimum loss in smoothed data
                min_idx = np.argmin(smoothed_losses)
                min_loss = smoothed_losses[min_idx]
            else:
                # Plot original data
                line, = ax.plot(epochs, losses, 
                             color=style["color"], 
                             linestyle=style["linestyle"],
                             linewidth=2, 
                             label=f"{classifier.upper()}",
                             marker=style["marker"],
                             markersize=style["markersize"],
                             markevery=style["markevery"])
                # Find minimum loss
                min_idx = np.argmin(losses)
                min_loss = losses[min_idx]
            
            # Store minimum loss data for annotation
            min_loss_data[classifier] = {
                "epoch": int(epochs[min_idx]),
                "loss": min_loss
            }
            
            # Highlight minimum point
            ax.scatter(epochs[min_idx], min_loss, color=style["color"], 
                     s=80, zorder=10, edgecolor='black', linewidth=1)
    
    # Set axis labels and title with good typography
    ax.set_xlabel("Epochs", fontsize=12, fontweight='bold')
    ax.set_ylabel("Loss", fontsize=12, fontweight='bold')
    ax.set_title("Loss Evolution During Training", fontsize=14, fontweight='bold')
    
    # Add legend with nice formatting
    legend = ax.legend(fontsize=10, frameon=True, fancybox=True, framealpha=0.8, 
                      loc='upper right', title="Classifiers")
    legend.get_title().set_fontweight('bold')
    
    # Add annotations for minimum loss points
    # Calculate offsets to prevent overlap
    annotation_offsets = {
        "linear": {"x": 10, "y": 10},
        "QRC": {"x": 10, "y": -30},
        "NN": {"x": -60, "y": 10}
    }
    
    for classifier, data in min_loss_data.items():
        style = styles[classifier]
        offset = annotation_offsets[classifier]
        
        ax.annotate(f"Min: {data['loss']:.4f}\nEpoch: {data['epoch']}",
                  xy=(data['epoch'], data['loss']),
                  xytext=(offset["x"], offset["y"]),
                  textcoords="offset points",
                  color=style["color"],
                  fontweight='bold',
                  fontsize=8,
                  arrowprops=dict(arrowstyle="->", color=style["color"], alpha=0.7),
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    # Set y-axis to logarithmic scale if requested and appropriate
    loss_values = []
    for classifier in metrics:
        if "losses" in metrics[classifier]:
            loss_values.extend(metrics[classifier]["losses"])
    
    if loss_values:
        loss_min, loss_max = min(loss_values), max(loss_values)
        
        # Cap y-axis to 1.0
        if log_scale and loss_max / loss_min > 5:  # Use log scale if range is wide
            ax.set_yscale('log')
            ax.set_ylabel("Loss (log scale)", fontsize=12, fontweight='bold')
            
            # Set y-limits with a cap at 1.0
            ax.set_ylim(min(loss_values) * 0.8, 1.0)
            
            # Format y-axis tick labels nicely in log scale
            from matplotlib.ticker import LogFormatter
            formatter = LogFormatter(labelOnlyBase=False)
            ax.yaxis.set_major_formatter(formatter)
        else:
            # For linear scale, also cap at 1.0
            ax.set_ylim(0, 1.0)
    
    # Add a grid for better readability
    ax.grid(True, which='major', linestyle='-', alpha=0.3)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)
    
    # Force integer x-ticks for epochs
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add subtle background shading for better visual appearance
    ax.set_facecolor('#f8f9fa')
    
    # Add a thin border around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.5)
    
    # Improve tick formatting
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save as PDF
    pdf_path = os.path.join(output_dir, f"{filename_prefix}_losses.pdf")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    
    # Save as PNG with specified DPI
    png_path = os.path.join(output_dir, f"{filename_prefix}_losses.png")
    plt.savefig(png_path, format="png", dpi=dpi, bbox_inches="tight")
    
    print(f"Plots saved to:")
    print(f"  - PDF: {pdf_path}")
    print(f"  - PNG: {png_path}")
    
    return pdf_path, png_path

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Plot classifier losses from metrics JSON file.')
    
    parser.add_argument('--input', '-i', type=str, required=False,
                        default="",
                        help='Path to the classifier_metrics.json file')
    
    parser.add_argument('--output-dir', '-o', type=str, required=False,
                        default="",
                        help='Directory to save the output plots')
    
    parser.add_argument('--prefix', '-p', type=str, default="classifier",
                        help='Prefix for output filenames')
    
    parser.add_argument('--log-scale', '-l', action='store_true',
                        help='Use logarithmic scale for y-axis')
    
    parser.add_argument('--smooth', '-s', type=int, default=0,
                        help='Window size for moving average smoothing (0 for no smoothing)')
    
    parser.add_argument('--dpi', '-d', type=int, default=300,
                        help='DPI for PNG output')
    
    return parser.parse_args()

def main():
    """Main function to load data and create plots."""
    args = parse_arguments()
    
    # Get input file path
    if args.input:
        file_path = args.input
    else:
        default_path = "/home/nbatista/GIC-quAI-QRC/results/generated/guided_autoencoder_update_frequency/guided_autoencoder_update_frequency_20250520_160624_20250520_160624/exp_0/classifier_metrics.json"
        file_path = input(f"Enter path to classifier_metrics.json file (default: {default_path}): ").strip()
        if not file_path:
            file_path = default_path
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return
    
    # Load metrics
    metrics = load_metrics(file_path)
    if not metrics:
        print("Error: Could not load metrics data.")
        return
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.path.dirname(file_path), "plots")
    
    # Plot losses
    plot_losses(
        metrics=metrics, 
        output_dir=output_dir, 
        filename_prefix=args.prefix,
        log_scale=args.log_scale,
        smooth_window=args.smooth,
        dpi=args.dpi
    )
    
    print("\nPlotting complete!")

if __name__ == "__main__":
    main()
