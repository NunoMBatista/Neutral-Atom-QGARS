import os
import pickle
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

def save_experiment_results(results: Dict[str, Tuple], filepath: str) -> None:
    """
    Save experiment results to a file.
    
    Parameters
    ----------
    results : Dict[str, Tuple]
        Results dictionary from main function
    filepath : str
        Path to save the results
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    # Save using pickle
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)

def load_experiment_results(filepath: str) -> Dict[str, Tuple]:
    """
    Load experiment results from a file.
    
    Parameters
    ----------
    filepath : str
        Path to the results file
        
    Returns
    -------
    Dict[str, Tuple]
        Results dictionary
    """
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    return results

def generate_learning_curves(results: Dict[str, Tuple], 
                           title: str = "Learning Curves",
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Generate learning curves from experiment results.
    
    Parameters
    ----------
    results : Dict[str, Tuple]
        Results dictionary from main function
    title : str, optional
        Plot title, by default "Learning Curves"
    save_path : Optional[str], optional
        Path to save the figure, by default None
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot training and test accuracy
    for model_name, (losses, train_acc, test_acc, _) in results.items():
        epochs = range(1, len(train_acc) + 1)
        ax1.plot(epochs, train_acc, 'o-', label=f"{model_name} (train)")
        ax1.plot(epochs, test_acc, 's--', label=f"{model_name} (test)")
    
    ax1.set_title('Training and Testing Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot training loss
    for model_name, (losses, _, _, _) in results.items():
        epochs = range(1, len(losses) + 1)
        ax2.plot(epochs, losses, '-', label=model_name)
    
    ax2.set_title('Training Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def extract_performance_summary(results_dir: str) -> pd.DataFrame:
    """
    Extract performance summary from a directory of experiment results.
    
    Parameters
    ----------
    results_dir : str
        Directory containing experiment results
        
    Returns
    -------
    pd.DataFrame
        DataFrame with performance summary
    """
    # Get all experiment directories
    exp_dirs = [d for d in os.listdir(results_dir) 
               if os.path.isdir(os.path.join(results_dir, d)) and d != "plots"]
    
    summaries = []
    for exp_dir in tqdm(exp_dirs, desc="Processing experiments"):
        metrics_path = os.path.join(results_dir, exp_dir, "metrics.json")
        config_path = os.path.join(results_dir, exp_dir, "config.json")
        
        # Skip if metrics file doesn't exist
        if not os.path.exists(metrics_path):
            continue
            
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
        # Add experiment ID
        metrics['experiment_id'] = exp_dir
        
        # Add configuration parameters if available
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Merge config into metrics, but don't overwrite existing metrics
            for k, v in config.items():
                if k not in metrics:
                    metrics[k] = v
        
        summaries.append(metrics)
    
    # Convert to DataFrame
    if summaries:
        return pd.DataFrame(summaries)
    else:
        return pd.DataFrame()

def generate_comparison_matrix(results_dir: str, 
                             output_path: Optional[str] = None) -> plt.Figure:
    """
    Generate a comparison matrix of model performance from experiment results.
    
    Parameters
    ----------
    results_dir : str
        Directory containing experiment results
    output_path : Optional[str], optional
        Path to save the figure, by default None
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    # Extract performance summary
    df = extract_performance_summary(results_dir)
    
    if df.empty:
        print("No results found")
        return None
    
    # Find test accuracy columns
    acc_cols = [col for col in df.columns if col.endswith('_test_acc')]
    
    if not acc_cols:
        print("No test accuracy metrics found")
        return None
    
    # Melt the dataframe to get a long format
    id_vars = ['experiment_id', 'reduction_method', 'dim_reduction', 'readout_type', 'n_shots']
    id_vars = [col for col in id_vars if col in df.columns]
    
    melt_df = df.melt(id_vars=id_vars, 
                     value_vars=acc_cols,
                     var_name='model', 
                     value_name='accuracy')
    
    # Extract model name from column
    melt_df['model'] = melt_df['model'].str.replace('_final_test_acc', '')
    
    # Create plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    pivot_df = melt_df.pivot_table(
        index='model',
        columns='reduction_method',
        values='accuracy',
        aggfunc='mean'
    )
    
    # Plot heatmap
    sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="viridis", ax=ax)
    
    ax.set_title('Model Performance Comparison')
    plt.tight_layout()
    
    # Save figure if path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def combine_sweep_results(sweep_dirs: List[str], output_path: str) -> pd.DataFrame:
    """
    Combine results from multiple parameter sweeps.
    
    Parameters
    ----------
    sweep_dirs : List[str]
        List of sweep directories
    output_path : str
        Path to save the combined results
        
    Returns
    -------
    pd.DataFrame
        Combined results DataFrame
    """
    all_results = []
    
    for sweep_dir in sweep_dirs:
        # Look for all_results.csv
        results_path = os.path.join(sweep_dir, "all_results.csv")
        
        if os.path.exists(results_path):
            df = pd.read_csv(results_path)
            
            # Add sweep directory as source
            df['sweep_source'] = os.path.basename(sweep_dir)
            
            all_results.append(df)
        else:
            # Try to extract summary from experiment results
            df = extract_performance_summary(sweep_dir)
            
            if not df.empty:
                df['sweep_source'] = os.path.basename(sweep_dir)
                all_results.append(df)
    
    if not all_results:
        print("No results found in the provided directories")
        return pd.DataFrame()
    
    # Combine all dataframes
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save to CSV
    combined_df.to_csv(output_path, index=False)
    
    return combined_df
