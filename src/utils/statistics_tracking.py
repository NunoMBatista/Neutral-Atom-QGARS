import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional
import json
import datetime
import logging

def ensure_directory_exists(directory: str) -> None:
    """
    Ensure the specified directory exists, creating it if necessary.
    
    Parameters
    ----------
    directory : str
        Directory path to check/create
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def setup_stats_directory(base_dir: str = "run_stats") -> str:
    """
    Set up a directory for saving run statistics.
    
    Parameters
    ----------
    base_dir : str, optional
        Base directory for statistics, by default "run_stats"
        
    Returns
    -------
    str
        Path to the created directory
    """
    # Create a timestamped directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), base_dir, timestamp)
    ensure_directory_exists(stats_dir)
    
    # Set up logging
    logging.basicConfig(
        filename=os.path.join(stats_dir, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    return stats_dir

def save_classifier_loss_plot(results_dict: Dict[str, Tuple[List[float], List[float], List[float], Any]], 
                             output_dir: str) -> None:
    """
    Plot and save training losses for each classifier.
    
    Parameters
    ----------
    results_dict : Dict[str, Tuple[List[float], List[float], List[float], Any]]
        Dictionary mapping model names to (losses, accs_train, accs_test, model) tuples
    output_dir : str
        Directory to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    for name, (losses, _, _, _, _, _, _, _) in results_dict.items():
        plt.plot(losses, label=f"{name} loss")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, "classifier_losses.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved classifier loss plot to {plot_path}")

def save_classifier_accuracy_plot(results_dict: Dict[str, Tuple[List[float], List[float], List[float], Any]], 
                                output_dir: str) -> None:
    """
    Plot and save training and test accuracies for each classifier.
    
    Parameters
    ----------
    results_dict : Dict[str, Tuple[List[float], List[float], List[float], Any]]
        Dictionary mapping model names to (losses, accs_train, accs_test, model) tuples
    output_dir : str
        Directory to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    for name, (_, accs_train, accs_test, _, _, _, _, _) in results_dict.items():
        plt.plot(accs_train, label=f"{name} train")
        plt.plot(accs_test, label=f"{name} test")
    
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Test Accuracies")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, "classifier_accuracies.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved classifier accuracy plot to {plot_path}")

def save_guided_autoencoder_losses(losses: Dict[str, List[float]], output_dir: str) -> None:
    """
    Plot and save guided autoencoder losses.
    
    Parameters
    ----------
    losses : Dict[str, List[float]]
        Dictionary with keys 'total_loss', 'recon_loss', 'class_loss', 'surrogate_loss' mapping to loss histories
    output_dir : str
        Directory to save the plot
    """
    # Plot main losses (reconstruction, classification, total)
    plt.figure(figsize=(10, 6))
    
    # Use the correct key names from the loss_history dictionary
    plt.plot(losses['total_loss'], label='Total Loss', color='blue', linewidth=2)
    plt.plot(losses['recon_loss'], label='Reconstruction Loss', color='green', linewidth=1.5, linestyle='--')
    plt.plot(losses['class_loss'], label='Classification Loss', color='red', linewidth=1.5, linestyle=':')
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Guided Autoencoder Training Losses")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, "guided_autoencoder_losses.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved guided autoencoder loss plot to {plot_path}")
    
    # Plot surrogate loss separately if available
    if 'surrogate_loss' in losses and any(x is not None for x in losses['surrogate_loss']):
        plt.figure(figsize=(10, 6))
        
        # Extract valid surrogate loss values and their corresponding epochs
        surrogate_epochs = []
        surrogate_values = []
        for i, value in enumerate(losses['surrogate_loss']):
            if value is not None:
                surrogate_epochs.append(i)
                surrogate_values.append(value)
        
        plt.plot(surrogate_epochs, surrogate_values, marker='o', color='purple', linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Quantum Surrogate Model Loss")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the plot
        surrogate_plot_path = os.path.join(output_dir, "surrogate_loss.png")
        plt.savefig(surrogate_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save surrogate loss to a separate JSON file for easier access
        surrogate_json_path = os.path.join(output_dir, "surrogate_loss.json")
        with open(surrogate_json_path, 'w') as f:
            json.dump({
                "epochs": surrogate_epochs,
                "loss_values": surrogate_values
            }, f, indent=4)
        
        logging.info(f"Saved surrogate model loss plot to {surrogate_plot_path}")
        logging.info(f"Saved surrogate model loss data to {surrogate_json_path}")

def save_loss_logs(results_dict: Dict[str, Tuple[List[float], List[float], List[float], Any]], 
                  output_dir: str) -> None:
    """
    Save loss and accuracy logs for each classifier.
    
    Parameters
    ----------
    results_dict : Dict[str, Tuple[List[float], List[float], List[float], Any]]
        Dictionary mapping model names to (losses, accs_train, accs_test, model) tuples
    output_dir : str
        Directory to save the logs
    """
    log_data = {}
    
    for name, (losses, accs_train, accs_test, _, _, _, _, _) in results_dict.items():
        log_data[name] = {
            'losses': losses,
            'train_accuracy': accs_train,
            'test_accuracy': accs_test
        }
    
    # Save as JSON
    log_path = os.path.join(output_dir, "classifier_metrics.json")
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=4)
    
    logging.info(f"Saved classifier metrics to {log_path}")

def save_guided_autoencoder_logs(losses: Dict[str, List[float]], output_dir: str) -> None:
    """
    Save guided autoencoder loss logs.
    
    Parameters
    ----------
    losses : Dict[str, List[float]]
        Dictionary with loss histories
    output_dir : str
        Directory to save the logs
    """
    # Save as JSON
    log_path = os.path.join(output_dir, "guided_autoencoder_metrics.json")
    with open(log_path, 'w') as f:
        json.dump(losses, f, indent=4)
    
    logging.info(f"Saved guided autoencoder metrics to {log_path}")

# Rename the internal function to be exported
def extract_metrics(results_dict: Dict[str, Tuple[List[float], List[float], List[float], Any]]) -> Dict[str, Any]:
    """
    Extract key metrics from results dictionary.
    
    Parameters
    ----------
    results_dict : Dict[str, Tuple[List[float], List[float], List[float], Any]]
        Dictionary mapping model names to (losses, accs_train, accs_test, model) tuples
    
    Returns
    -------
    Dict[str, Any]
        Dictionary of extracted metrics
    """
    metrics = {
        "status": "success",
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Extract final metrics from each model
    for model_name, (losses, accs_train, accs_test, _, confusion_matrix_train, confusion_matrix_test, f1_train, f1_test) in results_dict.items():
        metrics[f"{model_name}_final_train_acc"] = float(accs_train[-1])
        metrics[f"{model_name}_final_test_acc"] = float(accs_test[-1])
        metrics[f"{model_name}_final_loss"] = float(losses[-1])
        metrics[f"{model_name}_confusion_matrix_train"] = confusion_matrix_train
        metrics[f"{model_name}_confusion_matrix_test"] = confusion_matrix_test
        metrics[f"{model_name}_f1_train"] = float(f1_train)
        metrics[f"{model_name}_f1_test"] = float(f1_test)
        
        # Add mean metrics for the last 10% of training (stability)
        if len(accs_test) > 5:
            stability_window = max(1, int(len(accs_test) * 0.1))
            metrics[f"{model_name}_stability"] = float(np.std(accs_test[-stability_window:]))
        
        # Additional metrics
        metrics[f"{model_name}_max_test_acc"] = float(max(accs_test))
        metrics[f"{model_name}_convergence_epoch"] = int(np.argmax(accs_test))
    
    return metrics

def save_metrics_json(metrics: Dict[str, Any], output_dir: str) -> None:
    """
    Save metrics to a JSON file.
    
    Parameters
    ----------
    metrics : Dict[str, Any]
        Dictionary of metrics to save
    output_dir : str
        Directory to save the metrics
    """
    metrics_path = os.path.join(output_dir, "metrics.json")
    
    # Replace confusion matrices with their string representations
    for key in metrics.keys():
        if 'confusion_matrix' in key:
            metrics[key] = str(metrics[key])
    
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logging.info(f"Saved metrics to {metrics_path}")

def save_config_file(args: Any, output_dir: str) -> None:
    """
    Save the configuration as a JSON file.
    
    Parameters
    ----------
    args : Any
        Command line arguments or configuration object
    output_dir : str
        Directory to save the configuration
    """
    # Convert args to dictionary if it's not already
    if hasattr(args, '__dict__'):
        config = vars(args)
    else:
        config = args
    
    # Convert numpy arrays and other non-serializable types
    config_serializable = {}
    for k, v in config.items():
        if isinstance(v, np.ndarray):
            config_serializable[k] = v.tolist()
        elif isinstance(v, (int, float, str, bool, list, dict, tuple)) or v is None:
            config_serializable[k] = v
        else:
            # Convert other types to string representation
            config_serializable[k] = str(v)
    
    # Save as JSON
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config_serializable, f, indent=4)
    
    logging.info(f"Saved configuration to {config_path}")

def save_all_statistics(results_dict: Dict[str, Tuple[List[float], List[float], List[float], Any]], 
                       guided_losses: Optional[Dict[str, List[float]]] = None,
                       output_dir: Optional[str] = None,
                       args: Optional[Any] = None) -> str:
    """
    Save all statistics including plots and logs.
    
    Parameters
    ----------
    results_dict : Dict[str, Tuple[List[float], List[float], List[float], Any]]
        Dictionary mapping model names to (losses, accs_train, accs_test, model) tuples
    guided_losses : Optional[Dict[str, List[float]]], optional
        Dictionary with guided autoencoder loss histories, by default None
    output_dir : Optional[str], optional
        Directory to save statistics, by default None (creates one)
    args : Optional[Any], optional
        Command line arguments or configuration object, by default None
        
    Returns
    -------
    str
        Path to the statistics directory
    """
    # Create stats directory if not provided
    if output_dir is None:
        output_dir = setup_stats_directory()
    else:
        ensure_directory_exists(output_dir)
    
    # Save configuration if provided
    if args is not None:
        save_config_file(args, output_dir)
    
    # Save classifier metrics
    save_classifier_loss_plot(results_dict, output_dir)
    save_classifier_accuracy_plot(results_dict, output_dir)
    save_loss_logs(results_dict, output_dir)
    
    # Save guided autoencoder metrics if provided
    if guided_losses is not None:
        save_guided_autoencoder_losses(guided_losses, output_dir)
        save_guided_autoencoder_logs(guided_losses, output_dir)
    
    # Extract and save metrics in JSON format (using the common function)
    metrics = extract_metrics(results_dict)
    
    # Add guided autoencoder metrics if available
    if guided_losses is not None:
        metrics["guided_autoencoder_final_total_loss"] = float(guided_losses["total_loss"][-1])
        metrics["guided_autoencoder_final_recon_loss"] = float(guided_losses["recon_loss"][-1])
        metrics["guided_autoencoder_final_class_loss"] = float(guided_losses["class_loss"][-1])
        
        # Add surrogate metrics if available
        if 'surrogate_loss' in guided_losses and any(x is not None for x in guided_losses['surrogate_loss']):
            # Get the last valid surrogate loss
            valid_surrogate_losses = [x for x in guided_losses['surrogate_loss'] if x is not None]
            if valid_surrogate_losses:
                metrics["surrogate_final_loss"] = float(valid_surrogate_losses[-1])
                metrics["surrogate_min_loss"] = float(min(valid_surrogate_losses))
    
    save_metrics_json(metrics, output_dir)
    
    return output_dir
