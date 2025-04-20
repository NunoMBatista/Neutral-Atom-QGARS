from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt

def plot_training_results(results_dict: Dict[str, Tuple[List[float], List[float], List[float], Any]]) -> None:
    """
    Plot training results for multiple models.
    
    Creates a figure with two subplots: one for accuracy and one for loss.
    
    Parameters
    ----------
    results_dict : Dict[str, Tuple[List[float], List[float], List[float], Any]]
        Dictionary mapping model names to tuples containing:
        (losses, accs_train, accs_test, model) for each model
    
    Returns
    -------
    None
    """
    plt.figure(figsize=(12, 6))
    
    # Plot accuracies
    plt.subplot(1, 2, 1)
    for name, (_, accs_train, accs_test, _) in results_dict.items():
        plt.plot(accs_train, label=f"{name} train")
        plt.plot(accs_test, label=f"{name} test")
    
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Test Accuracy")
    
    # Plot losses
    plt.subplot(1, 2, 2)
    for name, (losses, _, _, _) in results_dict.items():
        plt.plot(losses, label=name)
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss")
    
    plt.tight_layout()
    plt.show()

def print_results(results_dict: Dict[str, Tuple[List[float], List[float], List[float], Any]]) -> None:
    """
    Print the final test accuracies for each model.
    
    Displays a formatted summary of test accuracies for all models in the results.
    
    Parameters
    ----------
    results_dict : Dict[str, Tuple[List[float], List[float], List[float], Any]]
        Dictionary mapping model names to tuples containing:
        (losses, accs_train, accs_test, model) for each model
    
    Returns
    -------
    None
    """
    print("\nResults Summary:")
    print("-" * 50)
    for name, (_, _, accs_test, _) in results_dict.items():
        print(f"{name} test accuracy = {accs_test[-1]*100:.2f}%")
    print("-" * 50)
