import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Any

def plot_training_results(results_dict: Dict[str, Tuple[List[float], List[float], List[float], Any]]) -> None:
    """
    Plot training results for multiple models.
    
    Parameters
    ----------
    results_dict : Dict[str, Tuple[List[float], List[float], List[float], Any]]
        Dictionary mapping model names to (losses, accs_train, accs_test, model) tuples
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
    
    Parameters
    ----------
    results_dict : Dict[str, Tuple[List[float], List[float], List[float], Any]]
        Dictionary mapping model names to (losses, accs_train, accs_test, model) tuples
    """
    print("\nResults Summary:")
    print("-" * 50)
    for name, (_, _, accs_test, _) in results_dict.items():
        print(f"{name} test accuracy = {accs_test[-1]*100:.2f}%")
    print("-" * 50)