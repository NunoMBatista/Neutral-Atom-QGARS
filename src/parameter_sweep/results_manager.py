import os
import pickle
from typing import Dict, Tuple

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
