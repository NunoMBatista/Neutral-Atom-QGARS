from pathlib import Path 
from typing import Dict, List, Tuple, Any
import numpy as np
import torch.nn as nn

# Define the root directory of the project
ROOT_DIR = Path(__file__).resolve().parent.parent

# Define the path to the datasets directory
DEFAULT_DATA_DIR = ROOT_DIR / 'data' / 'datasets'

# Define the path to the default results directory
DEFAULT_RESULTS_DIR = ROOT_DIR / 'results' / 'default_results'
    
# Define results dictionary typing
# ResultsDict = Dict[str, 
#                    Tuple[
#                         List[float],    # Training losses per epoch
#                         List[float],    # Training accuracies per epoch
#                         List[float],    # Test accuracies per epoch
#                         nn.Module,      # Trained model
#                         np.ndarray,     # Confusion matrix for training set
#                         np.ndarray,     # Confusion matrix for test set
#                         float,          # F1 score for training set
#                         float           # F1 score for test set
#                         ]
#                    ]

TrainingResults = Tuple[
     List[float],
     List[float],
     List[float],
     nn.Module,
     np.ndarray,
     np.ndarray,
     float,
     float
]

ResultsDict = Dict[str, TrainingResults]