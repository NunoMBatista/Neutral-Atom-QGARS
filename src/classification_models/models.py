import numpy as np 
import torch
import torch.nn as nn
from typing import Any, List, Optional, Union, Tuple
from tqdm import tqdm 
from src.utils.cli_printing import print_sequential_model
from src.globals import TrainingResults

class LinearClassifier(nn.Module):
    """
    Simple linear classifier with softmax output.
    
    Parameters
    ----------
    input_dim : int
        Dimension of input features
    output_dim : int
        Number of output classes
    bias : bool, optional
        Whether to include bias, by default True
    """
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        #self.softmax = nn.Softmax(dim=1)
        # No need for softmax, as it's applies in the CrossEntropyLoss function
    
    def __str__(self, use_colors: bool = True) -> str:
        iterable = nn.Sequential(self.linear)
        return print_sequential_model(model=iterable, model_name="Linear Classifier", use_colors=use_colors)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        
        Returns
        -------
        torch.Tensor
            Output probabilities
        """
        #return self.softmax(self.linear(x))
        return self.linear(x)  # Softmax is applied in the loss function
        
class NeuralNetwork(nn.Module):
    """
    Multi-layer neural network with configurable architecture.
    
    Parameters
    ----------
    input_dim : int
        Dimension of input features
    output_dim : int
        Number of output classes
    hidden_dims : List[int], optional
        Sizes of hidden layers, by default [100, 100]
    activation : Union[str, nn.Module], optional
        Activation function to use, by default "relu"
    dropout : float, optional
        Dropout probability, by default 0.0
    batch_norm : bool, optional
        Whether to use batch normalization, by default False
    """
    def __init__(self, input_dim: int, output_dim: int, 
                 hidden_dims: List[int] = [100, 100],
                 activation: Union[str, nn.Module] = "relu",
                 dropout: float = 0.0,
                 batch_norm: bool = False):
        super(NeuralNetwork, self).__init__()
        
        # Define activation function
        if isinstance(activation, str):
            if activation.lower() == "relu":
                act_fn = nn.ReLU()
            elif activation.lower() == "leaky_relu":
                act_fn = nn.LeakyReLU()
            elif activation.lower() == "tanh":
                act_fn = nn.Tanh()
            elif activation.lower() == "sigmoid":
                act_fn = nn.Sigmoid()
            else:
                raise ValueError(f"Unknown activation function: {activation}")
        else:
            act_fn = activation
        
        # Build model architecture
        layers = []
        prev_dim = input_dim
        
        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(dim))
                
            layers.append(act_fn)
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
                
            prev_dim = dim
        
        # Output layer - no softmax since we're using CrossEntropyLoss
        layers.append(nn.Linear(prev_dim, output_dim))
        # Remove the softmax layer to work properly with CrossEntropyLoss
        # layers.append(nn.Softmax(dim=1))
        
        self.layers = nn.Sequential(*layers)
    
    def __str__(self, use_colors: bool = True) -> str:
        return print_sequential_model(model=self.layers, model_name="Neural Network", use_colors=use_colors)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        
        Returns
        -------
        torch.Tensor
            Output probabilities
        """
        return self.layers(x)

class QRCModel:
    """
    Full QRC model combining quantum embedding and classical training.
    
    Parameters
    ----------
    quantum_layer : Any
        Quantum embedding layer
    classifier : nn.Module, optional
        Classical classifier, by default None
    """
    # def __init__(self, quantum_layer: Any, classifier: Optional[nn.Module] = None):
    #     self.quantum_layer = quantum_layer
    #     self.classifier = classifier if classifier is not None else LinearClassifier
      
    def __init__(self, quantum_layer: Any, classifier: Optional[nn.Module] = None):
        self.quantum_layer = quantum_layer
        # Create an instance if classifier is None (with default parameters)
        self.classifier = classifier  # Just store the provided classifier
        
    def __str__(self, use_colors: bool = True) -> str:
        if isinstance(self.classifier, nn.Module):
            # If it's already a module instance
            if isinstance(self.classifier, nn.Sequential):
                return print_sequential_model(model=self.classifier, model_name="QRC Model", use_colors=use_colors)
            else:
                # Wrap non-Sequential modules in a Sequential container
                return print_sequential_model(model=nn.Sequential(self.classifier), model_name="QRC Model", use_colors=use_colors)
        else:
            # If it's a class (not instantiated), just return a simple string
            return f"QRC Model with the provided classifier (not instantiated)"
        
    def fit(self, x_train: np.ndarray, y_train: np.ndarray, 
            x_test: np.ndarray, y_test: np.ndarray, 
            **kwargs) -> TrainingResults:
        """
        Train the model end-to-end.
        
        Parameters
        ----------
        x_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        x_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
        **kwargs
            Additional arguments for the training function
            
        Returns
        -------
        Tuple[List[float], List[float], List[float], nn.Module]
            Training metrics and trained model
        """
        from src.classification_models.training import train
                
        # Get quantum embeddings
        tqdm.write("Computing quantum embeddings for training data...")
        train_embeddings = self.quantum_layer.apply_layer(x_train)
        
        tqdm.write("Computing quantum embeddings for test data...")
        test_embeddings = self.quantum_layer.apply_layer(x_test)
        
        # Train classical model
        return train(train_embeddings, y_train, test_embeddings, y_test, **kwargs)

