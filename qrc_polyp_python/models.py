import torch
import torch.nn as nn

class LinearClassifier(nn.Module):
    """
    Simple linear classifier with softmax output.
    
    A single-layer neural network that applies a linear transformation
    followed by a softmax activation.
    
    Attributes
    ----------
    linear : nn.Linear
        Linear transformation layer
    softmax : nn.Softmax
        Softmax activation function
    """
    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        Initialize the linear classifier.
        
        Parameters
        ----------
        input_dim : int
            Dimension of input features
        output_dim : int
            Dimension of output (number of classes)
        """
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
    
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
        return self.softmax(self.linear(x))

class NeuralNetwork(nn.Module):
    """
    Multi-layer neural network with two hidden layers.
    
    A feed-forward neural network with two hidden layers and ReLU activations,
    followed by a softmax output layer.
    
    Attributes
    ----------
    layers : nn.Sequential
        Sequential container of network layers
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 100) -> None:
        """
        Initialize the neural network.
        
        Parameters
        ----------
        input_dim : int
            Dimension of input features
        output_dim : int
            Dimension of output (number of classes)
        hidden_dim : int, optional
            Dimension of hidden layers (default is 100)
        """
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=1)
        )
    
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
