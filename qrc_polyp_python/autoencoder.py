import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from typing import Tuple, List, Dict, Any, Optional

class Autoencoder(nn.Module):
    """
    Autoencoder for dimensionality reduction.
    
    Parameters
    ----------
    input_dim : int
        Dimension of input features (flattened image size)
    encoding_dim : int
        Dimension of the encoded representation
    hidden_dims : List[int], optional
        Dimensions of hidden layers, by default None
    """
    def __init__(self, input_dim: int, encoding_dim: int, hidden_dims: Optional[List[int]] = None):
        super(Autoencoder, self).__init__()
        
        # Default architecture if hidden_dims not provided
        if hidden_dims is None:
            # Create a symmetric architecture
            hidden_dims = [input_dim // 2, input_dim // 4]
            if encoding_dim >= hidden_dims[-1]:
                hidden_dims = [input_dim // 2]
        
        # Create encoder layers
        encoder_layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = dim
        
        # Final encoding layer
        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Create decoder layers (reverse of encoder)
        decoder_layers = []
        prev_dim = encoding_dim
        
        # Hidden layers in reverse order
        for dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = dim
        
        # Final reconstruction layer
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())  # Sigmoid for pixel values in [0,1]
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            encoding, reconstruction
        """
        encoding = self.encoder(x)
        reconstruction = self.decoder(encoding)
        return encoding, reconstruction
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input data.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
            
        Returns
        -------
        torch.Tensor
            Encoded representation
        """
        return self.encoder(x)


def train_autoencoder(data: np.ndarray, encoding_dim: int, 
                     hidden_dims: Optional[List[int]] = None,
                     batch_size: int = 64, 
                     epochs: int = 50,
                     learning_rate: float = 0.001,
                     device: str = 'cpu',
                     verbose: bool = True) -> Tuple[Autoencoder, float]:
    """
    Train an autoencoder for dimensionality reduction.
    
    Parameters
    ----------
    data : np.ndarray
        Input data (flattened images) with shape (n_features, n_samples)
    encoding_dim : int
        Dimension of the encoded representation
    hidden_dims : Optional[List[int]], optional
        Dimensions of hidden layers, by default None
    batch_size : int, optional
        Batch size for training, by default 64
    epochs : int, optional
        Number of training epochs, by default 50
    learning_rate : float, optional
        Learning rate, by default 0.001
    device : str, optional
        Device to use ('cpu' or 'cuda'), by default 'cpu'
    verbose : bool, optional
        Whether to show progress bars, by default True
        
    Returns
    -------
    Tuple[Autoencoder, float]
        Trained autoencoder and maximum absolute value for scaling
    """
    # Prepare data
    input_dim = data.shape[0]
    X = torch.tensor(data.T, dtype=torch.float32)
    
    # Create dataset and dataloader
    dataset = TensorDataset(X, X)  # Input = target for autoencoder
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = Autoencoder(input_dim, encoding_dim, hidden_dims)
    model = model.to(device)
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    
    progress_bar = tqdm(range(epochs), desc="Training autoencoder") if verbose else range(epochs)
    for epoch in progress_bar:
        running_loss = 0.0
        
        for batch_X, _ in dataloader:
            batch_X = batch_X.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            _, reconstructed = model(batch_X)
            
            # Compute loss
            loss = criterion(reconstructed, batch_X)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Report progress
        avg_loss = running_loss / len(dataloader)
        if verbose:
            tqdm.write(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    # Get encoded representations for all data
    model.eval()
    with torch.no_grad():
        encoded_data = model.encode(X.to(device)).cpu().numpy()
    
    # Compute max absolute value for scaling
    spectral = max(abs(encoded_data.max()), abs(encoded_data.min()))
    
    return model, spectral


def encode_data(model: Autoencoder, data: np.ndarray, device: str = 'cpu',
                verbose: bool = True) -> np.ndarray:
    """
    Encode data using a trained autoencoder.
    
    Parameters
    ----------
    model : Autoencoder
        Trained autoencoder model
    data : np.ndarray
        Input data with shape (n_features, n_samples)
    device : str, optional
        Device to use ('cpu' or 'cuda'), by default 'cpu'
    verbose : bool, optional
        Whether to show progress bar, by default True
        
    Returns
    -------
    np.ndarray
        Encoded data with shape (encoding_dim, n_samples)
    """
    # Convert to tensor
    X = torch.tensor(data.T, dtype=torch.float32)
    
    # Encode in batches to avoid memory issues
    batch_size = 128
    n_samples = X.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    model.eval()
    encodings = []
    
    progress_bar = tqdm(range(n_batches), desc="Encoding data") if verbose else range(n_batches)
    
    with torch.no_grad():
        for i in progress_bar:
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch_X = X[start_idx:end_idx].to(device)
            
            # Encode batch
            batch_encoding = model.encode(batch_X).cpu().numpy()
            encodings.append(batch_encoding)
    
    # Concatenate batches
    encoded_data = np.vstack(encodings)
    
    # Return in the format (encoding_dim, n_samples)
    return encoded_data.T
