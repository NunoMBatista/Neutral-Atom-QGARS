from telnetlib import SUPDUP
from venv import create
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from typing import Tuple, List, Dict, Any, Optional, Mapping

from src.feature_reduction.autoencoder.autoencoder_architectures import create_default_architecture, create_convolutional_architecture
from src.utils.cli_printing import print_sequential_model

SUPPORTED_AUTOENCODER_TYPES = ['default', 'convolutional']

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
    use_batch_norm : bool, optional
        Whether to use batch normalization, by default True
    dropout : float, optional
        Dropout probability for regularization, by default 0.1
    """

    
    def __init__(self, 
             input_dim: int, 
             encoding_dim: int, 
             use_batch_norm: bool = True,
             dropout: float = 0.1,
             ae_type: str = 'default'):
        super(Autoencoder, self).__init__()

        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.use_batch_norm = use_batch_norm
        self.dropout = dropout
        self.ae_type = ae_type
        
        if self.ae_type == 'default':
            self.encoder, self.decoder = create_default_architecture(
                                            input_dim=input_dim,            
                                            encoding_dim=encoding_dim,
                                            use_batch_norm=use_batch_norm,
                                            dropout=dropout
                                            
                                        )
        else:
            raise ValueError(f"Unknown architecture type: {self.ae_type}. Supported types: {SUPPORTED_AUTOENCODER_TYPES}")

    
    def __str__(self, use_colors: bool = True) -> str:
        return print_sequential_model(model=self.encoder, model_name="Encoder", use_colors=use_colors) + \
               print_sequential_model(model=self.decoder, model_name="Decoder", use_colors=use_colors)
               
    
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


def prepare_autoencoder_data(data: np.ndarray, batch_size: int, verbose: bool = True) -> Tuple[torch.Tensor, torch.utils.data.DataLoader]:
    """
    Prepare data for autoencoder training.
    
    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_features, n_samples)
    batch_size : int
        Batch size for training
    verbose : bool, optional
        Whether to show warning messages, by default True
        
    Returns
    -------
    Tuple[torch.Tensor, torch.utils.data.DataLoader]
        Input tensor and data loader
    """
    # Transpose data to have samples as first dimension (PyTorch convention)
    X = torch.tensor(data.T, dtype=torch.float32)
    
    # Convert batch_size to Python native int to avoid PyTorch DataLoader errors
    batch_size = int(batch_size)
    
    # Adjust batch size if it's larger than dataset size
    adjusted_batch_size = min(batch_size, X.shape[0])
    if adjusted_batch_size != batch_size and verbose:
        tqdm.write(f"Warning: Reducing batch size from {batch_size} to {adjusted_batch_size} to match dataset size")
    
    # Create dataset and dataloader - input and target are the same for autoencoder
    dataset = TensorDataset(X, X)
    dataloader = DataLoader(dataset=dataset, batch_size=adjusted_batch_size, shuffle=True)
    
    return X, dataloader


# def initialize_autoencoder(input_dim: int, 
#                            encoding_dim: int,
#                            use_batch_norm: bool = True, 
#                            dropout: float = 0.1, 
#                            device: str = 'cpu') -> Autoencoder:
#     """
#     Initialize the autoencoder model.
    
#     Parameters
#     ----------
#     input_dim : int
#         Dimension of input features
#     encoding_dim : int
#         Dimension of the encoded representation
#     hidden_dims : Optional[List[int]], optional
#         Dimensions of hidden layers, by default None
#     use_batch_norm : bool, optional
#         Whether to use batch normalization, by default True
#     dropout : float, optional
#         Dropout probability, by default 0.1
#     device : str, optional
#         Device to use ('cpu' or 'cuda'), by default 'cpu'
        
#     Returns
#     -------
#     Autoencoder
#         Initialized autoencoder model
#     """
#     # Initialize model
#     model = Autoencoder(
#         input_dim=input_dim,
#         encoding_dim=encoding_dim,
#         use_batch_norm=use_batch_norm,
#         dropout=dropout
#     )
#     # Move model to specified device
#     return model.to(device)


def setup_training(model: nn.Module, 
                   learning_rate: float, 
                   autoencoder_regularization: float = 1e-5) -> Tuple[nn.Module, optim.Optimizer, optim.lr_scheduler.ReduceLROnPlateau]:
    """
    Set up training components: loss function, optimizer, and scheduler.
    
    Parameters
    ----------
    model : nn.Module
        The autoencoder model
    learning_rate : float
        Initial learning rate
    autoencoder_regularization : float
        Weight decay for regularization, by default 1e-5
        
    Returns
    -------
    Tuple[nn.Module, optim.Optimizer, optim.lr_scheduler.ReduceLROnPlateau]
        Loss criterion, optimizer, and learning rate scheduler
    """
    # Define loss function
    criterion = nn.MSELoss()
    
    # Define optimizer with weight decay for regularization
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=autoencoder_regularization
    )
    
    # Learning rate scheduler that reduces LR when loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',  # Reduce LR when monitored quantity stops decreasing
        patience=5,  # Number of epochs with no improvement after which LR will be reduced
        factor=0.5   # Multiply LR by this factor when reducing
    )
    
    return criterion, optimizer, scheduler


def train_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader, 
                criterion: nn.Module, optimizer: optim.Optimizer, device: str = 'cpu') -> float:
    """
    Train the autoencoder for one epoch.
    
    Parameters
    ----------
    model : nn.Module
        The autoencoder model
    dataloader : torch.utils.data.DataLoader
        DataLoader for training data
    criterion : nn.Module
        Loss function
    optimizer : optim.Optimizer
        Optimizer
    device : str, optional
        Device to use, by default 'cpu'
        
    Returns
    -------
    float
        Average loss for the epoch
    """
    model.train()  # Set model to training mode
    running_loss = 0.0
    
    # Iterate through batches
    for batch_X, _ in dataloader:
        batch_X = batch_X.to(device)
        
        # Zero gradients before forward pass
        optimizer.zero_grad()
        
        # Forward pass through autoencoder
        _, reconstructed = model(batch_X)
        
        # Compute reconstruction loss
        loss = criterion(reconstructed, batch_X)
        
        # Backward pass and optimize
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights
        
        # Track loss
        running_loss += loss.item()
    
    # Calculate average loss for the epoch
    avg_loss = running_loss / len(dataloader)
    return avg_loss


def encode_all_data(model: Autoencoder, X: torch.Tensor, device: str = 'cpu', batch_size: int = 256) -> np.ndarray:
    """
    Encode all data using the trained autoencoder.
    
    Parameters
    ----------
    model : nn.Module
        Trained autoencoder model
    X : torch.Tensor
        Input data tensor
    device : str, optional
        Device to use, by default 'cpu'
    batch_size : int, optional
        Batch size for processing, by default 256
        
    Returns
    -------
    np.ndarray
        Encoded representations
    """
    model.eval()  # Set model to evaluation mode
    encodings = []
    
    # Process data in batches to avoid memory issues
    with torch.no_grad():  # No need to track gradients for inference
        for i in range(0, X.shape[0], batch_size):
            batch_X = X[i:i+batch_size].to(device)
            batch_encoding = model.encode(batch_X).cpu().numpy()
            encodings.append(batch_encoding)
    
    # Concatenate all batches
    encoded_data = np.vstack(encodings)
    
    return encoded_data


def train_autoencoder(data: np.ndarray, encoding_dim: int, 
                     batch_size: int = 64, 
                     epochs: int = 50,
                     learning_rate: float = 0.001,
                     device: str = 'cpu',
                     verbose: bool = True,
                     use_batch_norm: bool = True,
                     dropout: float = 0.1,
                     autoencoder_regularization: float = 1e-5) -> Tuple[Autoencoder, float]:
    """
    Train an autoencoder for dimensionality reduction with improved feature extraction.
    
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
    use_batch_norm : bool, optional
        Whether to use batch normalization, by default True
    dropout : float, optional
        Dropout probability, by default 0.1
    autoencoder_regularization : float, optional
        Regularization parameter for autoencoder, by default 1e-5
        
    Returns
    -------
    Tuple[Autoencoder, float]
        Trained autoencoder and maximum absolute value for scaling
    """
    # Prepare data
    input_dim = data.shape[0]  # Number of features
    X, dataloader = prepare_autoencoder_data(data, batch_size, verbose)
    
    print(f"Input : {data.shape}")
    # Initialize model
    model = Autoencoder(
        input_dim=input_dim,
        encoding_dim=encoding_dim,
        use_batch_norm=use_batch_norm,
        dropout=dropout,
    ).to(device)
    
    # Setup training components
    criterion, optimizer, scheduler = setup_training(
        model=model,
        learning_rate=learning_rate,
        autoencoder_regularization=autoencoder_regularization
    )
    
    # Store current learning rate for tracking changes
    current_lr = learning_rate
    
    # Early stopping parameters
    best_loss = float('inf')
    patience_counter = 0
    patience = 10  # Early stopping after 10 epochs without improvement
    best_model_state: Mapping[str, Any] = {}

    # Create progress bar for training
    progress_bar = tqdm(range(epochs), desc="Training autoencoder") if verbose else range(epochs)
    
    # Training loop
    for epoch in progress_bar:
        # Train for one epoch
        avg_loss = train_epoch(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )
        
        # Update learning rate using scheduler
        prev_lr = current_lr
        scheduler.step(avg_loss)
        
        # Check if learning rate changed
        current_lr = optimizer.param_groups[0]['lr']
        if verbose and current_lr != prev_lr:
            tqdm.write(f"Learning rate changed from {prev_lr:.6f} to {current_lr:.6f}")
        
        # Report progress
        if verbose:
            tqdm.write(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model weights
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    tqdm.write(f"Early stopping triggered after {epoch+1} epochs")
                # Restore best model
                model.load_state_dict(best_model_state)
                break
    
    # Get encoded representations for all data
    encoded_data = encode_all_data(model, X, device)
    
    # Compute max absolute value for scaling
    spectral = max(abs(encoded_data.max()), abs(encoded_data.min()))
    
    return model, spectral


def encode_data(model: Autoencoder, data: np.ndarray, device: str = 'cpu', 
               verbose: bool = True, batch_size: int = 256) -> np.ndarray:
    """
    Encode data using a trained autoencoder model.
    
    Parameters
    ----------
    model : Autoencoder
        Trained autoencoder model
    data : np.ndarray
        Input data with shape (features, samples)
    device : str, optional
        Device to run encoding on ('cpu' or 'cuda'), by default 'cpu'
    verbose : bool, optional
        Whether to show progress bar, by default True
    batch_size : int, optional
        Batch size for processing data, by default 256
        
    Returns
    -------
    np.ndarray
        Encoded data with shape (encoding_dim, samples)
    """
    model.eval()
    n_samples = data.shape[1]
    encoded_data = []
    
    # Process data in batches
    with torch.no_grad():
        batch_iterator = range(0, n_samples, batch_size)
        if verbose:
            batch_iterator = tqdm(batch_iterator, desc="Encoding data")
        
        for i in batch_iterator:
            batch_end = min(i + batch_size, n_samples)
            
            batch = torch.tensor(
                        data=data[:, i:batch_end].T, 
                        dtype=torch.float32
                    ).to(device)
            
            batch_encoded = model.encode(batch).cpu().numpy()
            encoded_data.append(batch_encoded)
    
    # Concatenate all batches and transpose to match expected shape
    return np.vstack(encoded_data).T
