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
    use_batch_norm : bool, optional
        Whether to use batch normalization, by default True
    dropout : float, optional
        Dropout probability for regularization, by default 0.1
    """
    def __init__(self, 
                 input_dim: int, 
                 encoding_dim: int, 
                 hidden_dims: Optional[List[int]] = None,
                 use_batch_norm: bool = True,
                 dropout: float = 0.1):
        super(Autoencoder, self).__init__()

        # Default architecture if hidden_dims not provided
        if hidden_dims is None:
            # TODO: MIGHT BE BETTER TO USE A FUNCTION OF THE INPUT DIMENSION
            hidden_dims = [
                max(input_dim // 2, encoding_dim * 8),
                max(input_dim // 4, encoding_dim * 4), 
                max(input_dim // 8, encoding_dim * 2)
            ]

            # Remove layers that are smaller than encoding_dim
            hidden_dims = [dim for dim in hidden_dims if dim > encoding_dim]


        print("""

            **************************
             Creating Encoder Layers
            **************************        
        
            """)


        # Create encoder layers
        encoder_layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            # Add a linear layer
            encoder_layers.append(
                nn.Linear(
                        in_features=prev_dim, 
                        out_features=dim
                    )
                )
            
            # Add batch normalization if specified
            if use_batch_norm:
                encoder_layers.append(
                    nn.BatchNorm1d(
                            num_features=dim
                        )
                    )
            
            # Add LeakyReLU activation function
            encoder_layers.append(
                nn.LeakyReLU(
                        negative_slope=0.2
                    )
                )
            
            # Add dropout if specified
            if dropout > 0:
                encoder_layers.append(
                    nn.Dropout(
                            p=dropout
                        )
                    )
                
            # Update previous dimension
            prev_dim = dim
        
        
        # Final encoding layer
        encoder_layers.append(
            nn.Linear(
                    in_features=prev_dim, 
                    out_features=encoding_dim
                )
            )
        
        # Add batch normalization if specified
        if use_batch_norm:
            encoder_layers.append(
                nn.BatchNorm1d(
                        num_features=encoding_dim
                    )
                )
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        
        print("""

            **************************
             Creating Decoder Layers
            **************************    
        
            """)


        # Create decoder layers (reverse of encoder)
        decoder_layers = []
        prev_dim = encoding_dim
        
        # Hidden layers in reverse order
        for dim in reversed(hidden_dims):
            decoder_layers.append(
                nn.Linear(
                        in_features=prev_dim,
                        out_features=dim
                    )
                )
            
            if use_batch_norm:
                decoder_layers.append(
                    nn.BatchNorm1d(
                            num_features=dim
                        )
                    )
            
            decoder_layers.append(
                nn.LeakyReLU(
                        negative_slope=0.2
                    )
                )
            
            if dropout > 0:
                decoder_layers.append(
                    nn.Dropout(
                            p=dropout
                        )
                    )
            
            prev_dim = dim
        
        # Final reconstruction layer
        decoder_layers.append(
            nn.Linear(
                    in_features=prev_dim,
                    out_features=input_dim
                )
            )
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
    input_dim = data.shape[0]
    X = torch.tensor(
                data=data.T, 
                dtype=torch.float32
            )
    
    # Convert batch_size to Python native int to avoid PyTorch DataLoader errors
    batch_size = int(batch_size)
    
    # Adjust batch size if it's larger than dataset size
    adjusted_batch_size = min(batch_size, X.shape[0])
    if adjusted_batch_size != batch_size and verbose:
        tqdm.write(f"Warning: Reducing batch size from {batch_size} to {adjusted_batch_size} to match dataset size")
    
    # Create dataset and dataloader
    dataset = TensorDataset(X, X)  # Input = target for autoencoder because we want to reconstruct the input
    dataloader = DataLoader(
            dataset=dataset, 
            batch_size=adjusted_batch_size, 
            shuffle=True
        )
    
    # Initialize model
    model = Autoencoder(
                    input_dim=input_dim, 
                    encoding_dim=encoding_dim, 
                    hidden_dims=hidden_dims, 
                    use_batch_norm=use_batch_norm, 
                    dropout=dropout
                )
    model = model.to(device)
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=autoencoder_regularization
        )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Store current learning rate for tracking changes
    current_lr = learning_rate
    
    # Training loop
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    patience = 10  # Early stopping patience
    
    progress_bar = tqdm(range(epochs), desc="Training autoencoder") if verbose else range(epochs)
    for epoch in progress_bar:
        running_loss = 0.0
        
        for batch_X, _ in dataloader:
            batch_X = batch_X.to(device)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass
            _, reconstructed = model(batch_X)
            
            # Compute loss
            loss = criterion(reconstructed, batch_X)
            
            # Backward pass and optimize
            loss.backward() # Compute the gradients
            optimizer.step() # Update weights
            
            running_loss += loss.item()
        
        # Calculate average loss for the epoch
        avg_loss = running_loss / len(dataloader)
        
        # Update learning rate
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
    model.eval()
    encodings = []
    batch_size_eval = 256  # Larger batch size for evaluation
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size_eval):
            batch_X = X[i:i+batch_size_eval].to(device)
            batch_encoding = model.encode(batch_X).cpu().numpy()
            encodings.append(batch_encoding)
    
    encoded_data = np.vstack(encodings)
    
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
