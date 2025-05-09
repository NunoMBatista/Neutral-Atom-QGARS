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
                     weight_decay: float = 1e-5) -> Tuple[Autoencoder, float]:
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
    weight_decay : float, optional
        Weight decay for regularization, by default 1e-5
        
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
        print(f"Warning: Reducing batch size from {batch_size} to {adjusted_batch_size} to match dataset size")
    
    # Create dataset and dataloader
    dataset = TensorDataset(X, X)  # Input = target for autoencoder
    dataloader = DataLoader(
            dataset=dataset, 
            batch_size=adjusted_batch_size, 
            shuffle=True
        )
    
    # Initialize model
    model = Autoencoder(input_dim, encoding_dim, hidden_dims, use_batch_norm, dropout)
    model = model.to(device)
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler (remove verbose parameter)
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
        #batch_iterator = tqdm(range(0, n_samples, batch_size), desc="Encoding data") if verbose else range(0, n_samples, batch_size)
        
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



class GuidedAutoencoder:
    """
    Quantum guided autoencoder that jointly optimizes reconstruction and quantum classification.
    
    Parameters
    ----------
    input_dim : int
        Dimension of input features (flattened image size)
    encoding_dim : int
        Dimension of the encoded representation
    output_dim : int
        Number of output classes
    quantum_dim : int
        Dimension of quantum embeddings
    hidden_dims : Optional[List[int]], optional
        Dimensions of hidden layers for autoencoder, by default None
    alpha : float, optional
        Weight for reconstruction loss, by default 0.7
    beta : float, optional
        Weight for classification loss, by default 0.3
    use_batch_norm : bool, optional
        Whether to use batch normalization, by default True
    dropout : float, optional
        Dropout probability, by default 0.1
    """
    def __init__(self, 
                 input_dim: int,
                 encoding_dim: int, 
                 output_dim: int,
                 quantum_dim: int = None,
                 hidden_dims: Optional[List[int]] = None,
                 alpha: float = 0.7, 
                 beta: float = 0.3,
                 use_batch_norm: bool = True,
                 dropout: float = 0.1):
        self.autoencoder = Autoencoder(
                                input_dim=input_dim, 
                                encoding_dim=encoding_dim, 
                                hidden_dims=hidden_dims, 
                                use_batch_norm=use_batch_norm, 
                                dropout=dropout
                                )
        self.alpha = alpha
        self.beta = beta
        self.encoding_dim = encoding_dim
        self.output_dim = output_dim
        self.quantum_dim = quantum_dim
        
        # Classifier will be initialized when quantum_dim is known
        self.classifier = None
        
        # Embedding cache
        self.quantum_embeddings_cache = {}
    
    def initialize_classifier(self, quantum_dim: int):
        """Initialize classifier with the correct input dimension"""
        self.quantum_dim = quantum_dim
        self.classifier = nn.Sequential(
            nn.Linear(
                    in_features=quantum_dim, 
                    out_features=self.output_dim
                ),
            nn.Softmax(
                    dim=1
                )
        )
        return self.classifier
        
    def to(self, device):
        """Move the model to specified device"""
        self.autoencoder = self.autoencoder.to(device)
        if self.classifier is not None:
            self.classifier = self.classifier.to(device)
        return self
    
    def get_encoder(self):
        """Get the encoder part of the autoencoder"""
        return self.autoencoder.encoder
    
    def clear_cache(self):
        """Clear the quantum embeddings cache"""
        self.quantum_embeddings_cache = {}


def train_guided_autoencoder(
    data: np.ndarray, 
    labels: np.ndarray,
    quantum_layer,
    encoding_dim: int,
    hidden_dims: Optional[List[int]] = None,
    alpha: float = 0.7,
    beta: float = 0.3,
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = 'cpu',
    quantum_update_frequency: int = 5,
    n_shots: int = 1000,
    verbose: bool = True,
    use_batch_norm: bool = True,
    dropout: float = 0.1,
    weight_decay: float = 1e-5
) -> Tuple[GuidedAutoencoder, float]:
    """
    Train a guided autoencoder jointly with quantum embeddings.
    
    Parameters
    ----------
    data : np.ndarray
        Input data (flattened images) with shape (n_features, n_samples)
    labels : np.ndarray
        Class labels with shape (n_samples,)
    quantum_layer : DetuningLayer
        Quantum reservoir layer to generate embeddings
    encoding_dim : int
        Dimension of the encoded representation
    hidden_dims : Optional[List[int]], optional
        Dimensions of hidden layers, by default None
    alpha : float, optional
        Weight for reconstruction loss, by default 0.7
    beta : float, optional
        Weight for classification loss, by default 0.3
    batch_size : int, optional
        Batch size for training, by default 32
    epochs : int, optional
        Number of training epochs, by default 50
    learning_rate : float, optional
        Learning rate, by default 0.001
    device : str, optional
        Device to use ('cpu' or 'cuda'), by default 'cpu'
    quantum_update_frequency : int, optional
        Update quantum embeddings every N epochs, by default 5
    n_shots : int, optional
        Number of shots for quantum simulation, by default 1000
    verbose : bool, optional
        Whether to show progress bars, by default True
    use_batch_norm : bool, optional
        Whether to use batch normalization, by default True
    dropout : float, optional
        Dropout probability, by default 0.1
    weight_decay : float, optional
        Weight decay for regularization, by default 1e-5
        
    Returns
    -------
    Tuple[GuidedAutoencoder, float]
        Trained guided autoencoder and maximum absolute value for scaling
    """
    
    # Prepare data
    input_dim = data.shape[0]
    n_samples = data.shape[1]
    output_dim = len(np.unique(labels))
    
    # Convert to PyTorch tensors
    X = torch.tensor(
            data.T, dtype=torch.float32)
    
    # Convert batch_size to Python native int to avoid PyTorch DataLoader errors
    batch_size = int(batch_size)

    # Adjust batch size if it's larger than dataset size
    adjusted_batch_size = min(batch_size, n_samples)
    if adjusted_batch_size != batch_size and verbose:
        print(f"Warning: Reducing batch size from {batch_size} to {adjusted_batch_size} to match dataset size")


    y_one_hot = np.zeros((n_samples, output_dim))
    y_one_hot[np.arange(n_samples), labels] = 1
    y = torch.tensor(y_one_hot, dtype=torch.float32)
    
    
    # Create dataset and dataloader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=adjusted_batch_size, shuffle=True)
    
    # Initialize model
    model = GuidedAutoencoder(
        input_dim, encoding_dim, output_dim, 
                             hidden_dims=hidden_dims, alpha=alpha, beta=beta,
                             use_batch_norm=use_batch_norm, dropout=dropout)
    model = model.to(device)
    
    # Store indices to embeddings mapping (memory efficient)
    sample_indices = torch.arange(n_samples)
    
    # Get a small batch of data to determine quantum embedding dimension
    initial_batch_size = min(5, n_samples)
    model.autoencoder.eval()
    with torch.no_grad():
        initial_encoding = model.autoencoder.encode(X[:initial_batch_size].to(device)).cpu().numpy()
        spectral = max(abs(initial_encoding.max()), abs(initial_encoding.min()))
        initial_encoding_scaled = initial_encoding.T / spectral * 6.0
        
        if verbose:
            print(f"Getting initial quantum embeddings to determine dimension...")
        
        # Get quantum embeddings to determine dimension
        initial_quantum_embs = quantum_layer.apply_layer(
            initial_encoding_scaled, n_shots=n_shots, show_progress=True
        )
        quantum_dim = initial_quantum_embs.shape[0]
        
        if verbose:
            print(f"Quantum embedding dimension detected: {quantum_dim}")
        
        # Initialize classifier with correct dimension
        model.initialize_classifier(quantum_dim)
        model.classifier = model.classifier.to(device)
    
    # Define loss functions
    recon_criterion = nn.MSELoss()
    class_criterion = nn.CrossEntropyLoss()
    
    # Add optimizer for both autoencoder and classifier
    optimizer = optim.Adam(list(model.autoencoder.parameters()) + list(model.classifier.parameters()), 
                          lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler (remove verbose parameter)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Store current learning rate for tracking changes
    current_lr = learning_rate
    
    # Training loop
    model.autoencoder.train()
    
    # For early stopping
    best_loss = float('inf')
    patience_counter = 0
    patience = 10
    best_model_state = None
    
    progress_bar = tqdm(range(epochs), desc="Training guided autoencoder") if verbose else range(epochs)
    for epoch in progress_bar:
        running_recon_loss = 0.0
        running_class_loss = 0.0
        running_total_loss = 0.0
        
        # Update quantum embeddings periodically to save computation
        if epoch % quantum_update_frequency == 0:
            if verbose:
                print(f"Epoch {epoch+1}: Updating quantum embeddings...")
            model.clear_cache()
            
            # Get encoder's current state
            model.autoencoder.eval()
            with torch.no_grad():
                batch_size_qe = 50  # Process in smaller batches for memory efficiency
                for i in range(0, n_samples, batch_size_qe):
                    end_idx = min(i + batch_size_qe, n_samples)
                    batch_indices = sample_indices[i:end_idx]
                    batch_X = X[batch_indices].to(device)
                    
                    # Encode with current autoencoder
                    batch_encoding = model.autoencoder.encode(batch_X).cpu().numpy()
                    
                    # Scale for quantum processing
                    cur_spectral = max(abs(batch_encoding.max()), abs(batch_encoding.min()))
                    if cur_spectral > 0:  # Avoid division by zero
                        batch_encoding_scaled = batch_encoding.T / cur_spectral * 6.0
                    else:
                        batch_encoding_scaled = batch_encoding.T
                    
                    # Get quantum embeddings
                    quantum_embs = quantum_layer.apply_layer(
                        batch_encoding_scaled, n_shots=n_shots, show_progress=True
                    )
                    
                    # Cache quantum embeddings for each sample index
                    for j, idx in enumerate(batch_indices.numpy()):
                        model.quantum_embeddings_cache[idx] = quantum_embs[:, j]
            
            model.autoencoder.train()
            
        # Training batches
        for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_indices = sample_indices[batch_idx * batch_size:batch_idx * batch_size + len(batch_X)]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass through autoencoder
            batch_encoding, reconstructed = model.autoencoder(batch_X)
            
            # Compute reconstruction loss
            recon_loss = recon_criterion(reconstructed, batch_X)
            
            # Get quantum embeddings from cache for these samples
            quantum_embs_batch = []
            for idx in batch_indices.numpy():
                if idx in model.quantum_embeddings_cache:
                    quantum_embs_batch.append(model.quantum_embeddings_cache[idx])
                else:
                    # If not in cache (shouldn't happen with proper update frequency)
                    # Create a zero placeholder with correct dimension
                    quantum_embs_batch.append(np.zeros(model.quantum_dim))
            
            # Create a classifier for the current batch's quantum embeddings
            quantum_embs_tensor = torch.tensor(np.array(quantum_embs_batch), dtype=torch.float32).to(device)
            batch_output = model.classifier(quantum_embs_tensor)
            
            # Compute classification loss
            class_loss = class_criterion(batch_output, torch.argmax(batch_y, dim=1))
            
            # Compute total loss with weighting
            total_loss = model.alpha * recon_loss + model.beta * class_loss
            
            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()
            
            # Track losses
            running_recon_loss += recon_loss.item()
            running_class_loss += class_loss.item()
            running_total_loss += total_loss.item()
        
        # Calculate average losses
        avg_recon_loss = running_recon_loss / len(dataloader)
        avg_class_loss = running_class_loss / len(dataloader)
        avg_total_loss = running_total_loss / len(dataloader)
        
        # Update learning rate scheduler
        prev_lr = current_lr
        scheduler.step(avg_total_loss)
        # Check if learning rate changed
        current_lr = optimizer.param_groups[0]['lr']
        if verbose and current_lr != prev_lr:
            tqdm.write(f"Learning rate changed from {prev_lr:.6f} to {current_lr:.6f}")
        
        # Report progress
        if verbose:
            tqdm.write(f"Epoch {epoch+1}/{epochs}, "
                      f"Recon Loss: {avg_recon_loss:.6f}, "
                      f"Class Loss: {avg_class_loss:.6f}, "
                      f"Total Loss: {avg_total_loss:.6f}")
        
        # Early stopping check
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            patience_counter = 0
            # Save best model state
            best_model_state = {
                'autoencoder': model.autoencoder.state_dict().copy(),
                'classifier': model.classifier.state_dict().copy()
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    tqdm.write(f"Early stopping triggered after {epoch+1} epochs")
                # Restore best model state
                model.autoencoder.load_state_dict(best_model_state['autoencoder'])
                model.classifier.load_state_dict(best_model_state['classifier'])
                break
    
    # Get final encoded representations for all data
    model.autoencoder.eval()
    encodings = []
    batch_size_eval = 256  # Larger batch size for evaluation
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size_eval):
            batch_X = X[i:i+batch_size_eval].to(device)
            batch_encoding = model.autoencoder.encode(batch_X).cpu().numpy()
            encodings.append(batch_encoding)
    
    encoded_data = np.vstack(encodings)
    
    # Compute max absolute value for scaling
    spectral = max(abs(encoded_data.max()), abs(encoded_data.min()))
    if spectral == 0:  # Avoid division by zero
        spectral = 1.0
    
    return model, spectral


def encode_data_guided(model: GuidedAutoencoder, data: np.ndarray, device: str = 'cpu', 
                      verbose: bool = True, batch_size: int = 256) -> np.ndarray:
    """
    Encode data using a trained guided autoencoder model.
    
    Parameters
    ----------
    model : GuidedAutoencoder
        Trained guided autoencoder model
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
    model.autoencoder.eval()
    n_samples = data.shape[1]
    encoded_data = []
    
    # Process data in batches
    with torch.no_grad():
        batch_iterator = tqdm(range(0, n_samples, batch_size), desc="Encoding data") if verbose else range(0, n_samples, batch_size)
        for i in batch_iterator:
            batch_end = min(i + batch_size, n_samples)
            batch = torch.tensor(data[:, i:batch_end].T, dtype=torch.float32).to(device)
            batch_encoded = model.autoencoder.encode(batch).cpu().numpy()
            encoded_data.append(batch_encoded)
    
    # Concatenate all batches and transpose to match expected shape
    return np.vstack(encoded_data).T

