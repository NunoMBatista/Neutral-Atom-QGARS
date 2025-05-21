import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from typing import Tuple, List, Dict, Any, Optional

from autoencoder import Autoencoder
from qrc_layer import DetuningLayer
from quantum_surrogate import QuantumSurrogate, create_and_train_surrogate, train_surrogate
from models import LinearClassifier 

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
    guided_lambda : float, optional
        Weight for classification loss (0-1), by default 0.3
        Loss = (1-lambda)*reconstruction_loss + lambda*classification_loss
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
                 guided_lambda: float = 0.3,
                 use_batch_norm: bool = True,
                 dropout: float = 0.1):
        self.autoencoder = Autoencoder(
                                input_dim=input_dim, 
                                encoding_dim=encoding_dim, 
                                hidden_dims=hidden_dims, 
                                use_batch_norm=use_batch_norm, 
                                dropout=dropout
                                )
        self.guided_lambda = guided_lambda
        self.encoding_dim = encoding_dim
        self.output_dim = output_dim
        self.quantum_dim = quantum_dim
        
        # Classifier will be initialized when quantum_dim is known
        self.classifier = None
        
        # Quantum surrogate model (can only be initialized after quantum_dim is known)
        self.surrogate = None
        
        # Embedding cache
        self.quantum_embeddings_cache = {}
    
    def initialize_classifier(self, quantum_dim: int):
        """Initialize classifier with the correct input dimension"""
        self.quantum_dim = quantum_dim
        
        # Use LinearClassifier from models.py
        self.classifier = LinearClassifier(
            input_dim=quantum_dim,
            output_dim=self.output_dim,
        )
        return self.classifier
    
    def initialize_surrogate(self, surrogate_model: QuantumSurrogate):
        """Initialize surrogate model"""
        self.surrogate = surrogate_model
        return self.surrogate
        
    def to(self, device):
        """Move the model to specified device"""
        self.autoencoder = self.autoencoder.to(device)
        if self.classifier is not None:
            self.classifier = self.classifier.to(device)
        if self.surrogate is not None:
            self.surrogate = self.surrogate.to(device)
        return self
    
    def get_encoder(self):
        """Get the encoder part of the autoencoder"""
        return self.autoencoder.encoder
    
    def clear_cache(self):
        """Clear the quantum embeddings cache"""
        self.quantum_embeddings_cache = {}


def prepare_guided_autoencoder_data(data: np.ndarray, labels: np.ndarray, 
                                   batch_size: int, device: str, 
                                   verbose: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.utils.data.DataLoader]:
    """
    Prepare data for guided autoencoder training.
    
    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_features, n_samples)
    labels : np.ndarray
        Label data
    batch_size : int
        Batch size for training
    device : str
        Device to use ('cpu' or 'cuda')
    verbose : bool, optional
        Whether to show warning messages, by default True
        
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.utils.data.DataLoader]
        Input tensor, label tensor, and data loader
    """
    # Get number of samples and output classes
    n_samples = data.shape[1]
    output_dim = len(np.unique(labels))
    
    # Convert to PyTorch tensors
    X = torch.tensor(data.T, dtype=torch.float32).to(device)
    
    # Convert batch_size to Python native int to avoid PyTorch DataLoader errors
    batch_size = int(batch_size)

    # Adjust batch size if it's larger than dataset size
    adjusted_batch_size = min(batch_size, n_samples)
    if adjusted_batch_size != batch_size and verbose:
        print(f"Warning: Reducing batch size from {batch_size} to {adjusted_batch_size} to match dataset size")

    # One-hot encode labels
    y_one_hot = np.zeros((n_samples, output_dim))
    y_one_hot[np.arange(n_samples), labels] = 1
    y = torch.tensor(y_one_hot, dtype=torch.float32).to(device)
    
    # Create dataset and dataloader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(
            dataset=dataset, 
            batch_size=adjusted_batch_size, 
            shuffle=True
        )
        
    return X, y, dataloader

def initialize_guided_autoencoder(input_dim: int, encoding_dim: int, output_dim: int,
                                 hidden_dims: Optional[List[int]] = None,
                                 guided_lambda: float = 0.3,
                                 use_batch_norm: bool = True,
                                 dropout: float = 0.1,
                                 device: str = 'cpu') -> GuidedAutoencoder:
    """
    Initialize the guided autoencoder model.
    
    Parameters
    ----------
    input_dim : int
        Dimension of input features
    encoding_dim : int
        Dimension of the encoded representation
    output_dim : int
        Number of output classes
    hidden_dims : Optional[List[int]], optional
        Dimensions of hidden layers, by default None
    guided_lambda : float, optional
        Weight for classification loss, by default 0.3
    use_batch_norm : bool, optional
        Whether to use batch normalization, by default True
    dropout : float, optional
        Dropout probability, by default 0.1
    device : str, optional
        Device to use ('cpu' or 'cuda'), by default 'cpu'
        
    Returns
    -------
    GuidedAutoencoder
        Initialized guided autoencoder model
    """
    # Initialize model
    model = GuidedAutoencoder(
                    input_dim=input_dim, 
                    encoding_dim=encoding_dim, 
                    output_dim=output_dim, 
                    hidden_dims=hidden_dims, 
                    guided_lambda=guided_lambda,
                    use_batch_norm=use_batch_norm,
                    dropout=dropout
                )
    return model.to(device)

def setup_guided_training(model: GuidedAutoencoder, learning_rate: float, 
                         autoencoder_regularization: float = 1e-5
                         ) -> Tuple[nn.Module, nn.Module, optim.Optimizer, optim.lr_scheduler.ReduceLROnPlateau]:
    """
    Set up training components for guided autoencoder.
    
    Parameters
    ----------
    model : GuidedAutoencoder
        The guided autoencoder model
    learning_rate : float
        Initial learning rate
    autoencoder_regularization : float, optional
        Weight decay for regularization, by default 1e-5
        
    Returns
    -------
    Tuple[nn.Module, nn.Module, optim.Optimizer, optim.lr_scheduler.ReduceLROnPlateau]
        Reconstruction loss criterion, classification loss criterion, optimizer, and learning rate scheduler
    """
    # Define loss functions
    recon_criterion = nn.MSELoss()
    class_criterion = nn.CrossEntropyLoss()
    
    # Add optimizer for both autoencoder and classifier
    optimizer = optim.Adam(
                        params=(list(model.autoencoder.parameters()) + 
                              (list(model.classifier.parameters()) if model.classifier is not None else [])), 
                        lr=learning_rate, 
                        weight_decay=autoencoder_regularization
                    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                                        optimizer=optimizer, 
                                        mode='min', 
                                        patience=5, 
                                        factor=0.5
                                    )
    
    return recon_criterion, class_criterion, optimizer, scheduler

def update_quantum_embeddings(model: GuidedAutoencoder, 
                             X: torch.Tensor,
                             quantum_layer: Any,
                             n_shots: int,
                             device: str,
                             batch_size_qe: int = 50,
                             detuning_max: float = 6.0,
                             verbose: bool = True) -> QuantumSurrogate:
    """
    Update quantum embeddings and train surrogate model.
    
    Parameters
    ----------
    model : GuidedAutoencoder
        The guided autoencoder model
    X : torch.Tensor
        Input data tensor
    quantum_layer : Any
        Quantum layer for generating embeddings
    n_shots : int
        Number of shots for quantum simulation
    device : str
        Device to use ('cpu' or 'cuda')
    batch_size_qe : int, optional
        Batch size for quantum embedding computation, by default 50
    detuning_max : float, optional
        Maximum detuning for scaling, by default 6.0
    verbose : bool, optional
        Whether to show progress, by default True
        
    Returns
    -------
    QuantumSurrogate
        Trained surrogate model
    """
    model.clear_cache()
    n_samples = X.shape[0]
            
    # Get encoder's current state
    model.autoencoder.eval()
    
    # Get quantum embeddings for all samples
    with torch.no_grad():
        # Process in batches for memory efficiency
        all_encodings = []
        all_quantum_embs = []
        
        batch_iterator = tqdm(range(0, n_samples, batch_size_qe), desc="Computing quantum embeddings", 
                              position=1, leave=False) if verbose else range(0, n_samples, batch_size_qe)
        
        for i in batch_iterator:
            end_idx = min(i + batch_size_qe, n_samples)
            batch_X = X[i:end_idx]
            
            # Encode with current autoencoder
            batch_encoding = model.autoencoder.encode(batch_X)
            all_encodings.append(batch_encoding)
            
            # Get numpy encoding for quantum layer
            batch_encoding_np = batch_encoding.cpu().numpy()
            
            # Scale for quantum processing
            cur_spectral = max(abs(batch_encoding_np.max()), abs(batch_encoding_np.min()))
            if cur_spectral > 0:  # Avoid division by zero
                batch_encoding_scaled = batch_encoding_np.T / cur_spectral * detuning_max
            else:
                batch_encoding_scaled = batch_encoding_np.T
            
            # Get quantum embeddings
            quantum_embs = quantum_layer.apply_layer(
                                        x=batch_encoding_scaled, 
                                        n_shots=n_shots, 
                                        show_progress=False
                                    )
            
            # Cache quantum embeddings and collect for surrogate training
            for j, idx in enumerate(range(i, end_idx)):
                model.quantum_embeddings_cache[idx] = quantum_embs[:, j]
            
            # Convert to tensor and collect for surrogate training
            quantum_embs_tensor = torch.tensor(
                                    quantum_embs.T, 
                                    dtype=torch.float32
                                ).to(device)
            all_quantum_embs.append(quantum_embs_tensor)
        
        # Concatenate all encodings and quantum embeddings for surrogate training
        all_encodings_tensor = torch.cat(all_encodings, dim=0)
        all_quantum_embs_tensor = torch.cat(all_quantum_embs, dim=0)
        
        # Train or update surrogate model
        if verbose:
            tqdm.write("Training surrogate model with current quantum embeddings...")
        
        if model.surrogate is None:
            # Create and train new surrogate
            surrogate = create_and_train_surrogate(
                quantum_layer=quantum_layer,
                encodings=all_encodings_tensor,
                quantum_embeddings=all_quantum_embs_tensor,
                device=device,
                verbose=verbose,
                n_shots=n_shots
            )
        else:
            # Update existing surrogate
            surrogate = train_surrogate(
                surrogate_model=model.surrogate,
                quantum_layer=quantum_layer,
                input_data=all_encodings_tensor,
                quantum_embeddings=all_quantum_embs_tensor,
                epochs=100,  # Increase from 20 to 100
                batch_size=64,  # Larger batch size for better gradient estimates
                learning_rate=0.0005,  # Lower learning rate for more stable training
                device=device,
                verbose=verbose,
                n_shots=n_shots
            )
        
        # Set surrogate in model
        model.initialize_surrogate(surrogate)
        
    model.autoencoder.train()
    return surrogate

def train_guided_epoch(model: GuidedAutoencoder,
                      dataloader: torch.utils.data.DataLoader,
                      recon_criterion: nn.Module,
                      class_criterion: nn.Module,
                      optimizer: optim.Optimizer,
                      device: str = 'cpu',
                      verbose: bool = True,
                      recon_scale: float = 100.0,
                      class_scale: float = 1.0) -> Tuple[float, float, float]:
    """
    Train the guided autoencoder for one epoch.
    
    Parameters
    ----------
    model : GuidedAutoencoder
        The guided autoencoder model
    dataloader : torch.utils.data.DataLoader
        DataLoader for training data
    recon_criterion : nn.Module
        Reconstruction loss function
    class_criterion : nn.Module
        Classification loss function
    optimizer : optim.Optimizer
        Optimizer
    device : str, optional
        Device to use, by default 'cpu'
    verbose : bool, optional
        Whether to show detailed progress, by default True
    recon_scale : float, optional
        Scaling factor for reconstruction loss, by default 100.0
    class_scale : float, optional
        Scaling factor for classification loss, by default 1.0
        
    Returns
    -------
    Tuple[float, float, float]
        Average reconstruction loss, classification loss, and total loss for the epoch
    """
    model.autoencoder.train()  # Set model to training mode
    running_recon_loss = 0.0
    running_class_loss = 0.0
    running_total_loss = 0.0
    
    # Iterate through batches
    for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass through autoencoder
        batch_encoding, reconstructed = model.autoencoder(batch_X)

        if not batch_encoding.requires_grad:
            batch_encoding.requires_grad_(True)

        # Compute reconstruction loss
        recon_loss = recon_criterion(reconstructed, batch_X)
        
        # Forward pass through surrogate for differentiable training
        if model.surrogate is not None:
            # Use surrogate to get differentiable quantum embeddings
            surrogate_quantum_embs = model.surrogate(batch_encoding)
            
            # Forward pass through classifier
            batch_output = model.classifier(surrogate_quantum_embs)
            
            # Compute classification loss
            class_loss = class_criterion(batch_output, torch.argmax(batch_y, dim=1))
            
            # Scale the losses
            recon_loss_scaled = recon_loss * recon_scale
            class_loss_scaled = class_loss * class_scale

            if verbose and batch_idx % 10 == 0:
                tqdm.write(f"Batch {batch_idx}: Recon Loss Influence: {((1-model.guided_lambda)*recon_loss_scaled.item()):.4f}, "
                          f"Class Loss Influence: {(model.guided_lambda*class_loss_scaled.item()):.4f}")
            
            # Compute total loss with the weighting approach
            total_loss = (1 - model.guided_lambda) * recon_loss_scaled + model.guided_lambda * class_loss_scaled
        else:
            # If surrogate not available yet, use only reconstruction loss
            total_loss = recon_loss * recon_scale
            class_loss = torch.tensor(0.0).to(device)
            class_loss_scaled = class_loss
        
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
    
    return avg_recon_loss, avg_class_loss, avg_total_loss

def train_guided_autoencoder(
    data: np.ndarray, 
    labels: np.ndarray,
    quantum_layer: DetuningLayer,
    encoding_dim: int,
    hidden_dims: Optional[List[int]] = None,
    guided_lambda: float = 0.3,
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = 'cpu',
    quantum_update_frequency: int = 5,
    n_shots: int = 1000,
    verbose: bool = True,
    use_batch_norm: bool = True,
    dropout: float = 0.1,
    autoencoder_regularization: float = 1e-5,
    detuning_max: float = 6.0,
    recon_scale: float = 100.0,
    class_scale: float = 1.0
) -> Tuple[GuidedAutoencoder, float, Dict[str, List[float]]]:
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
    guided_lambda : float, optional
        Weight for classification loss (0-1), by default 0.3
        Loss = (1-lambda)*reconstruction_loss + lambda*classification_loss
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
    autoencoder_regularization : float, optional
        Regularization parameter for autoencoder, by default 1e-5
    detuning_max : float, optional
        Maximum detuning for scaling, by default 6.0    
    recon_scale : float, optional
        Scaling factor for reconstruction loss, by default 100.0
    class_scale : float, optional
        Scaling factor for classification loss, by default 1.0
        
    Returns
    -------
    Tuple[GuidedAutoencoder, float, Dict[str, List[float]]]
        Trained guided autoencoder, maximum absolute value for scaling, and loss history dictionary
    """
    
    # Prepare data
    input_dim = data.shape[0]
    output_dim = len(np.unique(labels))
    X, y, dataloader = prepare_guided_autoencoder_data(data, labels, batch_size, device, verbose)
    
    # Initialize model
    model = initialize_guided_autoencoder(
        input_dim=input_dim,
        encoding_dim=encoding_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        guided_lambda=guided_lambda,
        use_batch_norm=use_batch_norm,
        dropout=dropout,
        device=device
    )
    
    # Process one quantum embedding to determine dimension
    model.autoencoder.eval()
    with torch.no_grad():
        initial_encoding = model.autoencoder.encode(X[:1]).cpu().numpy()
        
        spectral = max(abs(initial_encoding.max()), abs(initial_encoding.min()))
        initial_encoding_scaled = initial_encoding.T / spectral * detuning_max
        
        if verbose:
            print(f"Getting initial quantum embeddings to determine dimension...")
        
        # Get quantum embeddings to determine dimension
        initial_quantum_embs = quantum_layer.apply_layer(
                    x=initial_encoding_scaled, 
                    n_shots=n_shots, 
                    show_progress=True
                )
        
        quantum_dim = initial_quantum_embs.shape[0]
        
        if verbose:
            tqdm.write(f"Quantum embedding dimension detected: {quantum_dim}")
        
        # Initialize classifier with correct dimension
        model.initialize_classifier(quantum_dim)
        model.classifier = model.classifier.to(device)
    
    # Setup training components
    recon_criterion, class_criterion, optimizer, scheduler = setup_guided_training(
        model=model,
        learning_rate=learning_rate,
        autoencoder_regularization=autoencoder_regularization
    )
    
    # Store current learning rate for tracking changes
    current_lr = learning_rate
    
    # For early stopping
    best_loss = float('inf') 
    patience_counter = 0
    patience = 10
    best_model_state = None
    
    # Track loss history
    loss_history = {
        "recon_loss": [],
        "class_loss": [],
        "total_loss": []
    }
    
    # Training loop
    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc="Training guided autoencoder", position=0, leave=False)
    
    # Initialize surrogate to None - will be created during first update
    surrogate = None
    
    for epoch in iterator:
        # Update quantum embeddings and surrogate periodically
        if (epoch % quantum_update_frequency == 0):
            if verbose:
                tqdm.write(f"Epoch {epoch+1}: Updating quantum embeddings and surrogate model...")
            
            surrogate = update_quantum_embeddings(
                model=model,
                X=X,
                quantum_layer=quantum_layer,
                n_shots=n_shots,
                device=device,
                batch_size_qe=50,
                detuning_max=detuning_max,
                verbose=verbose
            )
        
        # Train for one epoch
        avg_recon_loss, avg_class_loss, avg_total_loss = train_guided_epoch(
            model=model,
            dataloader=dataloader,
            recon_criterion=recon_criterion,
            class_criterion=class_criterion,
            optimizer=optimizer,
            device=device,
            verbose=verbose,
            recon_scale=recon_scale,
            class_scale=class_scale
        )
        
        # Store losses in history
        loss_history["recon_loss"].append(avg_recon_loss)
        loss_history["class_loss"].append(avg_class_loss)
        loss_history["total_loss"].append(avg_total_loss)
        
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
                'classifier': model.classifier.state_dict().copy(),
                'surrogate': model.surrogate.state_dict().copy() if model.surrogate is not None else None
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    tqdm.write(f"Early stopping triggered after {epoch+1} epochs")
                # Restore best model state
                model.autoencoder.load_state_dict(best_model_state['autoencoder'])
                model.classifier.load_state_dict(best_model_state['classifier'])
                if best_model_state['surrogate'] is not None and model.surrogate is not None:
                    model.surrogate.load_state_dict(best_model_state['surrogate'])
                break
    
    # Get final encoded representations for all data
    model.autoencoder.eval()
    encodings = []
    batch_size_eval = 256  # Larger batch size for evaluation
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size_eval):
            batch_X = X[i:i+batch_size_eval]
            batch_encoding = model.autoencoder.encode(batch_X).cpu().numpy()
            encodings.append(batch_encoding)
    
    encoded_data = np.vstack(encodings)
    
    # Compute max absolute value for scaling
    spectral = max(abs(encoded_data.max()), abs(encoded_data.min()))
    if spectral == 0:  # Avoid division by zero
        spectral = 1.0
    
    return model, spectral, loss_history


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
