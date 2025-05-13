import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from typing import Tuple, List, Dict, Any, Optional

from autoencoder import Autoencoder
from qrc_layer import DetuningLayer
    
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
        
        # Embedding cache
        self.quantum_embeddings_cache = {}
    
    def initialize_classifier(self, quantum_dim: int):
        """Initialize classifier with the correct input dimension"""
        self.quantum_dim = quantum_dim
        
        # TODO: MIGHT USE MORE THAN A SINGLE LINEAR LAYER
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
    detuning_max: float = 6.0
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
                data=data.T, 
                dtype=torch.float32
            )
    
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
    dataloader = DataLoader(
            dataset=dataset, 
            batch_size=adjusted_batch_size, 
            shuffle=True
        )
    
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
    model = model.to(device)
    
    # Store indices to embeddings mapping (memory efficient)
    sample_indices = torch.arange(n_samples)
    
    # Process one quantum embedding to determine dimension
    model.autoencoder.eval()
    with torch.no_grad():
        initial_encoding = model.autoencoder.encode(X[:1].to(device)).cpu().numpy()
        
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
            print(f"Quantum embedding dimension detected: {quantum_dim}")
        
        # Initialize classifier with correct dimension
        model.initialize_classifier(quantum_dim)
        model.classifier = model.classifier.to(device)
    
    # Define loss functions
    recon_criterion = nn.MSELoss()
    class_criterion = nn.CrossEntropyLoss()
    
    # Add optimizer for both autoencoder and classifier
    optimizer = optim.Adam(
                        params=(list(model.autoencoder.parameters()) + list(model.classifier.parameters())), 
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
    
    # Store current learning rate for tracking changes
    current_lr = learning_rate
    
    # Training loop
    model.autoencoder.train() # Set autoencoder to training mode
    
    # For early stopping
    best_loss = float('inf') 
    patience_counter = 0
    patience = 10
    best_model_state = None
    
    iterator = range(epochs)
    if(verbose):
        iterator = tqdm(iterator, desc="Training guided autoencoder", position=0, leave=False)
    
    for epoch in iterator:
        running_recon_loss = 0.0
        running_class_loss = 0.0
        running_total_loss = 0.0
        
        # Update quantum embeddings periodically to save computation
        if epoch % quantum_update_frequency == 0:
            if verbose:
                tqdm.write(f"Epoch {epoch+1}: Updating quantum embeddings...")
            model.clear_cache()
            
            # Get encoder's current state
            model.autoencoder.eval() # Set autoencoder to evaluation mode
            # Get quantum embeddings for all samples
            with torch.no_grad():
                batch_size_qe = 50  # Process in smaller batches for memory efficiency
                
                batch_iterator = tqdm(range(0, n_samples, batch_size_qe), desc="Updating embeddings in batches", position=1, leave=False) if verbose else range(0, n_samples, batch_size_qe)
                for i in batch_iterator:
                    end_idx = min(i + batch_size_qe, n_samples)
                    batch_indices = sample_indices[i:end_idx]
                    batch_X = X[batch_indices].to(device)
                    
                    # Encode with current autoencoder
                    batch_encoding = model.autoencoder.encode(batch_X).cpu().numpy()
                    
                    # Scale for quantum processing
                    cur_spectral = max(abs(batch_encoding.max()), abs(batch_encoding.min()))
                    if cur_spectral > 0:  # Avoid division by zero
                        batch_encoding_scaled = batch_encoding.T / cur_spectral * detuning_max
                    else:
                        batch_encoding_scaled = batch_encoding.T
                    
                    # Get quantum embeddings
                    quantum_embs = quantum_layer.apply_layer(
                                                x=batch_encoding_scaled, 
                                                n_shots=n_shots, 
                                                show_progress=True
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
                quantum_embs_batch.append(model.quantum_embeddings_cache[idx])

            # Create a classifier for the current batch's quantum embeddings
            quantum_embs_tensor = torch.tensor(np.array(quantum_embs_batch), dtype=torch.float32).to(device)
            batch_output = model.classifier(quantum_embs_tensor)

            # Compute classification loss
            class_loss = class_criterion(batch_output, torch.argmax(batch_y, dim=1))

            # Scale the losses
            recon_loss *= 100.0
            class_loss *= 1.0

            tqdm.write(f"Reconstruction Loss Influence: {((1-model.guided_lambda)*recon_loss.item()):.4f}, Classification Loss Influence: {(model.guided_lambda*class_loss.item()):.4f}")
            
            # Compute total loss with the new weighting approach: (1-lambda)*recon_loss + lambda*class_loss
            total_loss = (1 - model.guided_lambda) * recon_loss + model.guided_lambda * class_loss
            
            # Backward pass and optimize
            total_loss.backward() # Compute gradients
            optimizer.step() # Update weights
            
            for name, param in model.classifier.named_parameters():
                if param.grad is None or torch.all(param.grad == 0):
                    print(f"Warning: {name} has no gradient!")
                print(param.grad)
            
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
