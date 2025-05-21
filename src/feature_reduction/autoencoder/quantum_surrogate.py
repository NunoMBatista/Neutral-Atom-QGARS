import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple, List

from src.utils.cli_printing import print_sequential_model

def get_proportional_hidden_dims(input_dim: int, output_dim: int, 
                               scaling_factor: float = 1.0, 
                               min_width: int = 256, 
                               max_width: int = 8192) -> List[int]:
    """
    Calculate hidden dimensions proportional to input and output dimensions.
    
    Parameters
    ----------
    input_dim : int
        Dimension of input features
    output_dim : int
        Dimension of quantum embeddings
    scaling_factor : float, optional
        Scale the network width, by default 1.0
    min_width : int, optional
        Minimum width for any layer, by default 256
    max_width : int, optional
        Maximum width for any layer, by default 8192
        
    Returns
    -------
    List[int]
        List of hidden layer dimensions
    """
    # Base the network size on geometric mean of input and output dimensions
    geo_mean = int(np.sqrt(input_dim * output_dim) * scaling_factor)
    
    # Determine network width based on dimensions
    max_dim = max(input_dim, output_dim)
    
    # Scale network depth based on dimensionality
    num_layers = max(3, min(6, int(np.log2(max_dim / 128))))
    
    # Determine maximum hidden layer width
    max_hidden = min(max(geo_mean, min_width), max_width)
    
    # Create pyramid architecture that expands then contracts
    hidden_dims = []
    
    # First expanding half
    for i in range(num_layers // 2):
        layer_width = min_width + int((max_hidden - min_width) * 
                                     ((i + 1) / (num_layers // 2 + 1)))
        hidden_dims.append(layer_width)
    
    # Middle layer(s) at maximum width
    hidden_dims.append(max_hidden)
    if num_layers % 2 == 0:
        hidden_dims.append(max_hidden)
    
    # Contracting half
    for i in range(num_layers // 2):
        layer_width = max_hidden - int((max_hidden - min_width) * 
                                     ((i + 1) / (num_layers // 2 + 1)))
        hidden_dims.append(layer_width)
    
    return hidden_dims

class QuantumSurrogate(nn.Module):
    """
    Neural network surrogate model that mimics the quantum layer.
    
    Parameters
    ----------
    input_dim : int
        Dimension of input features
    output_dim : int
        Dimension of quantum embeddings
    hidden_dims : List[int], optional
        Dimensions of hidden layers, by default [256, 512, 256]
    activation : str, optional
        Activation function to use, by default 'relu'
    dropout : float, optional
        Dropout probability, by default 0.1
    use_batch_norm : bool, optional
        Whether to use batch normalization, by default True
    """
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int,
                 activation: str = 'relu',
                 dropout: float = 0.1,
                 use_batch_norm: bool = True):
        super(QuantumSurrogate, self).__init__()
        
        # TODO: MAKE THIS A FUNCTION OF THE QUANTUM EMBEDDINGS SIZE
        #if hidden_dims is None:
            # Default architecture
            #hidden_dims = [256, 512, 256]
            # For high-dimensional quantum embeddings
        if output_dim > 1000:
            scaling_factor = 2
        else:
            scaling_factor = 1.0
            
        hidden_dims = get_proportional_hidden_dims(
            input_dim=input_dim,
            output_dim=output_dim,
            scaling_factor=scaling_factor,
            min_width=256,
            max_width=8192
        )
        
        print(f"Auto-configured surrogate hidden dimensions: {hidden_dims}")
    
        # Select activation function
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU(0.2)
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        else:
            act_fn = nn.ReLU()
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dim))
            
            layers.append(act_fn)
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        print_sequential_model(self.network, model_name="Quantum Surrogate Model")
        
        
    def __str__(self, use_colors: bool = True) -> str:
        return print_sequential_model(self.network, model_name="Quantum Surrogate Model", use_colors=use_colors)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the surrogate model"""
        
        # Make sure x has gradients
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        
        
        
        return self.network(x)



def train_surrogate(surrogate_model: QuantumSurrogate,
                   quantum_layer: Any,
                   input_data: torch.Tensor,
                   quantum_embeddings: Optional[torch.Tensor] = None,
                   epochs: int = 50,
                   batch_size: int = 32,
                   learning_rate: float = 0.001,
                   device: str = 'cpu',
                   verbose: bool = True,
                   n_shots: int = 1000) -> QuantumSurrogate:
    """
    Train the surrogate model to mimic the quantum layer.
    
    Parameters
    ----------
    surrogate_model : QuantumSurrogate
        The surrogate model to train
    quantum_layer : Any
        The quantum layer to mimic
    input_data : torch.Tensor
        Input data for training
    quantum_embeddings : Optional[torch.Tensor], optional
        Pre-computed quantum embeddings, by default None
    epochs : int, optional
        Number of training epochs, by default 50
    batch_size : int, optional
        Batch size for training, by default 32
    learning_rate : float, optional
        Learning rate, by default 0.001
    device : str, optional
        Device to use ('cpu' or 'cuda'), by default 'cpu'
    verbose : bool, optional
        Whether to show progress bars, by default True
    n_shots : int, optional
        Number of shots for quantum simulation, by default 1000
        
    Returns
    -------
    QuantumSurrogate
        Trained surrogate model
    """
    # Ensure model is on the correct device and in training mode
    surrogate_model = surrogate_model.to(device)
    surrogate_model.train()
    
    # Set up optimizer and loss function
    optimizer = optim.Adam(surrogate_model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Convert input data to numpy for quantum layer if needed
    input_numpy = None
    if quantum_embeddings is None:
        # Detach and convert to numpy - no gradients needed for quantum computation
        if isinstance(input_data, torch.Tensor):
            input_numpy = input_data.detach().cpu().numpy()
            # If 2D input, transpose it to match expected quantum layer input format
            if len(input_numpy.shape) == 2:
                input_numpy = input_numpy.T
        else:
            input_numpy = input_data
            
        # Get quantum embeddings from the quantum layer
        if verbose:
            tqdm.write("Computing quantum embeddings for surrogate training...")
        quantum_embs = quantum_layer.apply_layer(
            x=input_numpy,
            n_shots=n_shots,
            show_progress=verbose
        )
        
        # Convert to PyTorch tensor
        quantum_embeddings = torch.tensor(quantum_embs.T, dtype=torch.float32)
    
    # Make sure quantum embeddings are on the correct device
    # quantum_embeddings = torch.tensor(quantum_embeddings, dtype=torch.float32).to(device)
    if isinstance(quantum_embeddings, torch.Tensor):
        quantum_embeddings = quantum_embeddings.clone().detach().to(device).requires_grad_(True)
    else:
        quantum_embeddings = torch.tensor(quantum_embeddings, dtype=torch.float32, requires_grad=True).to(device)
        
    # Prepare input data tensor
    if isinstance(input_data, np.ndarray):
        input_tensor = torch.tensor(input_data.T, dtype=torch.float32).to(device)
    else:
        input_tensor = input_data.detach().cpu()#.to(device)  # Detach to ensure we don't track unnecessary gradients
        
    quantum_embeddings_cpu = quantum_embeddings.cpu() if hasattr(quantum_embeddings, 'is_cuda') else quantum_embeddings
    # Determine batch size based on dataset size
    num_samples = input_tensor.shape[0]
    actual_batch_size = min(batch_size, num_samples)
    
    # Create simple dataset and dataloader with pinned memory for faster data transfer
    dataset = torch.utils.data.TensorDataset(input_tensor, quantum_embeddings_cpu)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=actual_batch_size, 
        shuffle=True,
        pin_memory=(device == 'cuda')  # Use pinned memory for GPU transfers
    )
    
    # Training loop
    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc="Training quantum surrogate")
    
    # For early stopping
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    best_model_state = None
    
    for epoch in iterator:
        running_loss = 0.0
        
        for inputs, targets in dataloader:
            # Move tensors to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = surrogate_model(inputs)
            
            if not outputs.requires_grad:
                outputs.requires_grad_(True)
            
            # Compute loss
            # The targets don't have gradients, so PyTorch won't make the loss require gradients
            # To fix this, we add a zero term that maintains the gradient path
            gradient_path = outputs.sum() * 0.0  # Creates path for gradients without affecting loss value
            loss = criterion(outputs, targets) + gradient_path
            
            # Explicitly set requires_grad=True on the loss tensor
            loss.requires_grad_(True)
            
            # Backward pass and optimize
            assert loss.requires_grad, "Loss must require grad to backprop!"
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        # Compute average loss
        epoch_loss = running_loss / len(dataset)
        
        if verbose:
            tqdm.write(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")
        
        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            best_model_state = surrogate_model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    tqdm.write(f"Early stopping triggered after {epoch+1} epochs")
                surrogate_model.load_state_dict(best_model_state)
                break
    
    # Ensure model is in evaluation mode before returning
    surrogate_model.eval()
    return surrogate_model


def create_and_train_surrogate(quantum_layer: Any,
                              encodings: torch.Tensor,
                              quantum_embeddings: Optional[torch.Tensor] = None,
                              device: str = 'cpu',
                              verbose: bool = True,
                              n_shots: int = 1000) -> QuantumSurrogate:
    """
    Create and train a surrogate model for the quantum layer.
    
    Parameters
    ----------
    quantum_layer : Any
        The quantum layer to mimic
    encodings : torch.Tensor
        Encoded data from autoencoder
    quantum_embeddings : Optional[torch.Tensor], optional
        Pre-computed quantum embeddings, by default None
    device : str, optional
        Device to use ('cpu' or 'cuda'), by default 'cpu'
    verbose : bool, optional
        Whether to show progress bars, by default True
    n_shots : int, optional
        Number of shots for quantum simulation, by default 1000
        
    Returns
    -------
    QuantumSurrogate
        Trained surrogate model
    """
    # Determine the output dimension for the surrogate model
    output_dim = None
    
    # If quantum embeddings are provided, use their shape
    if quantum_embeddings is not None:
        if isinstance(quantum_embeddings, torch.Tensor):
            output_dim = quantum_embeddings.shape[1]
        else:
            output_dim = quantum_embeddings.shape[1] if len(quantum_embeddings.shape) > 1 else quantum_embeddings.shape[0]
    else:
        # Otherwise, compute a sample quantum embedding to determine the dimension
        if verbose:
            print("Sampling one quantum embedding to determine dimension...")
            
        # Get a single sample for testing
        if isinstance(encodings, torch.Tensor):
            sample_encoding = encodings[0:1].detach().cpu().numpy()
            if len(sample_encoding.shape) == 2 and sample_encoding.shape[0] == 1:
                sample_encoding = sample_encoding.T
        else:
            sample_encoding = encodings[:, 0:1] if len(encodings.shape) > 1 else encodings[0:1]
            
        # Apply quantum layer
        sample_quantum_emb = quantum_layer.apply_layer(
            x=sample_encoding,
            n_shots=n_shots,
            show_progress=False
        )
        
        # Determine output dimension from sample
        output_dim = sample_quantum_emb.shape[0]
    
    # Determine input dimension
    if isinstance(encodings, torch.Tensor):
        input_dim = encodings.shape[1]
    else:
        input_dim = encodings.shape[0] if len(encodings.shape) == 1 else encodings.shape[0]
    
    if verbose:
        print(f"Creating surrogate model with input dim {input_dim} and output dim {output_dim}")
    
    # Create and initialize the surrogate model
    surrogate = QuantumSurrogate(
        input_dim=input_dim,
        output_dim=output_dim,
        dropout=0.1,
        use_batch_norm=True
    )
    
    # Train the surrogate model
    trained_surrogate = train_surrogate(
        surrogate_model=surrogate,
        quantum_layer=quantum_layer,
        input_data=encodings,
        quantum_embeddings=quantum_embeddings,
        epochs=50,
        batch_size=32,
        learning_rate=0.003,
        device=device,
        verbose=verbose,
        n_shots=n_shots
    )
    
    return trained_surrogate

