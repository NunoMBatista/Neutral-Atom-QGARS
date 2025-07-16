from venv import create
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from typing import Tuple, List, Dict, Any, Optional

def create_default_architecture(
                    input_dim: int,
                    encoding_dim: int,
                    use_batch_norm: bool = True,
                    dropout: float = 0.0):
    
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
    print(f"Input dimension: {input_dim}")
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
        
    encoder = nn.Sequential(*encoder_layers)
    
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
    
    decoder = nn.Sequential(*decoder_layers)
    
    return encoder, decoder




def create_convolutional_architecture(
                    input_dim: int,
                    encoding_dim: int,
                    image_shape: Optional[Tuple[int, int, int]] = None,  # (H, W, C)
                    use_batch_norm: bool = True,
                    dropout: float = 0.0):
    """
    Create a convolutional autoencoder architecture.
    
    Parameters
    ----------
    input_dim : int
        Dimension of input features (flattened image size)
    encoding_dim : int
        Dimension of the encoded representation
    image_shape : Optional[Tuple[int, int, int]], optional
        Shape of the input image (height, width, channels), by default None
        If None, will try to infer a square image with 1 or 3 channels
    use_batch_norm : bool, optional
        Whether to use batch normalization, by default True
    dropout : float, optional
        Dropout probability, by default 0.0
        
    Returns
    -------
    Tuple[nn.Module, nn.Module]
        Encoder and decoder modules
    """
    # If image_shape is not provided, try to infer it
    if image_shape is None:
        # Check if input_dim is a perfect square (grayscale)
        sqrt_dim = int(np.sqrt(input_dim))
        if sqrt_dim * sqrt_dim == input_dim:
            # Assume square grayscale image
            height, width, channels = sqrt_dim, sqrt_dim, 1
        else:
            # Check if input_dim / 3 is a perfect square (RGB)
            sqrt_dim = int(np.sqrt(input_dim / 3))
            if sqrt_dim * sqrt_dim * 3 == input_dim:
                # Assume square RGB image
                height, width, channels = sqrt_dim, sqrt_dim, 3
            else:
                # Default to MNIST-like dimensions if we can't determine
                height, width = 28, 28
                # Try to determine channels
                if input_dim % (height * width) == 0:
                    channels = input_dim // (height * width)
                else:
                    channels = 1
                # Warn that we're using default dimensions
                print(f"Warning: Could not infer image shape from input_dim {input_dim}, "
                      f"using default {height}x{width}x{channels}")
    else:
        height, width, channels = image_shape
    
    print(f"""
        ********************************
         Creating Convolutional Encoder
        ********************************
        Input image shape: {height}x{width}x{channels}
        Encoding dimension: {encoding_dim}
    """)
    
    # Channel progression (increase channels as spatial dims decrease)
    c1 = 32
    c2 = 64
    c3 = 128
    
    # Create encoder
    encoder_layers = []
    
    # Initial reshape layer to convert from flat to 4D tensor
    # Convolutional layers expect input in (batch_size, channels, height, width) format
    class Reshape(nn.Module):
        def __init__(self, height, width, channels):
            super(Reshape, self).__init__()
            self.height = height
            self.width = width
            self.channels = channels
            
        def forward(self, x):
            # -1 infers batch size automatically
            return x.view(-1, self.channels, self.height, self.width)
        
    
    encoder_layers.append(Reshape(height, width, channels))
    
    # Create a test tensor to determine output dimensions dynamically
    with torch.no_grad():
        # Create a single dummy input tensor with batch dimension of 1
        x = torch.zeros(1, channels, height, width)
        
        # Apply first conv layer
        conv1 = nn.Conv2d(channels, c1, kernel_size=3, stride=2, padding=1)
        x = conv1(x)
        h1, w1 = x.shape[2], x.shape[3]
        
        # Apply second conv layer
        conv2 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)
        x = conv2(x)
        h2, w2 = x.shape[2], x.shape[3]
        
        # Apply third conv layer
        conv3 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1)
        x = conv3(x)
        h3, w3 = x.shape[2], x.shape[3]
        
        # Calculate actual flattened dimension
        flat_dim = int(np.prod(x.shape[1:]))  # This is c3 * h3 * w3
        
    
    print(f"""
        Calculated actual dimensions:
        - Conv1 output: {h1}x{w1}x{c1}
        - Conv2 output: {h2}x{w2}x{c2}
        - Conv3 output: {h3}x{w3}x{c3}
        - Flattened dimension: {flat_dim}
    """)
    
    # Add actual conv layers to the encoder
    encoder_layers.append(nn.Conv2d(channels, c1, kernel_size=3, stride=2, padding=1))
    if use_batch_norm:
        encoder_layers.append(nn.BatchNorm2d(c1))
    encoder_layers.append(nn.LeakyReLU(0.2))
    if dropout > 0:
        encoder_layers.append(nn.Dropout2d(dropout))
    
    
    encoder_layers.append(nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1))
    if use_batch_norm:
        encoder_layers.append(nn.BatchNorm2d(c2))
    encoder_layers.append(nn.LeakyReLU(0.2))
    if dropout > 0:
        encoder_layers.append(nn.Dropout2d(dropout))
    
    
    encoder_layers.append(nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1))
    if use_batch_norm:
        encoder_layers.append(nn.BatchNorm2d(c3))
    encoder_layers.append(nn.LeakyReLU(0.2))
    if dropout > 0:
        encoder_layers.append(nn.Dropout2d(dropout))
    
    
    # Flatten layer
    class Flatten(nn.Module):
        def forward(self, x):
            return x.view(x.size(0), -1)
    
    
    encoder_layers.append(Flatten())
    
    # Create a more gradual bottleneck with intermediate fully-connected layers
    # Instead of directly going from flat_dim to encoding_dim
    intermediate_dims = []
    
    # Calculate intermediate dimensions for a smoother reduction
    # Only add intermediate layers if flat_dim is significantly larger than encoding_dim
    if flat_dim > encoding_dim * 8:
        # Geometric progression of dimensions
        dim1 = min(flat_dim // 4, 512)  # First reduction
        dim2 = min(dim1 // 4, 128)      # Second reduction
        
        intermediate_dims = [dim1, dim2]
        
        # Remove dimensions that are smaller than encoding_dim
        intermediate_dims = [dim for dim in intermediate_dims if dim > encoding_dim]
        
        print(f"Adding intermediate encoder dimensions for smoother bottleneck: {intermediate_dims}")
    
    # Add intermediate layers if needed
    prev_dim = flat_dim
    for dim in intermediate_dims:
        # Add linear layer
        encoder_layers.append(nn.Linear(prev_dim, dim))
        if use_batch_norm:
            encoder_layers.append(nn.BatchNorm1d(dim))
        encoder_layers.append(nn.LeakyReLU(0.2))
        if dropout > 0:
            encoder_layers.append(nn.Dropout(dropout))
        prev_dim = dim
    
    # Final encoding layer
    encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
    if use_batch_norm:
        encoder_layers.append(nn.BatchNorm1d(encoding_dim))
    
    encoder = nn.Sequential(*encoder_layers)
    
    print("""
        ********************************
         Creating Convolutional Decoder
        ********************************    
    
    """)
    
    # Create decoder
    decoder_layers = []
    
    # First expand from encoding_dim to intermediate dimensions in reverse
    prev_dim = encoding_dim
    for dim in reversed(intermediate_dims):
        decoder_layers.append(nn.Linear(prev_dim, dim))
        if use_batch_norm:
            decoder_layers.append(nn.BatchNorm1d(dim))
        decoder_layers.append(nn.LeakyReLU(0.2))
        if dropout > 0:
            decoder_layers.append(nn.Dropout(dropout))
        prev_dim = dim
    
    # Then expand to flattened convolutional dimension
    decoder_layers.append(nn.Linear(prev_dim, flat_dim))
    if use_batch_norm:
        decoder_layers.append(nn.BatchNorm1d(flat_dim))
    decoder_layers.append(nn.LeakyReLU(0.2))
    
    # Reshape to 4D tensor
    class ReshapeTo4D(nn.Module):
        def __init__(self, c3, h3, w3):
            super(ReshapeTo4D, self).__init__()
            self.c3 = c3
            self.h3 = h3
            self.w3 = w3
            
        def forward(self, x):
            return x.view(-1, self.c3, self.h3, self.w3)
    
    decoder_layers.append(ReshapeTo4D(c3, h3, w3))
    
    # Transposed conv layers (upsampling)
    decoder_layers.append(nn.ConvTranspose2d(
        c3, c2, kernel_size=3, stride=2, padding=1, 
        output_padding=(1 if h2 % 2 != 0 or h3 * 2 != h2 else 0, 
                       1 if w2 % 2 != 0 or w3 * 2 != w2 else 0)))
    if use_batch_norm:
        decoder_layers.append(nn.BatchNorm2d(c2))
    decoder_layers.append(nn.LeakyReLU(0.2))
    if dropout > 0:
        decoder_layers.append(nn.Dropout2d(dropout))
    
    decoder_layers.append(nn.ConvTranspose2d(
        c2, c1, kernel_size=3, stride=2, padding=1,
        output_padding=(1 if h1 % 2 != 0 or h2 * 2 != h1 else 0,
                       1 if w1 % 2 != 0 or w2 * 2 != w1 else 0)))
    if use_batch_norm:
        decoder_layers.append(nn.BatchNorm2d(c1))
    decoder_layers.append(nn.LeakyReLU(0.2))
    if dropout > 0:
        decoder_layers.append(nn.Dropout2d(dropout))
    
    decoder_layers.append(nn.ConvTranspose2d(
        c1, channels, kernel_size=3, stride=2, padding=1,
        output_padding=(1 if height % 2 != 0 or h1 * 2 != height else 0,
                       1 if width % 2 != 0 or w1 * 2 != width else 0)))
    decoder_layers.append(nn.Sigmoid())  # Sigmoid for pixel values in [0,1]
    
    # Final reshape to match input shape
    class FlattenOutput(nn.Module):
        def forward(self, x):
            return x.view(x.size(0), -1)
    
    decoder_layers.append(FlattenOutput())
    
    # Test the decoder up to this point to see actual output dimension
    test_decoder = nn.Sequential(*decoder_layers)
    
    # Set decoder to evaluation mode for dimension testing
    test_decoder.eval()  # Important: prevents BatchNorm error
    
    with torch.no_grad():
        # Create test input with batch size of 2 to avoid BatchNorm issues
        test_input = torch.zeros(2, encoding_dim)
        # Run through decoder except final projection
        decoded = test_decoder(test_input)
        actual_output_dim = decoded.size(1)
        
        print(f"Actual decoder output dimension: {actual_output_dim}")
        print(f"Target input dimension: {input_dim}")
        
        # Add final projection layer if dimensions don't match
        if actual_output_dim != input_dim:
            print(f"Adding projection layer to match dimensions: {actual_output_dim} -> {input_dim}")
            decoder_layers.append(nn.Linear(actual_output_dim, input_dim))
            decoder_layers.append(nn.Sigmoid())  # Re-apply activation to ensure output range [0,1]
    
    decoder = nn.Sequential(*decoder_layers)
    
    return encoder, decoder