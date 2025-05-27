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

def create_convolutional_architecture():
    pass