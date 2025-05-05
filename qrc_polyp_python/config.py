"""
Configuration file for QRC experiments
"""

import numpy as np
from typing import Dict, Any

# Default configurations for different experiments
DEFAULT_CONFIGS = {
    # Configuration for chain topology (only supported geometry)
    "chain": {
        # Dataset parameters
        "target_size": (128, 128),
        "split_ratio": 0.8,
        
        # PCA parameters
        "dim_pca": 8,
        "num_examples": 1000,
        "num_test_examples": 400,
        
        # Quantum parameters
        "geometry": "chain",
        "lattice_spacing": 10.0,
        "rabi_freq": 2*np.pi,
        "evolution_time": 4.0,
        "time_steps": 8,
        "readout_type": "ZZ",
        "n_shots": 1000,
        "detuning_max": 6.0,
        "encoding_scale": 9.0,
        
        # Training parameters
        "regularization": 0.0005,
        "nepochs": 100,
        "batchsize": 1000,
        "learning_rate": 0.01
    },
    
    # Configuration with different readout types
    "all_readouts": {
        # Dataset parameters
        "target_size": (128, 128),
        "split_ratio": 0.8,
        
        # PCA parameters
        "dim_pca": 8,
        "num_examples": 1000,
        "num_test_examples": 400,
        
        # Quantum parameters
        "geometry": "chain",
        "lattice_spacing": 10.0,
        "rabi_freq": 2*np.pi,
        "evolution_time": 4.0,
        "time_steps": 8,
        "readout_type": "all",  # Use all readout types
        "n_shots": 1000,
        "detuning_max": 6.0,
        "encoding_scale": 9.0,
        
        # Training parameters
        "regularization": 0.0005,
        "nepochs": 100,
        "batchsize": 1000,
        "learning_rate": 0.01
    }
}

def get_config(name: str = "chain") -> Dict[str, Any]:
    """
    Get configuration by name.
    
    Parameters
    ----------
    name : str, optional
        Configuration name, by default "chain"
    
    Returns
    -------
    Dict[str, Any]
        Configuration dictionary
    """
    if name not in DEFAULT_CONFIGS:
        raise ValueError(f"Unknown configuration: {name}. Available configurations: {list(DEFAULT_CONFIGS.keys())}")
    
    return DEFAULT_CONFIGS[name].copy()

def create_custom_config(base_config: str = "chain", **kwargs) -> Dict[str, Any]:
    """
    Create a custom configuration based on an existing one.
    
    Parameters
    ----------
    base_config : str, optional
        Base configuration name, by default "chain"
    **kwargs
        Parameters to override
    
    Returns
    -------
    Dict[str, Any]
        Custom configuration dictionary
    """
    config = get_config(base_config)
    
    # Override with custom parameters
    for key, value in kwargs.items():
        config[key] = value
    
    return config
