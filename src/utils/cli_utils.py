import argparse
import numpy as np
import sys

from src.utils.config_manager import ConfigManager
from src.globals import *


# ALWAYS SET THE DEFAULTS TO NONE
# This allows the config file to override them
# If the user does not provide a value, it will use the default from the config file
def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Quantum Reservoir Computing for Image Classification")
    
    # Dataset parameters
    parser.add_argument("--dataset-type", type=str, default=None,
                       help="Type of dataset to use")
    parser.add_argument("--data-dir", type=str, default=None, 
                       help="Path to dataset directory containing class subfolders")
    parser.add_argument("--target-size", type=int, nargs=2, default=None,
                       help="Size to resize images to")
    parser.add_argument("--split-ratio", type=float, default=None,
                       help="Train/test split ratio")
    
    # Feature reduction parameters
    parser.add_argument("--reduction-method", type=str, choices=AVAILABLE_REDUCTION_METHODS, default=None,
                       help="Feature reduction method to use")
    parser.add_argument("--dim-reduction", type=int, default=None,
                       help="Number of dimensions after reduction (PCA components or autoencoder encoding dim)")
    parser.add_argument("--num-examples", type=int, default=None,
                       help="Number of examples to use for training")
    parser.add_argument("--num-test-examples", type=int, default=None,
                       help="Number of examples to use for testing")
    
    # Guided Autoencoder parameters
    parser.add_argument("--guided-lambda", type=float, default=None,
                      help="Weight for classification loss in guided autoencoding (0-1)")
    parser.add_argument("--quantum-update-frequency", type=int, default=None,
                      help="Update quantum embeddings every N epochs")
    parser.add_argument("--guided-batch-size", type=int, default=None,
                      help="Batch size for guided autoencoder training")
    
    # Autoencoder parameters
    parser.add_argument("--ae-type", type=str, default=None, choices=AVAILABLE_AUTOENCODER_TYPES,
                          help="Type of autoencoder architecture to use (default, convolutional)")
    parser.add_argument("--autoencoder-epochs", type=int, default=None,
                       help="Number of epochs for autoencoder training")
    parser.add_argument("--autoencoder-batch-size", type=int, default=None,
                       help="Batch size for autoencoder training")
    parser.add_argument("--autoencoder-learning-rate", type=float, default=None,
                       help="Learning rate for autoencoder training")
    parser.add_argument("--autoencoder-hidden-dims", type=int, nargs="+", default=None,
                       help="Hidden dimensions for autoencoder (e.g. --autoencoder-hidden-dims 256 128)")
    parser.add_argument("--autoencoder-regularization", type=float, default=None,
                       help="Regularization strength for autoencoder training (weight decay)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for autoencoder training if available")
    
    # Quantum parameters
    parser.add_argument("--geometry", type=str, choices=AVAILABLE_GEOMETRIES, default=None,
                       help="Atom geometry (only chain supported)")
    parser.add_argument("--lattice-spacing", type=float, default=None,
                       help="Spacing between atoms")
    parser.add_argument("--rabi-freq", type=float, default=None,
                       help="Rabi frequency")
    parser.add_argument("--evolution-time", type=float, default=None,
                       help="Total evolution time")
    parser.add_argument("--time-steps", type=int, default=None,
                       help="Number of time steps")
    parser.add_argument("--readout-type", type=str, choices=AVAILABLE_READOUT_TYPES, default=None,
                       help="Type of readout")
    parser.add_argument("--n-shots", type=int, default=None,
                       help="Number of shots for quantum simulation")
    parser.add_argument("--detuning-max", type=float, default=None,
                       help="Maximum detuning value")
    parser.add_argument("--encoding-scale", type=float, default=None,
                       help="Scale for encoding features as detunings")
    
    # Training parameters
    parser.add_argument("--classifier-regularization", type=float, default=None,
                       help="Regularization strength for final classifiers")
    parser.add_argument("--nepochs", type=int, default=None,
                       help="Number of training epochs")
    parser.add_argument("--batchsize", type=int, default=None,
                       help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                       help="Learning rate")
    
    # Misc parameters
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed")
    parser.add_argument("--no-progress", action="store_true", default=False,
                       help="Disable progress bars")
    parser.add_argument("--no-plot", action="store_true",
                       help="Disable plotting")
    
    # Config file argument
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file")
    
    parser.add_argument("--results-dir", type=str, default=None,
                       help="Directory to save results")
    
    return parser.parse_args()

def get_args() -> argparse.Namespace:
    """
    Get arguments from either command line or config file.
    If no command-line arguments provided, load from config.
    
    Returns
    -------
    argparse.Namespace
        Parsed arguments
    """

    # Get the default config and replace the default values with the ones provided in the command line
    config = ConfigManager.get_default_config()
    args = parse_args()
    
    if args.config:
        # Load config from file
        config = ConfigManager.load_config(args.config)
    else: 
        # Use default config
        config = ConfigManager.get_default_config()
        print(config)
        
        
    # Update config with command line arguments
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
            
            
    # Convert config to argparse Namespace
    args = argparse.Namespace()
    for key, value in config.items():
        setattr(args, key, value)
        
    return args
