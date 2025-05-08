import argparse
import numpy as np
import sys

# Define available atom geometries
AVAILABLE_GEOMETRIES = ["chain"] 
AVAILABLE_READOUT_TYPES = ["Z", "ZZ", "all"]
AVAILABLE_REDUCTION_METHODS = ["pca", "autoencoder", "guided_autoencoder"]

def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Quantum Reservoir Computing for Image Classification")
    
    # Dataset parameters
    parser.add_argument("--dataset-type", type=str, default="cvc_clinic_db_patches",
                       help="Type of dataset to use")
    parser.add_argument("--data-dir", type=str, default=None, 
                       help="Path to dataset directory containing class subfolders")
    parser.add_argument("--target-size", type=int, nargs=2, default=[128, 128],
                       help="Size to resize images to")
    parser.add_argument("--split-ratio", type=float, default=0.8,
                       help="Train/test split ratio")
    
    # Feature reduction parameters
    parser.add_argument("--reduction-method", type=str, choices=AVAILABLE_REDUCTION_METHODS, default="autoencoder",
                       help="Feature reduction method to use")
    parser.add_argument("--dim-reduction", type=int, default=12,
                       help="Number of dimensions after reduction (PCA components or autoencoder encoding dim)")
    parser.add_argument("--num-examples", type=int, default=10000,
                       help="Number of examples to use for training")
    parser.add_argument("--num-test-examples", type=int, default=400,
                       help="Number of examples to use for testing")
    
    # Guided Autoencoder parameters
    parser.add_argument("--guided-alpha", type=float, default=0.7,
                      help="Weight for reconstruction loss in guided autoencoding (0-1)")
    parser.add_argument("--guided-beta", type=float, default=0.3,
                      help="Weight for classification loss in guided autoencoding (0-1)")
    parser.add_argument("--quantum-update-frequency", type=int, default=1,
                      help="Update quantum embeddings every N epochs")
    parser.add_argument("--guided-batch-size", type=int, default=32,
                      help="Batch size for guided autoencoder training")
    
    # Autoencoder parameters
    parser.add_argument("--autoencoder-epochs", type=int, default=50,
                       help="Number of epochs for autoencoder training")
    parser.add_argument("--autoencoder-batch-size", type=int, default=64,
                       help="Batch size for autoencoder training")
    parser.add_argument("--autoencoder-learning-rate", type=float, default=0.001,
                       help="Learning rate for autoencoder training")
    parser.add_argument("--autoencoder-hidden-dims", type=int, nargs="+", default=None,
                       help="Hidden dimensions for autoencoder (e.g. --autoencoder-hidden-dims 256 128)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for autoencoder training if available")
    
    # Quantum parameters
    parser.add_argument("--geometry", type=str, choices=AVAILABLE_GEOMETRIES, default="chain",
                       help="Atom geometry (only chain supported)")
    parser.add_argument("--lattice-spacing", type=float, default=10.0,
                       help="Spacing between atoms")
    parser.add_argument("--rabi-freq", type=float, default=2*np.pi,
                       help="Rabi frequency")
    parser.add_argument("--evolution-time", type=float, default=4.0,
                       help="Total evolution time")
    parser.add_argument("--time-steps", type=int, default=16,
                       help="Number of time steps")
    parser.add_argument("--readout-type", type=str, choices=AVAILABLE_READOUT_TYPES, default="all",
                       help="Type of readout")
    parser.add_argument("--n-shots", type=int, default=1000,
                       help="Number of shots for quantum simulation")
    parser.add_argument("--detuning-max", type=float, default=6.0,
                       help="Maximum detuning value")
    parser.add_argument("--encoding-scale", type=float, default=9.0,
                       help="Scale for encoding features as detunings")
    
    # Training parameters
    parser.add_argument("--regularization", type=float, default=0.0005,
                       help="Regularization strength")
    parser.add_argument("--nepochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batchsize", type=int, default=1000,
                       help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                       help="Learning rate")
    
    # Misc parameters
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--no-progress", action="store_true", default=True,
                       help="Disable progress bars")
    parser.add_argument("--no-plot", action="store_true",
                       help="Disable plotting")
    
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
    # Check if any arguments were provided (beyond script name)
    if len(sys.argv) == 1:
        # No command line args, use config file
        from config_manager import get_config_args
        return get_config_args()
    else:
        # Command line args provided, use them
        return parse_args()
