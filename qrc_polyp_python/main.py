import os
import sys
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional

# Fix the import path for the qrc_polyp_python module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from data_processing import load_dataset, show_sample_image, flatten_images
from feature_reduction import (
    apply_pca, apply_pca_to_test_data, 
    apply_autoencoder, apply_autoencoder_to_test_data,
    scale_to_detuning_range
)
from qrc_layer import DetuningLayer
from training import train
from visualization import plot_training_results, print_results

from cli_utils import parse_args
import argparse

def main(args: Optional[argparse.Namespace] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, torch.nn.Module]]:
    """
    Main function to run the quantum reservoir computing pipeline.
    
    Parameters
    ----------
    Optional[argparse.Namespace]
    Command line arguments
    
    Returns
    -------
    Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, torch.nn.Module]]
        Dictionary of results for each model
    """
    
    # Parse arguments if not provided
    if args is None:
        args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    

    print("""
          
          =========================================
                    LOADING THE DATASET
          =========================================
          
          """)
    

    DATA_DIR = args.data_dir if args.data_dir else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "datasets")

    # Load dataset based on the specified dataset type
    if args.dataset_type == 'mnist':
        # Define the path to MNIST dataset
        data_train, data_test = load_dataset(
            'mnist',
            data_dir=DATA_DIR,
            target_size=tuple(args.target_size)
        )
        
    else:            
        DATASET_DIR = os.path.join(DATA_DIR, args.dataset_type)
        if not os.path.exists(DATASET_DIR):
            raise ValueError(f"Dataset directory does not exist: {DATASET_DIR}")
        
        data_train, data_test = load_dataset(
            'image_folder',
            data_dir=DATASET_DIR,
            target_size=tuple(args.target_size),
            split_ratio=args.split_ratio
        )

    print(f"Dataset loaded: {args.dataset_type}")
    print(f"Number of training samples: {data_train['metadata']['n_samples']}")
    print(f"Number of test samples: {data_test['metadata']['n_samples']}")
    print(f"Number of classes: {data_train['metadata']['n_classes']}")
    
    
    print("""
          
          =========================================
                PERFORMING FEATURE REDUCTION
          =========================================
          
          """)
    
    # Determine reduction dimension
    dim_reduction = args.dim_reduction
    
    # Perform feature reduction based on selected method
    method_name = args.reduction_method.lower()
    reduction_name = method_name.upper()  # For display in result labels
    
    if method_name == "pca":
        # Apply PCA reduction
        print("Using PCA for feature reduction...")
        xs_raw, ys, reduction_model, spectral, encoder = apply_pca(
            data_train, 
            dim_reduction, 
            args.num_examples
        )
        
        # Apply PCA to test data
        print("Applying PCA to test data...")
        test_features_raw = apply_pca_to_test_data(
            data_test,
            reduction_model,
            spectral,
            dim_reduction,
            args.num_test_examples
        )
        
    elif method_name == "autoencoder":
        # Use GPU if available and requested
        device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
        if args.gpu and not torch.cuda.is_available():
            print("Warning: GPU requested but not available. Using CPU instead.")
        
        print(f"Using autoencoder for feature reduction (device: {device})...")
        
        # Apply autoencoder reduction
        xs_raw, ys, reduction_model, spectral, encoder = apply_autoencoder(
            data_train,
            encoding_dim=dim_reduction,
            num_examples=args.num_examples,
            hidden_dims=args.autoencoder_hidden_dims,
            batch_size=args.autoencoder_batch_size,
            epochs=args.autoencoder_epochs,
            learning_rate=args.autoencoder_learning_rate,
            device=device,
            verbose=not args.no_progress
        )
        
        # Apply autoencoder to test data
        print("Applying autoencoder to test data...")
        test_features_raw = apply_autoencoder_to_test_data(
            data_test,
            reduction_model,
            args.num_test_examples,
            device=device,
            verbose=not args.no_progress
        )
        
    else:
        raise ValueError(f"Unknown reduction method: {method_name}")
    
    test_targets = data_test["targets"][:args.num_test_examples]

    print("""
          
        =========================================
                PREPARING QUANTUM LAYER
        =========================================
        
        """)
    
    # Scale features to detuning range
    xs = scale_to_detuning_range(xs_raw, spectral, args.detuning_max)
    
    # Scale test features to detuning range
    test_features = scale_to_detuning_range(test_features_raw, spectral, args.detuning_max)

    # Create quantum layer 
    quantum_layer = DetuningLayer(
        geometry=args.geometry,
        n_atoms=dim_reduction,
        lattice_spacing=args.lattice_spacing,
        rabi_freq=args.rabi_freq,
        t_end=args.evolution_time,
        n_steps=args.time_steps,
        readout_type=args.readout_type,
        encoding_scale=args.encoding_scale 
    )
   
    print("""
          
        =========================================
                 RUNNING QUANTUM LAYER
        =========================================
        
        """)
    
    print("Computing quantum embeddings for training data...")
    embeddings = quantum_layer.apply_layer(
        xs, 
        n_shots=args.n_shots, 
        show_progress=not args.no_progress
    )
    
    print("Computing quantum embeddings for test data...")
    test_embeddings = quantum_layer.apply_layer(
        test_features, 
        n_shots=args.n_shots, 
        show_progress=not args.no_progress
    )  

    
    print("""
          
        =========================================
             TRAINING LINEAR CLASSIFIER ON 
                   REDUCED FEATURES
        =========================================
        
        """)
    
    # Train different models
    results = {}
    
    loss_lin, accs_train_lin, accs_test_lin, model_lin = train(
        xs, ys, test_features, test_targets, 
        regularization=args.regularization, 
        nepochs=args.nepochs, 
        batchsize=args.batchsize, 
        learning_rate=args.learning_rate,
        verbose=not args.no_progress,
        nonlinear=False
    )
    results[f"{reduction_name}+linear"] = (loss_lin, accs_train_lin, accs_test_lin, model_lin)
    
    print("""
          
        =========================================
             TRAINING LINEAR CLASSIFIER ON 
                   QUANTUM EMBEDDINGS
        =========================================
        
        """)
    
    loss_qrc, accs_train_qrc, accs_test_qrc, model_qrc = train(
        embeddings, ys, test_embeddings, test_targets, 
        regularization=args.regularization, 
        nepochs=args.nepochs, 
        batchsize=args.batchsize, 
        learning_rate=args.learning_rate,
        verbose=not args.no_progress,
        nonlinear=False
    )
    results["QRC"] = (loss_qrc, accs_train_qrc, accs_test_qrc, model_qrc)
    
    
    print("""
          
        =========================================
                TRAINING NEURAL NETWORK 
                  ON REDUCED FEATURES
        =========================================
        
        """)
    loss_nn, accs_train_nn, accs_test_nn, model_nn = train(
        xs, ys, test_features, test_targets, 
        regularization=args.regularization, 
        nepochs=args.nepochs, 
        batchsize=args.batchsize, 
        learning_rate=args.learning_rate,
        verbose=not args.no_progress,
        nonlinear=True
    )
    results[f"{reduction_name}+NN"] = (loss_nn, accs_train_nn, accs_test_nn, model_nn)
    
    # Print and visualize results
    print_results(results)
    
    if not args.no_plot:
        plot_training_results(results)
    
    return results

if __name__ == "__main__":
    args = parse_args()
    main(args)
