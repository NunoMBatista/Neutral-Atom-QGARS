import os
import sys
import numpy as np
import random
import torch
import warnings
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional

# Fix the import path for the qrc_polyp_python module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from data_processing import create_polyp_dataset, show_sample_image
from pca_reduction import apply_pca, apply_pca_to_test_data, scale_to_detuning_range
from qrc_layer import DetuningLayer
from training import train
from visualization import plot_training_results, print_results

# Use local cli module, not system one
from qrc_polyp_python.cli import parse_args
import argparse

def main(args: Optional[argparse.Namespace] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, torch.nn.Module]]:
    """
    Main function to run the quantum reservoir computing pipeline.
    
    Parameters
    ----------
    args : Optional[argparse.Namespace]
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
    
    # Suppress warnings
    warnings.filterwarnings('ignore')

    # Define the path to the polyp dataset
    if args.data_dir is None:
        root_dir = os.path.dirname(os.path.abspath(__file__))
        generated_polyp_dataset = os.path.join(root_dir, "..", "data", "datasets", "cvc_clinic_db_patches")
    else:
        generated_polyp_dataset = args.data_dir
        
    polyp_dir = os.path.join(generated_polyp_dataset, "polyp")
    no_polyp_dir = os.path.join(generated_polyp_dataset, "no_polyp")
    
    print("Loading dataset...")
    data_train, data_test = create_polyp_dataset(
        polyp_dir, 
        no_polyp_dir, 
        split_ratio=args.split_ratio, 
        target_size=tuple(args.target_size)
    )
    
    # PCA Reduction
    print("\nPerforming PCA reduction...")
    xs_raw, ys, pca_model, spectral, encoder = apply_pca(
        data_train, 
        args.dim_pca, 
        args.num_examples
    )
    
    # Scale features to detuning range
    xs = scale_to_detuning_range(xs_raw, spectral, args.detuning_max)
    
    # Create quantum layer with Bloqade
    print("\nPreparing quantum simulation...")
    quantum_layer = DetuningLayer(
        geometry=args.geometry,
        n_atoms=args.dim_pca,
        lattice_spacing=args.lattice_spacing,
        rabi_freq=args.rabi_freq,
        t_end=args.evolution_time,
        n_steps=args.time_steps,
        readout_type=args.readout_type,
        encoding_scale=args.encoding_scale 
    )
    
    print("Running quantum simulation...")
    embeddings = quantum_layer.apply_layer(
        xs, 
        n_shots=args.n_shots, 
        show_progress=not args.no_progress
    )
    
    # Prepare test data
    print("\nPreparing test data...")
    test_features_raw = apply_pca_to_test_data(
        data_test,
        pca_model,
        spectral,
        args.dim_pca,
        args.num_test_examples
    )
    
    # Scale test features to detuning range
    test_features = scale_to_detuning_range(test_features_raw, spectral, args.detuning_max)
    
    test_targets = data_test["targets"][:args.num_test_examples]
    
    print("Computing quantum embeddings for test data...")
    test_embeddings = quantum_layer.apply_layer(
        test_features, 
        n_shots=args.n_shots, 
        show_progress=not args.no_progress
    )
    
    # Train different models
    results = {}
    
    # Train linear classifier using PCA features directly (baseline)
    print("\nTraining linear classifier on PCA features...")
    loss_lin, accs_train_lin, accs_test_lin, model_lin = train(
        xs, ys, test_features, test_targets, 
        regularization=args.regularization, 
        nepochs=args.nepochs, 
        batchsize=args.batchsize, 
        learning_rate=args.learning_rate,
        verbose=not args.no_progress,
        nonlinear=False
    )
    results["PCA+linear"] = (loss_lin, accs_train_lin, accs_test_lin, model_lin)
    
    # Train with QRC embeddings
    print("\nTraining linear classifier on QRC embeddings...")
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
    
    # Train neural network on PCA features
    print("\nTraining neural network on PCA features...")
    loss_nn, accs_train_nn, accs_test_nn, model_nn = train(
        xs, ys, test_features, test_targets, 
        regularization=args.regularization, 
        nepochs=args.nepochs, 
        batchsize=args.batchsize, 
        learning_rate=args.learning_rate,
        verbose=not args.no_progress,
        nonlinear=True
    )
    results["PCA+NN"] = (loss_nn, accs_train_nn, accs_test_nn, model_nn)
    
    # Print and visualize results
    print_results(results)
    
    if not args.no_plot:
        plot_training_results(results)
    
    return results

if __name__ == "__main__":
    args = parse_args()
    main(args)
