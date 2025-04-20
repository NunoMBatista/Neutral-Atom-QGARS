import os
import numpy as np
import random
import torch
import warnings
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

# Import custom modules
from data_processing import create_polyp_dataset, show_sample_image
from pca_reduction import apply_pca, apply_pca_to_test_data, scale_to_detuning_range
from qrc_layer import DetuningLayer
from training import train
from visualization import plot_training_results, print_results

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

def main() -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, torch.nn.Module]]:
    """
    Main function to run the quantum reservoir computing pipeline.
    
    Returns
    -------
    Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, torch.nn.Module]]
        Dictionary of results for each model
    """
    # Define the path to the generated polyp dataset
    root_dir = os.path.dirname(os.path.abspath(__file__))
    generated_polyp_dataset = os.path.join(root_dir, "..", "data", "datasets", "cvc_clinic_db_patches")
    polyp_dir = os.path.join(generated_polyp_dataset, "polyp")
    no_polyp_dir = os.path.join(generated_polyp_dataset, "no_polyp")
    
    print("Loading dataset...")
    data_train, data_test = create_polyp_dataset(polyp_dir, no_polyp_dir, split_ratio=0.8, target_size=(128, 128))
    
    # Display a random image from the training set
    show_sample_image(data_train)
    
    # PCA Reduction
    print("\nPerforming PCA reduction...")
    dim_pca = 8
    num_examples = 1000
    xs_raw, ys, pca_model, spectral, encoder = apply_pca(data_train, dim_pca, num_examples)
    
    # Scale features to detuning range
    detuning_max = 6.0
    xs = scale_to_detuning_range(xs_raw, spectral, detuning_max)
    
    # Create quantum layer with Bloqade
    print("\nPreparing quantum simulation...")
    quantum_layer = DetuningLayer(
        n_atoms=dim_pca,
        rabi_freq=2*np.pi,  # Rabi frequency
        t_end=4.0,         # Evolution time
        n_steps=8          # Number of time steps
    )
    
    print("Running quantum simulation...")
    embeddings = quantum_layer.apply_layer(xs)
    
    # Prepare test data
    print("\nPreparing test data...")
    num_test_examples = 400
    
    test_features_raw = apply_pca_to_test_data(
        data_test,
        pca_model,
        spectral,
        dim_pca,
        num_test_examples
    )
    
    # Scale test features to detuning range
    test_features = scale_to_detuning_range(test_features_raw, spectral, detuning_max)
    
    test_targets = data_test["targets"][:num_test_examples]
    
    print("Computing quantum embeddings for test data...")
    test_embeddings = quantum_layer.apply_layer(test_features)
    
    # Train different models
    results = {}
    
    # Train linear classifier using PCA features directly (baseline)
    print("\nTraining linear classifier on PCA features...")
    loss_lin, accs_train_lin, accs_test_lin, model_lin = train(
        xs, ys, test_features, test_targets, 
        regularization=0.0005, 
        nepochs=100, 
        batchsize=1000, 
        learning_rate=0.01,
        verbose=True,
        nonlinear=False
    )
    results["PCA+linear"] = (loss_lin, accs_train_lin, accs_test_lin, model_lin)
    
    # Train with QRC embeddings
    print("\nTraining linear classifier on QRC embeddings...")
    loss_qrc, accs_train_qrc, accs_test_qrc, model_qrc = train(
        embeddings, ys, test_embeddings, test_targets, 
        regularization=0.0005, 
        nepochs=100, 
        batchsize=1000, 
        learning_rate=0.01,
        verbose=True,
        nonlinear=False
    )
    results["QRC"] = (loss_qrc, accs_train_qrc, accs_test_qrc, model_qrc)
    
    # Train neural network on PCA features
    print("\nTraining neural network on PCA features...")
    loss_nn, accs_train_nn, accs_test_nn, model_nn = train(
        xs, ys, test_features, test_targets, 
        regularization=0.0005, 
        nepochs=100, 
        batchsize=1000, 
        learning_rate=0.01,
        verbose=True,
        nonlinear=True
    )
    results["PCA+NN"] = (loss_nn, accs_train_nn, accs_test_nn, model_nn)
    
    # Print and visualize results
    print_results(results)
    plot_training_results(results)
    
    return results

if __name__ == "__main__":
    main()
