import os
import numpy as np
import random
import torch
import warnings
import matplotlib.pyplot as plt

# Import custom modules
from data_processing import create_polyp_dataset, show_sample_image
from pca_reduction import apply_pca, apply_pca_to_test_data, scale_features
from qrc_layer import DetuningLayer
from training import train
from visualization import plot_training_results, print_results

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

def main():
    # Define the path to the generated polyp dataset
    root_dir = os.path.dirname(os.path.abspath(__file__))
    generated_polyp_dataset = os.path.join(root_dir, "..", "data", "datasets", "cvc_clinic_db_patches")
    polyp_dir = os.path.join(generated_polyp_dataset, "polyp")
    no_polyp_dir = os.path.join(generated_polyp_dataset, "no_polyp")
    
    print("Loading dataset...")
    data_train, data_test = create_polyp_dataset(polyp_dir, no_polyp_dir, split_ratio=0.8, target_size=(128, 128))
    
    # Display a random image from the training set
    #show_sample_image(data_train)
    
    # PCA Reduction
    print("\nPerforming PCA reduction...")
    dim_pca = 8
    num_examples = 1000
    xs, ys, pca_model, encoder = apply_pca(data_train, dim_pca, num_examples)
    
    # Scale features to detuning range for quantum processing
    print("Scaling features to detuning range...")
    detuning_range = 6.0  # Quantum detuning range
    xs_scaled, spectral = scale_features(xs, detuning_range)
    
    # Create quantum layer with Bloqade
    print("\nPreparing quantum simulation...")
    quantum_layer = DetuningLayer(
        n_atoms=dim_pca,
        rabi_freq=2*np.pi,  # Rabi frequency
        t_end=4.0,         # Evolution time
        n_steps=8          # Number of time steps
    )
    
    print("Running quantum simulation...")
    embeddings = quantum_layer.apply_layer(xs_scaled)
    
    # Prepare test data
    print("\nPreparing test data...")
    num_test_examples = 400
    
    test_features = apply_pca_to_test_data(
        data_test,
        pca_model,
        dim_pca,
        num_test_examples
    )
    
    # Scale test features using the same spectral factor
    test_features_scaled, _ = scale_features(test_features, detuning_range)
    
    # Convert test targets to one-hot encoding like training targets
    test_labels = data_test["targets"][:num_test_examples]
    test_targets = encoder.transform(test_labels.reshape(-1, 1)).T  # Transform and transpose
    
    print("Computing quantum embeddings for test data...")
    test_embeddings = quantum_layer.apply_layer(test_features_scaled)
    
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
