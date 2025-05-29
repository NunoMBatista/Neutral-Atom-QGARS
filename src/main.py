from logging import config
from multiprocessing import reduction
import os
import sys
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional, List
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from src.feature_reduction.autoencoder.autoencoder import Autoencoder
from src.feature_reduction.autoencoder.guided_autoencoder import GuidedAutoencoder
from src.utils.statistics_tracking import save_all_statistics, setup_stats_directory

from src.data_processing.data_processing import load_dataset, show_sample_image, flatten_images, one_hot_encode, select_random_samples
from src.feature_reduction.feature_reduction import (
    apply_pca, apply_pca_to_test_data, 
    apply_autoencoder, apply_autoencoder_to_test_data,
    apply_guided_autoencoder, apply_guided_autoencoder_to_test_data,
    scale_to_detuning_range
)
from src.quantum_layer.qrc_layer import DetuningLayer
from src.classification_models.training import train
from src.utils.visualization import plot_training_results, print_results

from utils.cli_utils import get_args
import argparse

def main(args: Optional[argparse.Namespace] = None, results_dir: str = None) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, torch.nn.Module]], Optional[Dict[str, List[float]]]]:
    """
    Main function to run the quantum reservoir computing pipeline.
    
    Parameters
    ----------
    Optional[argparse.Namespace]
    Command line arguments
    
    Returns
    -------
    Tuple[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, torch.nn.Module]], Optional[Dict[str, List[float]]]]
        Dictionary of results for each model and guided autoencoder losses if available
    """
    
    # Parse arguments if not provided
    if args is None:
        args = get_args()
        

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # torch.manual_seed(int(time.time()))

    print("""
          
          =========================================
                    LOADING THE DATASET
          =========================================
          
          """)
    

    DATA_DIR = args.data_dir if args.data_dir else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "datasets")

    # Load dataset based on the specified dataset type
    if args.dataset_type in ["mnist", "fashion_mnist", "binary_mnist"]:
        # Define the path to MNIST dataset
        data_train, data_test = load_dataset(
            args.dataset_type,
            data_dir=DATA_DIR,
            target_size=tuple(args.target_size),
            num_examples=args.num_examples,
            num_test_examples=args.num_test_examples
        )
        
    else:            
        DATASET_DIR = os.path.join(DATA_DIR, args.dataset_type)
        if not os.path.exists(DATASET_DIR):
            raise ValueError(f"Dataset directory does not exist: {DATASET_DIR}")
        
        data_train, data_test = load_dataset(
            'image_folder',
            data_dir=DATASET_DIR,
            target_size=tuple(args.target_size),
            split_ratio=args.split_ratio,
            num_examples=args.num_examples,
            num_test_examples=args.num_test_examples
        )

    print(f"Dataset loaded: {args.dataset_type}")
    print(f"Number of training samples: {data_train['metadata']['n_samples']}")
    print(f"Number of test samples: {data_test['metadata']['n_samples']}")
    print(f"Number of classes: {data_train['metadata']['n_classes']}")
    
    # Prepare variables for feature reduction
    train_features = data_train["features"]
    train_targets = data_train["targets"]
    test_features = data_test["features"]
    test_targets = data_test["targets"]

    
    print("""
          
          =========================================
                PERFORMING FEATURE REDUCTION
          =========================================
          
          """)
    
    
    # Determine reduction dimension
    dim_reduction = args.dim_reduction
    
    # Prepare a container for guided autoencoder losses
    guided_autoencoder_losses = None
    
    # Perform feature reduction based on selected method
    method_name = args.reduction_method.lower()
    reduction_name = method_name.upper()  # For display in result labels
    
    # If there is no lambda or queries to the reservoir, it's just an autoencoder
    if args.reduction_method == "guided_autoencoder" and (args.guided_lambda == 0 or args.quantum_update_frequency == 0):
        method_name = "autoencoder" 
    
    # Clean memory between runs to avoid cache issues
    import gc
    gc.collect()
    
    if method_name == "pca":
        # Apply PCA reduction
        print("Using PCA for feature reduction...")
        xs_raw, reduction_model, spectral = apply_pca(
            data=data_train, 
            dim_pca=dim_reduction, 
            selected_features=train_features
        )
        
        # Apply PCA to test data
        print("Applying PCA to test data...")
        test_features_raw = apply_pca_to_test_data(
            data=data_test,
            pca_model=reduction_model,
            dim_pca=dim_reduction,
            selected_features=test_features
        )
        
    elif method_name == "autoencoder":
        # Use GPU if available and requested
        device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
        if args.gpu and not torch.cuda.is_available():
            print("Warning: GPU requested but not available. Using CPU instead.")
        
        print(f"Using autoencoder for feature reduction (device: {device})...")
        
        # Apply autoencoder reduction with improved parameters
        xs_raw, reduction_model, spectral = apply_autoencoder(
            data=data_train,
            encoding_dim=dim_reduction,
            batch_size=args.autoencoder_batch_size,
            epochs=args.autoencoder_epochs,
            learning_rate=args.autoencoder_learning_rate,
            device=device,
            verbose=not args.no_progress,
            use_batch_norm=True,
            dropout=0.1,
            autoencoder_regularization=args.autoencoder_regularization,
            selected_features=train_features
        )
        
        # Log the spectral range to help diagnose scaling issues
        print(f"Encoded data spectral range: {spectral}")
        
        # Apply autoencoder to test data
        print("Applying autoencoder to test data...")
        test_features_raw = apply_autoencoder_to_test_data(
            data_test,
            reduction_model,
            device=device,
            verbose=not args.no_progress,
            selected_features=test_features
        )
    
        print(reduction_model)
    
    elif method_name == "guided_autoencoder":
        # Use GPU if available and requested
        device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
        if args.gpu and not torch.cuda.is_available():
            print("Warning: GPU requested but not available. Using CPU instead.")
        
        print(f"Using quantum guided autoencoder for feature reduction (device: {device})...")
        
        # First create quantum layer for guided training
        print("Creating quantum layer for guided autoencoder training...")
        quantum_layer = DetuningLayer(
            geometry=args.geometry,
            n_atoms=dim_reduction,
            lattice_spacing=args.lattice_spacing,
            rabi_freq=args.rabi_freq,
            t_end=args.evolution_time,
            n_steps=args.time_steps,
            readout_type=args.readout_type,
            encoding_scale=args.encoding_scale,
        )
        
        # Apply guided autoencoder reduction with improved parameters
        xs_raw, reduction_model, spectral, guided_autoencoder_losses = apply_guided_autoencoder(
            data_train,
            quantum_layer=quantum_layer,
            encoding_dim=dim_reduction,
            guided_lambda=args.guided_lambda,
            batch_size=args.guided_batch_size,
            epochs=args.autoencoder_epochs,
            learning_rate=args.autoencoder_learning_rate,
            quantum_update_frequency=args.quantum_update_frequency,
            n_shots=args.n_shots,
            device=device,
            verbose=not args.no_progress,
            use_batch_norm=True,  # Enable batch normalization
            dropout=0.1,  # Add dropout for regularization
            autoencoder_regularization=args.autoencoder_regularization,  # Use autoencoder regularization
            selected_features=train_features,
            selected_targets=train_targets
        )
        
        print(reduction_model)
        
        # Explicitly clear cache after training
        reduction_model.clear_cache()
        
        # Log the spectral range to help diagnose scaling issues
        print(f"Encoded data spectral range: {spectral}")
        
        # Apply guided autoencoder to test data
        print("Applying guided autoencoder to test data...")
        test_features_raw = apply_guided_autoencoder_to_test_data(
            data_test,
            reduction_model,
            device=device,
            verbose=not args.no_progress,
            selected_features=test_features
        )
        
    else:
        raise ValueError(f"Unknown reduction method: {method_name}")
    
    
    # We already have our targets from the random selection
    ys_encoded, encoder = one_hot_encode(train_targets, data_train["metadata"]["n_classes"])
    ys = ys_encoded.T  # Transpose to match expected format


    print("""
          
        =========================================
                PREPARING QUANTUM LAYER
        =========================================
        
        """)

    
    # Scale features to detuning range with more diagnostic info
    print(f"Scaling features with spectral value: {spectral}")
    xs = scale_to_detuning_range(xs_raw, spectral, args.detuning_max)
    print(f"Scaled feature range: {xs.min()} to {xs.max()}")
    
    # Scale test features to detuning range
    test_features = scale_to_detuning_range(test_features_raw, spectral, args.detuning_max)
    print(f"Scaled test feature range: {test_features.min()} to {test_features.max()}")

    # Create quantum layer (reuse if we already created one for guided autoencoder)
    if method_name == "guided_autoencoder" and 'quantum_layer' in locals():
        print("Reusing quantum layer from guided autoencoder...")
    else:
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
   
    print(quantum_layer)
    
    print("""
          
        =========================================
                 RUNNING QUANTUM LAYER
        =========================================
        
        """)
    
    print("Computing quantum embeddings for training data...")
    embeddings = quantum_layer.apply_layer(
        x=xs, 
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
    
    loss_lin, accs_train_lin, accs_test_lin, model_lin, confusion_matrix_train, confusion_matrix_test, f1_train, f1_test = train(
        x_train=xs, 
        y_train=ys, 
        x_test=test_features, 
        y_test=test_targets, 
        regularization=args.classifier_regularization,  
        nepochs=args.nepochs, 
        batchsize=args.batchsize, 
        learning_rate=args.learning_rate,
        verbose=not args.no_progress,
        nonlinear=False
    )
    results["linear"] = (loss_lin, accs_train_lin, accs_test_lin, model_lin, confusion_matrix_train, confusion_matrix_test, f1_train, f1_test)
    
    print(model_lin)
    
    print("""
          
        =========================================
             TRAINING LINEAR CLASSIFIER ON 
                   QUANTUM EMBEDDINGS
        =========================================
        
        """)
    
    loss_qrc, accs_train_qrc, accs_test_qrc, model_qrc, confusion_matrix_train, confusion_matrix_test, f1_train, f1_test = train(
        embeddings, ys, test_embeddings, test_targets, 
        regularization=args.classifier_regularization, 
        nepochs=args.nepochs, 
        batchsize=args.batchsize, 
        learning_rate=args.learning_rate,
        verbose=not args.no_progress,
        nonlinear=False
    )
    results["QRC"] = (loss_qrc, accs_train_qrc, accs_test_qrc, model_qrc, confusion_matrix_train, confusion_matrix_test, f1_train, f1_test)
    
    print(model_qrc)
    
    print("""
          
        =========================================
                TRAINING NEURAL NETWORK 
                  ON REDUCED FEATURES
        =========================================
        
        """)
    loss_nn, accs_train_nn, accs_test_nn, model_nn, confusion_matrix_train, confusion_matrix_test, f1_train, f1_test = train(
        xs, ys, test_features, test_targets, 
        regularization=args.classifier_regularization,
        nepochs=args.nepochs, 
        batchsize=args.batchsize, 
        learning_rate=args.learning_rate,
        verbose=not args.no_progress,
        nonlinear=True
    )
    results["NN"] = (loss_nn, accs_train_nn, accs_test_nn, model_nn, confusion_matrix_train, confusion_matrix_test, f1_train, f1_test)
    
    print(model_nn)
    
    print(f"""
        ==========================================
            Dataset: {args.dataset_type}
            Reduction Method: {reduction_name}
            Number of training samples: {data_train['metadata']['n_samples']}
            Number of test samples: {data_test['metadata']['n_samples']}
            Number of classes: {data_train['metadata']['n_classes']}
            Number of features: {dim_reduction}
            Number of epochs: {args.nepochs}
            Batch size: {args.batchsize}
            Learning rate: {args.learning_rate}
            Classifier regularization: {args.classifier_regularization}
            Autoencoder regularization: {args.autoencoder_regularization}
            Number of shots: {args.n_shots}
        ===========================================
          """)
    
    # Print and visualize results
    print_results(results)
    if not args.no_plot:
        plot_training_results(results)
    
    # Save statistics if running as main script (not as part of parameter sweep)
    #if not hasattr(args, '_parameter_sweep') or not args._parameter_sweep:
    
    if results_dir is None: 
        results_dir = args.results_dir
        results_dir = setup_stats_directory(results_dir)
    
    output_dir = save_all_statistics(
        results_dict=results, 
        guided_losses=guided_autoencoder_losses,
        args=args,
        output_dir=results_dir
    )
    print(f"Saved run statistics to {output_dir}")
    
    # Save model specification strings 
    with open(os.path.join(results_dir, "model_specifications.txt"), "w") as f:
        f.write(quantum_layer.__str__(use_colors=False))
        if reduction_name != "PCA":
            f.write(reduction_model.__str__(use_colors=False))
        f.write(model_lin.__str__(use_colors=False))
        f.write(model_qrc.__str__(use_colors=False))
        f.write(model_nn.__str__(use_colors=False))
        
    
    return results, guided_autoencoder_losses

if __name__ == "__main__":
    args = get_args()
    main(args)
