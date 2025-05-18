import os
import sys
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from autoencoder import Autoencoder
from guided_autoencoder import GuidedAutoencoder

from statistics_tracking import save_all_statistics
from data_processing import load_dataset, show_sample_image, flatten_images, one_hot_encode, select_random_samples
from feature_reduction import (
    apply_feature_reduction, apply_feature_reduction_to_test_data,
    scale_to_detuning_range
)
from qrc_layer import DetuningLayer
from training import train
from visualization import plot_training_results, print_results

from cli_utils import get_args
import argparse

def get_dataset_path(args: argparse.Namespace) -> str:
    """
    Get dataset path based on configuration.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
        
    Returns
    -------
    str
        Path to dataset
    """
    # Use specified data directory or the default data directory
    return args.data_dir if args.data_dir else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "datasets"
    )

def setup_quantum_layer(args: argparse.Namespace, dim_reduction: int, is_new: bool = True) -> DetuningLayer:
    """
    Set up quantum layer with given parameters.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    dim_reduction : int
        Dimension of the reduced features
    is_new : bool, optional
        Whether this is a new layer or reusing one, by default True
        
    Returns
    -------
    DetuningLayer
        Configured quantum layer
    """
    print_status = is_new  # Only print parameters for new layers
    
    return DetuningLayer(
        geometry=args.geometry,
        n_atoms=dim_reduction,
        lattice_spacing=args.lattice_spacing,
        rabi_freq=args.rabi_freq,
        t_end=args.evolution_time,
        n_steps=args.time_steps,
        readout_type=args.readout_type,
        encoding_scale=args.encoding_scale,
        print_params=print_status
    )

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
    
    # Get dataset path and load data
    data_dir = get_dataset_path(args)
    
    # For predefined datasets like mnist or cifar10, load directly
    if args.dataset_type.lower() in ['mnist', 'cifar10']:
        data_train, data_test = load_dataset(
            args.dataset_type,
            data_dir=data_dir,
            target_size=tuple(args.target_size),
            num_examples=args.num_examples,
            num_test_examples=args.num_test_examples
        )
    else:
        # For custom image folder datasets, construct the path
        dataset_dir = os.path.join(data_dir, args.dataset_type)
        if not os.path.exists(dataset_dir):
            raise ValueError(f"Dataset directory does not exist: {dataset_dir}")
        
        data_train, data_test = load_dataset(
            'image_folder',
            data_dir=dataset_dir,
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
    
    # Set up device for GPU-accelerated models if available
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    if args.gpu and not torch.cuda.is_available():
        print("Warning: GPU requested but not available. Using CPU instead.")
    
    # Create quantum layer for guided autoencoder if needed
    quantum_layer = None
    if args.reduction_method.lower() == "guided_autoencoder":
        print("Creating quantum layer for guided autoencoder training...")
        quantum_layer = setup_quantum_layer(args, dim_reduction, is_new=True)
    
    # Apply feature reduction with the selected method
    feature_reduction_params = {
        'dim_reduction': dim_reduction,
        'autoencoder_hidden_dims': args.autoencoder_hidden_dims,
        'autoencoder_batch_size': args.autoencoder_batch_size,
        'autoencoder_epochs': args.autoencoder_epochs,
        'autoencoder_learning_rate': args.autoencoder_learning_rate,
        'autoencoder_regularization': args.autoencoder_regularization,
        'guided_lambda': args.guided_lambda,
        'guided_batch_size': args.guided_batch_size,
        'quantum_update_frequency': args.quantum_update_frequency,
        'n_shots': args.n_shots,
        'device': device,
        'verbose': not args.no_progress,
        'selected_features': train_features,
        'selected_targets': train_targets,
        'quantum_layer': quantum_layer
    }
    
    # Apply feature reduction using the router function
    xs_raw, reduction_model, spectral, guided_autoencoder_losses = apply_feature_reduction(
        method_name=args.reduction_method,
        data=data_train,
        **feature_reduction_params
    )
    
    # Log the spectral range to help diagnose scaling issues
    print(f"Encoded data spectral range: {spectral}")
    
    # Apply feature reduction to test data
    test_features_raw = apply_feature_reduction_to_test_data(
        method_name=args.reduction_method,
        data=data_test,
        reduction_model=reduction_model,
        **feature_reduction_params
    )
    
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

    # Create or reuse quantum layer
    if args.reduction_method.lower() == "guided_autoencoder" and quantum_layer is not None:
        print("Reusing quantum layer from guided autoencoder...")
    else:
        quantum_layer = setup_quantum_layer(args, dim_reduction)
   
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
    
    loss_lin, accs_train_lin, accs_test_lin, model_lin = train(
        x_train=xs, 
        y_train=ys, 
        x_test=test_features, 
        y_test=test_targets, 
        regularization=args.classifier_regularization,  # Use classifier regularization 
        nepochs=args.nepochs, 
        batchsize=args.batchsize, 
        learning_rate=args.learning_rate,
        verbose=not args.no_progress,
        nonlinear=False
    )
    results["linear"] = (loss_lin, accs_train_lin, accs_test_lin, model_lin)
    
    print("""
          
        =========================================
             TRAINING LINEAR CLASSIFIER ON 
                   QUANTUM EMBEDDINGS
        =========================================
        
        """)
    
    loss_qrc, accs_train_qrc, accs_test_qrc, model_qrc = train(
        embeddings, ys, test_embeddings, test_targets, 
        regularization=args.classifier_regularization,  # Use classifier regularization
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
        regularization=args.classifier_regularization,  # Use classifier regularization 
        nepochs=args.nepochs, 
        batchsize=args.batchsize, 
        learning_rate=args.learning_rate,
        verbose=not args.no_progress,
        nonlinear=True
    )
    results["NN"] = (loss_nn, accs_train_nn, accs_test_nn, model_nn)
    
    print(f"""
        ==========================================
            Dataset: {args.dataset_type}
            Reduction Method: {args.reduction_method.upper()}
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
    
    # Always make guided_autoencoder_losses accessible to parameter_sweep
    import __main__
    __main__.guided_autoencoder_losses = guided_autoencoder_losses
    
    # Always save statistics regardless of whether it's a parameter sweep or not
    # Convert args to dictionary for saving
    config_dict = vars(args)
    # Don't save again if we're in a parameter sweep - parameter_sweep will handle it
    if not hasattr(args, '_parameter_sweep') or not args._parameter_sweep:
        output_dir = save_all_statistics(results, guided_autoencoder_losses, config=config_dict)
        print(f"Saved run statistics to {output_dir}")
    
    return results

if __name__ == "__main__":
    args = get_args()
    main(args)
