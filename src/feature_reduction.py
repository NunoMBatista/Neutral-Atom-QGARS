import numpy as np
from typing import Dict, Tuple, Optional, Any, Union, List, TYPE_CHECKING
from sklearn.decomposition import PCA
from tqdm import tqdm
from data_processing import flatten_images

from autoencoder import Autoencoder, train_autoencoder, encode_data
from guided_autoencoder import GuidedAutoencoder, train_guided_autoencoder, encode_data_guided


def apply_pca(data: Dict[str, Any], 
              dim_pca: int = 8, 
              selected_indices: Optional[np.ndarray] = None,
              selected_features: Optional[np.ndarray] = None) -> Tuple[np.ndarray, PCA, float]:
    """
    Apply PCA to the image features.
    
    Parameters
    ----------
    data : dict
        Dictionary containing 'features' (image data) and 'targets' (labels)
    dim_pca : int, optional
        Number of principal components to extract, by default 8
    selected_indices : Optional[np.ndarray], optional
        Indices of selected samples, by default None
    selected_features : Optional[np.ndarray], optional
        Pre-selected features to use, by default None
    
    Returns
    -------
    Tuple[np.ndarray, PCA, float]
        - xs: Principal components for the selected examples
        - pca: The fitted PCA model
        - spectral: Max absolute value of the PCA components (for scaling)
    """
    # Use provided features if available, otherwise use all data
    if selected_features is not None:
        print(f"Using {selected_features.shape[2]} pre-selected samples for PCA")
        data_to_use = selected_features
    else:
        # If no pre-selection, use full dataset
        data_to_use = data["features"]
        
    # Flatten images
    print("Flattening images...")
    data_flat = flatten_images(data_to_use)
    
    # Apply PCA
    print("Fitting PCA model...")
    with tqdm(total=1, desc="Fitting PCA model") as pbar:
        pca = PCA(n_components=dim_pca)
        pca.fit(data_flat.T)
        pbar.update(1)
    
    # Transform data
    print("Computing PCA...")
    with tqdm(total=data_flat.shape[1], desc=f"Mapping {data_flat.shape[1]} images to Principal Components") as pbar:
        tranformed_data = pca.transform(data_flat.T)
        x = tranformed_data.T
        pbar.update(data_flat.shape[1])
    
    # We don't need to select a subset now, as we already have the selected samples
    xs = x
    
    # Calculate spectral range (max absolute value)
    spectral = max(abs(xs.max()), abs(xs.min()))
    
    return xs, pca, spectral


def apply_autoencoder(data: Dict[str, Any],
                    encoding_dim: int = 8,
                    hidden_dims: Optional[List[int]] = None,
                    batch_size: int = 64,
                    epochs: int = 50,
                    learning_rate: float = 0.001,
                    device: str = 'cpu',
                    verbose: bool = True,
                    use_batch_norm: bool = True,
                    dropout: float = 0.1,
                    weight_decay: float = 1e-5,
                    autoencoder_regularization: Optional[float] = None,
                    selected_indices: Optional[np.ndarray] = None,
                    selected_features: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Autoencoder, float]:
    """
    Apply improved autoencoder to reduce image dimensions.
    
    Parameters
    ----------
    data : dict
        Dictionary containing 'features' (image data) and 'targets' (labels)
    encoding_dim : int, optional
        Dimension of the encoded representation, by default 8
    hidden_dims : Optional[List[int]], optional
        Dimensions of hidden layers, by default None
    batch_size : int, optional
        Batch size for autoencoder training, by default 64
    epochs : int, optional
        Number of training epochs, by default 50
    learning_rate : float, optional
        Learning rate for training, by default 0.001
    device : str, optional
        Device to use ('cpu' or 'cuda'), by default 'cpu'
    verbose : bool, optional
        Whether to show progress bars, by default True
    use_batch_norm : bool, optional
        Whether to use batch normalization, by default True
    dropout : float, optional
        Dropout probability, by default 0.1
    weight_decay : float, optional
        Weight decay for regularization, by default 1e-5
    autoencoder_regularization : Optional[float], optional
        Regularization parameter for autoencoder, by default None
    selected_indices : Optional[np.ndarray], optional
        Indices of selected samples, by default None
    selected_features : Optional[np.ndarray], optional
        Pre-selected features to use, by default None
        
    Returns
    -------
    Tuple[np.ndarray, Autoencoder, float]
        - xs: Encoded features for the selected examples
        - model: Trained autoencoder model
        - spectral: Max absolute value of the encoded features (for scaling)
    """
    
    # Use provided features if available, otherwise use all data
    if selected_features is not None:
        print(f"Using {selected_features.shape[2]} pre-selected samples for autoencoder")
        data_to_use = selected_features
    else:
        # If no pre-selection, use full dataset
        data_to_use = data["features"]
    
    # Flatten images
    print("Flattening images...")
    data_flat = flatten_images(data_to_use)
    
    # Train autoencoder
    print("Training autoencoder...")
    model, spectral = train_autoencoder(
        data=data_flat, 
        encoding_dim=encoding_dim, 
        hidden_dims=hidden_dims, 
        batch_size=batch_size, 
        epochs=epochs, 
        learning_rate=learning_rate, 
        device=device, 
        verbose=verbose, 
        use_batch_norm=use_batch_norm, 
        dropout=dropout, 
        #weight_decay=weight_decay,
        autoencoder_regularization=autoencoder_regularization
    )
    
    # Encode data
    print("Encoding data...")
    encoded_data = encode_data(
                            model=model,
                            data=data_flat, 
                            device=device, 
                            verbose=verbose
                        )
    
    # Verify encoded data has meaningful values
    if np.all(encoded_data == 0):
        print("WARNING: Encoded data contains all zeros!")
    elif np.allclose(encoded_data, 0, atol=1e-5):
        print("WARNING: Encoded data is very close to zero. Features may not be discriminative.")
    
    # No need to select a subset now, as we already have the selected samples
    xs = encoded_data
    
    return xs, model, spectral


def apply_pca_to_test_data(data: Dict[str, Any], 
                           pca_model: PCA, 
                           dim_pca: int, 
                           selected_indices: Optional[np.ndarray] = None,
                           selected_features: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Apply pre-trained PCA to test data.
    
    Parameters
    ----------
    data : dict
        Test dataset containing 'features'
    pca_model : PCA
        Trained PCA model
    dim_pca : int
        Number of PCA components
    selected_indices : Optional[np.ndarray], optional
        Indices of selected samples, by default None
    selected_features : Optional[np.ndarray], optional
        Pre-selected features to use, by default None
        
    Returns
    -------
    np.ndarray
        Transformed features (not scaled)
    """
    # Use provided features if available, otherwise use all data
    if selected_features is not None:
        print(f"Using {selected_features.shape[2]} pre-selected test samples for PCA")
        data_to_use = selected_features
    else:
        # If no pre-selection, use full dataset
        data_to_use = data["features"]
        
    print("Flattening test data...")
    data_flat = flatten_images(data_to_use, desc="Flattening test data")
    data_flat = data_flat.T  # Transpose for sklearn PCA
    
    print("Applying PCA to test data...")
    test_features = np.zeros((data_flat.shape[0], dim_pca))
    for i in tqdm(range(data_flat.shape[0]), desc="PCA transform"):
        test_features[i] = pca_model.transform(data_flat[i].reshape(1, -1))[0]
    
    # Return in the same format as training features 
    test_features = test_features.T  # Match the format of training features
    
    return test_features


def apply_autoencoder_to_test_data(data: Dict[str, Any], 
                                  autoencoder_model: Autoencoder,
                                  device: str = 'cpu',
                                  verbose: bool = True,
                                  selected_indices: Optional[np.ndarray] = None,
                                  selected_features: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Apply trained autoencoder to test data.
    
    Parameters
    ----------
    data : dict
        Test dataset containing 'features'
    autoencoder_model : Autoencoder
        Trained autoencoder model
    device : str, optional
        Device to use ('cpu' or 'cuda'), by default 'cpu'
    verbose : bool, optional
        Whether to show progress bars, by default True
    selected_indices : Optional[np.ndarray], optional
        Indices of selected samples, by default None
    selected_features : Optional[np.ndarray], optional
        Pre-selected features to use, by default None
        
    Returns
    -------
    np.ndarray
        Encoded features
    """
    
    # Use provided features if available, otherwise use all data
    if selected_features is not None:
        print(f"Using {selected_features.shape[2]} pre-selected test samples for autoencoder")
        data_to_use = selected_features
    else:
        # If no pre-selection, use full dataset
        data_to_use = data["features"]
    
    print("Flattening test data...")
    data_flat = flatten_images(data_to_use, desc="Flattening test data")
    
    # Encode data
    print("Encoding test data...")
    encoded_features = encode_data(
        autoencoder_model, data_flat, device, verbose
    )
    
    return encoded_features


def scale_to_detuning_range(xs: np.ndarray, spectral: float, detuning_max: float = 6.0) -> np.ndarray:
    """
    Scale data to a specific detuning range.
    
    Parameters
    ----------
    xs : np.ndarray
        Input data to scale
    spectral : float
        The spectral range (maximum absolute value) from the original data
    detuning_max : float, optional
        Maximum detuning value to scale to, by default 6.0
    
    Returns
    -------
    np.ndarray
        Scaled data within the range (-detuning_max, detuning_max)
    """
    # Safety check to avoid division by zero
    if spectral == 0 or np.isclose(spectral, 0):
        print("WARNING: Spectral value is close to zero, using default scaling.")
        return xs
    
    scaled_data = xs / spectral * detuning_max
    
    # Verify scaled data
    if np.all(scaled_data == 0):
        print("WARNING: Scaled data contains all zeros!")
    
    return scaled_data


def apply_guided_autoencoder(data: Dict[str, Any],
                            quantum_layer,
                            encoding_dim: int = 8,
                            hidden_dims: Optional[List[int]] = None,
                            guided_lambda: float = 0.3,
                            batch_size: int = 32,
                            epochs: int = 50,
                            learning_rate: float = 0.001,
                            quantum_update_frequency: int = 5,
                            n_shots: int = 1000,
                            device: str = 'cpu',
                            verbose: bool = True,
                            use_batch_norm: bool = True,
                            dropout: float = 0.1,
                            weight_decay: float = 1e-5,
                            autoencoder_regularization: Optional[float] = None,
                            selected_indices: Optional[np.ndarray] = None,
                            selected_features: Optional[np.ndarray] = None,
                            selected_targets: Optional[np.ndarray] = None) -> Tuple[np.ndarray, GuidedAutoencoder, float, Dict[str, List[float]]]:
    """
    Apply guided autoencoder to reduce image dimensions with quantum guidance.
    
    Parameters
    ----------
    data : dict
        Dictionary containing 'features' (image data) and 'targets' (labels)
    quantum_layer
        Quantum reservoir layer for generating embeddings
    encoding_dim : int, optional
        Dimension of the encoded representation, by default 8
    hidden_dims : Optional[List[int]], optional
        Dimensions of hidden layers, by default None
    guided_lambda : float, optional
        Weight for classification loss (0-1), by default 0.3
        Loss = (1-lambda)*reconstruction_loss + lambda*classification_loss
    batch_size : int, optional
        Batch size for autoencoder training, by default 32
    epochs : int, optional
        Number of training epochs, by default 50
    learning_rate : float, optional
        Learning rate for training, by default 0.001
    quantum_update_frequency : int, optional
        Update quantum embeddings every N epochs, by default 5
    n_shots : int, optional
        Number of shots for quantum simulation, by default 1000
    device : str, optional
        Device to use ('cpu' or 'cuda'), by default 'cpu'
    verbose : bool, optional
        Whether to show progress bars, by default True
    use_batch_norm : bool, optional
        Whether to use batch normalization, by default True
    dropout : float, optional
        Dropout probability, by default 0.1
    weight_decay : float, optional
        Weight decay for regularization, by default 1e-5
    autoencoder_regularization : Optional[float], optional
        Regularization parameter for guided autoencoder, by default None
    selected_indices : Optional[np.ndarray], optional
        Indices of selected samples, by default None
    selected_features : Optional[np.ndarray], optional
        Pre-selected features to use, by default None
    selected_targets : Optional[np.ndarray], optional
        Pre-selected targets to use, by default None
        
    Returns
    -------
    Tuple[np.ndarray, GuidedAutoencoder, float, Dict[str, List[float]]]
        - xs: Encoded features for the selected examples
        - model: Trained guided autoencoder model
        - spectral: Max absolute value of the encoded features (for scaling)
        - loss_history: Dictionary containing loss history
    """
    
    # Use provided features if available, otherwise use all data
    if selected_features is not None and selected_targets is not None:
        print(f"Using {selected_features.shape[2]} pre-selected samples for guided autoencoder")
        data_to_use = selected_features
        targets = selected_targets
    else:
        # If no pre-selection, use full dataset
        data_to_use = data["features"]
        targets = data["targets"]
    
    # Flatten images
    print("Flattening images...")
    data_flat = flatten_images(data_to_use)
    
    # Train guided autoencoder
    print("Training guided autoencoder with quantum feedback...")
    model, spectral, loss_history = train_guided_autoencoder(
        data=data_flat, 
        labels=targets, 
        quantum_layer=quantum_layer, 
        encoding_dim=encoding_dim, 
        hidden_dims=hidden_dims, 
        guided_lambda=guided_lambda, 
        batch_size=batch_size, 
        epochs=epochs, 
        learning_rate=learning_rate,
        device=device, 
        quantum_update_frequency=quantum_update_frequency, 
        n_shots=n_shots, 
        verbose=verbose,
        use_batch_norm=use_batch_norm, 
        dropout=dropout, 
        autoencoder_regularization=autoencoder_regularization,
    )
    
    # Encode data
    print("Encoding data with guided autoencoder...")
    encoded_data = encode_data_guided(model, data_flat, device, verbose)
    
    # Verify encoded data has meaningful values
    if np.all(encoded_data == 0):
        print("WARNING: Encoded data contains all zeros!")
    elif np.allclose(encoded_data, 0, atol=1e-5):
        print("WARNING: Encoded data is very close to zero. Features may not be discriminative.")
    
    # No need to select a subset now, as we already have the selected samples
    xs = encoded_data
    
    return xs, model, spectral, loss_history


def apply_guided_autoencoder_to_test_data(data: Dict[str, Any], 
                                         guided_autoencoder_model: GuidedAutoencoder,
                                         device: str = 'cpu',
                                         verbose: bool = True,
                                         selected_indices: Optional[np.ndarray] = None,
                                         selected_features: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Apply trained guided autoencoder to test data.
    
    Parameters
    ----------
    data : dict
        Test dataset containing 'features'
    guided_autoencoder_model : GuidedAutoencoder
        Trained guided autoencoder model
    device : str, optional
        Device to use ('cpu' or 'cuda'), by default 'cpu'
    verbose : bool, optional
        Whether to show progress bars, by default True
    selected_indices : Optional[np.ndarray], optional
        Indices of selected samples, by default None
    selected_features : Optional[np.ndarray], optional
        Pre-selected features to use, by default None
        
    Returns
    -------
    np.ndarray
        Encoded features
    """
    
    # Use provided features if available, otherwise use all data
    if selected_features is not None:
        print(f"Using {selected_features.shape[2]} pre-selected test samples for guided autoencoder")
        data_to_use = selected_features
    else:
        # If no pre-selection, use full dataset
        data_to_use = data["features"]
    
    print("Flattening test data...")
    data_flat = flatten_images(data_to_use, desc="Flattening test data")
    
    # Encode data
    print("Encoding test data with guided autoencoder...")
    encoded_features = encode_data_guided(
        guided_autoencoder_model, data_flat, device, verbose
    )
    
    return encoded_features
