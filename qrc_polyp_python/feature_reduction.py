import numpy as np
from typing import Dict, Tuple, Optional, Any, Union, List
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from data_processing import flatten_images

# Import autoencoder functionality
try:
    from autoencoder import Autoencoder, train_autoencoder, encode_data
    AUTOENCODER_AVAILABLE = True
except ImportError:
    AUTOENCODER_AVAILABLE = False

def apply_pca(data: Dict[str, Any], 
              dim_pca: int = 8, 
              num_examples: int = 1000) -> Tuple[np.ndarray, np.ndarray, PCA, float, OneHotEncoder]:
    """
    Apply PCA to the image features.
    
    Parameters
    ----------
    data : dict
        Dictionary containing 'features' (image data) and 'targets' (labels)
    dim_pca : int, optional
        Number of principal components to extract, by default 8
    num_examples : int, optional
        Number of examples to process, by default 1000
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, PCA, float, OneHotEncoder]
        - xs: Principal components for the selected examples
        - ys: One-hot encoded labels for the selected examples
        - pca: The fitted PCA model
        - spectral: Max absolute value of the PCA components (for scaling)
        - encoder: The fitted OneHotEncoder
    """
    # Flatten images
    print("Flattening images...")
    data_flat = flatten_images(data["features"])
    
    # Apply PCA
    print("Fitting PCA model...")
    with tqdm(total=100, desc="Fitting PCA model") as pbar:
        pca = PCA(n_components=dim_pca)
        pca.fit(data_flat.T)
        pbar.update(100)
    
    # Transform data
    print("Computing PCA...")
    transformed_data = []
    pca_iterator = tqdm(range(data_flat.shape[1]), desc="Computing PCA")
    for i in pca_iterator:
        transformed = pca.transform(data_flat[:, i].reshape(1, -1))
        transformed_data.append(transformed[0])
    x = np.array(transformed_data).T
    
    xs = x[:, :num_examples]  # Take first num_examples samples
    
    # Calculate spectral range (max absolute value)
    spectral = max(abs(xs.max()), abs(xs.min()))
    
    data_categories = data["metadata"]["n_classes"]
    if data_categories is None:
        raise ValueError("Metadata does not contain 'n_classes' key.")
        
    # One-hot encode the labels, ensuring the correct number of categories
    encoder = OneHotEncoder(categories=[np.arange(data_categories)], sparse_output=False)
    
    y = encoder.fit_transform(data["targets"].reshape(-1, 1)) 
    ys = y[:num_examples].T
    
    return xs, ys, pca, spectral, encoder

def apply_autoencoder(data: Dict[str, Any],
                    encoding_dim: int = 8,
                    num_examples: int = 1000,
                    hidden_dims: Optional[List[int]] = None,
                    batch_size: int = 64,
                    epochs: int = 50,
                    learning_rate: float = 0.001,
                    device: str = 'cpu',
                    verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, Autoencoder, float, OneHotEncoder]:
    """
    Apply autoencoder to reduce image dimensions.
    
    Parameters
    ----------
    data : dict
        Dictionary containing 'features' (image data) and 'targets' (labels)
    encoding_dim : int, optional
        Dimension of the encoded representation, by default 8
    num_examples : int, optional
        Number of examples to process, by default 1000
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
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Autoencoder, float, OneHotEncoder]
        - xs: Encoded features for the selected examples
        - ys: One-hot encoded labels for the selected examples
        - model: Trained autoencoder model
        - spectral: Max absolute value of the encoded features (for scaling)
        - encoder: The fitted OneHotEncoder
    """
    if not AUTOENCODER_AVAILABLE:
        raise ImportError("Autoencoder functionality requires the autoencoder module.")
    
    # Flatten images
    print("Flattening images...")
    data_flat = flatten_images(data["features"])
    
    # Train autoencoder
    print("Training autoencoder...")
    model, spectral = train_autoencoder(
        data_flat, encoding_dim, hidden_dims, 
        batch_size, epochs, learning_rate, device, verbose
    )
    
    # Encode data
    print("Encoding data...")
    encoded_data = encode_data(model, data_flat, device, verbose)
    
    # Take first num_examples samples
    xs = encoded_data[:, :num_examples]
    
    data_categories = data["metadata"]["n_classes"]
    if data_categories is None:
        raise ValueError("Metadata does not contain 'n_classes' key.")
        
    # One-hot encode the labels
    encoder = OneHotEncoder(categories=[np.arange(data_categories)], sparse_output=False)
    
    y = encoder.fit_transform(data["targets"].reshape(-1, 1))
    ys = y[:num_examples].T
    
    return xs, ys, model, spectral, encoder

def apply_pca_to_test_data(data: Dict[str, Any], pca_model: PCA, spectral: float, 
                           dim_pca: int, num_examples: int) -> np.ndarray:
    """
    Apply pre-trained PCA to test data.
    
    Parameters
    ----------
    data : dict
        Test dataset containing 'features'
    pca_model : PCA
        Trained PCA model
    spectral : float
        Scaling factor from training
    dim_pca : int
        Number of PCA components
    num_examples : int
        Number of samples to process
        
    Returns
    -------
    np.ndarray
        Transformed features (not scaled)
    """
    print("Flattening test data...")
    data_flat = flatten_images(data["features"], desc="Flattening test data")
    data_flat = data_flat.T  # Transpose for sklearn PCA
    
    print("Applying PCA to test data...")
    test_features = np.zeros((min(data_flat.shape[0], num_examples), dim_pca))
    for i in tqdm(range(min(data_flat.shape[0], num_examples)), desc="PCA transform"):
        test_features[i] = pca_model.transform(data_flat[i].reshape(1, -1))[0]
    
    # Return in the same format as training features
    test_features = test_features.T  # Match the format of training features
    
    return test_features

def apply_autoencoder_to_test_data(data: Dict[str, Any], 
                                  autoencoder_model: Autoencoder,
                                  num_examples: int,
                                  device: str = 'cpu',
                                  verbose: bool = True) -> np.ndarray:
    """
    Apply trained autoencoder to test data.
    
    Parameters
    ----------
    data : dict
        Test dataset containing 'features'
    autoencoder_model : Autoencoder
        Trained autoencoder model
    num_examples : int
        Number of samples to process
    device : str, optional
        Device to use ('cpu' or 'cuda'), by default 'cpu'
    verbose : bool, optional
        Whether to show progress bars, by default True
        
    Returns
    -------
    np.ndarray
        Encoded features
    """
    if not AUTOENCODER_AVAILABLE:
        raise ImportError("Autoencoder functionality requires the autoencoder module.")
    
    print("Flattening test data...")
    data_flat = flatten_images(data["features"], desc="Flattening test data")
    
    # Limit to the requested number of examples
    if data_flat.shape[1] > num_examples:
        data_flat = data_flat[:, :num_examples]
    
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
    return xs / spectral * detuning_max
