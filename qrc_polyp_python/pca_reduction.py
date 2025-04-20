import numpy as np
from typing import Dict, Tuple, Optional, Any, Union
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from data_processing import flatten_images

def apply_pca(data: Dict[str, Any], dim_pca: int = 8, num_examples: int = 1000) -> Tuple[np.ndarray, np.ndarray, PCA, float, OneHotEncoder]:
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
    
    # One-hot encode the labels
    # Handle different scikit-learn versions
    try:
        # For newer scikit-learn versions (0.24+)
        encoder = OneHotEncoder(categories=[[0, 1]], sparse_output=False)
    except TypeError:
        # For older scikit-learn versions
        encoder = OneHotEncoder(categories=[[0, 1]], sparse=False)
    
    y = encoder.fit_transform(data["targets"].reshape(-1, 1))
    ys = y[:num_examples].T  # Transpose to match Julia's format
    
    return xs, ys, pca, spectral, encoder

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
