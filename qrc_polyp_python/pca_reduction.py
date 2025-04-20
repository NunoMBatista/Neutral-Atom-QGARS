from typing import Dict, Any, Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from data_processing import flatten_images

def apply_pca(
    data: Dict[str, Any], 
    dim_pca: int = 8, 
    num_examples: int = 1000
) -> Tuple[np.ndarray, np.ndarray, PCA, OneHotEncoder]:
    """
    Apply PCA to the image features.
    
    Reduces dimensionality of image data using PCA without scaling.
    
    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary containing dataset with 'features' and 'targets'
    dim_pca : int, optional
        Number of PCA components (default is 8)
    num_examples : int, optional
        Number of examples to process (default is 1000)
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, PCA, OneHotEncoder]
        A tuple containing:
        - xs: Transformed features (without scaling)
        - ys: One-hot encoded labels
        - pca: Trained PCA model
        - encoder: Trained OneHotEncoder
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
    
    return xs, ys, pca, encoder

def scale_features(features: np.ndarray, target_range: float = 6.0) -> Tuple[np.ndarray, float]:
    """
    Scale features to a target range.
    
    Scales features to the range [-target_range, target_range] based on the
    maximum absolute value in the feature set.
    
    Parameters
    ----------
    features : np.ndarray
        Features to scale
    target_range : float, optional
        Target range for scaling (default is 6.0 for quantum detuning)
        
    Returns
    -------
    Tuple[np.ndarray, float]
        A tuple containing:
        - scaled_features: The scaled features
        - spectral: The scaling factor used
    """
    spectral = max(abs(features.max()), abs(features.min()))
    if spectral == 0:  # Avoid division by zero
        return features, 1.0
    scaled_features = features / spectral * target_range
    return scaled_features, spectral

def apply_pca_to_test_data(
    data: Dict[str, Any], 
    pca_model: PCA, 
    dim_pca: int, 
    num_examples: int
) -> np.ndarray:
    """
    Apply pre-trained PCA to test data.
    
    Transforms test data using a pre-trained PCA model without scaling.
    
    Parameters
    ----------
    data : Dict[str, Any]
        Test dataset containing 'features'
    pca_model : PCA
        Trained PCA model
    dim_pca : int
        Number of PCA components
    num_examples : int
        Number of samples to process
        
    Returns
    -------
    np.ndarray
        Transformed features (without scaling)
    """
    print("Flattening test data...")
    data_flat = flatten_images(data["features"], desc="Flattening test data")
    data_flat = data_flat.T  # Transpose for sklearn PCA
    
    print("Applying PCA to test data...")
    test_features = np.zeros((min(data_flat.shape[0], num_examples), dim_pca))
    for i in tqdm(range(min(data_flat.shape[0], num_examples)), desc="PCA transform"):
        test_features[i] = pca_model.transform(data_flat[i].reshape(1, -1))[0]
    
    # Transpose to match the format of training features
    test_features = test_features.T  
    
    return test_features
