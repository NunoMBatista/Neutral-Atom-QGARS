from typing import Tuple, Dict, Any, List, Optional
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import random

# Global settings
SHOW_PROGRESS_BAR: bool = True

def create_polyp_dataset(
    polyp_dir: str, 
    no_polyp_dir: str, 
    split_ratio: float = 0.8, 
    target_size: Tuple[int, int] = (28, 28)
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load and preprocess the polyp dataset.
    
    Loads images from polyp and no-polyp directories, preprocesses them,
    and splits them into training and test sets.
    
    Parameters
    ----------
    polyp_dir : str
        Directory containing polyp images
    no_polyp_dir : str
        Directory containing non-polyp images
    split_ratio : float, optional
        Ratio for train/test split (default is 0.8)
    target_size : Tuple[int, int], optional
        Size to resize images to (default is (28, 28))
    
    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any]]
        A tuple containing:
        - train_dataset: Dictionary with training data
        - test_dataset: Dictionary with test data
    """
    # Load and process polyp images
    polyp_files = [os.path.join(polyp_dir, f) for f in os.listdir(polyp_dir)]
    no_polyp_files = [os.path.join(no_polyp_dir, f) for f in os.listdir(no_polyp_dir)]
    
    # Shuffle files
    random.shuffle(polyp_files)
    random.shuffle(no_polyp_files)
    
    # Define train/test split
    n_polyp_train = int(len(polyp_files) * split_ratio)
    n_no_polyp_train = int(len(no_polyp_files) * split_ratio)
    
    polyp_train = polyp_files[:n_polyp_train]
    polyp_test = polyp_files[n_polyp_train:]
    no_polyp_train = no_polyp_files[:n_no_polyp_train]
    no_polyp_test = no_polyp_files[n_no_polyp_train:]
    
    # Create train dataset
    train_files = polyp_train + no_polyp_train
    train_targets = [1] * len(polyp_train) + [0] * len(no_polyp_train)
    
    # Create test dataset
    test_files = polyp_test + no_polyp_test
    test_targets = [1] * len(polyp_test) + [0] * len(no_polyp_test)
    
    # Shuffle train and test data
    train_indices = list(range(len(train_files)))
    test_indices = list(range(len(test_files)))
    random.shuffle(train_indices)
    random.shuffle(test_indices)
    
    train_files = [train_files[i] for i in train_indices]
    train_targets = [train_targets[i] for i in train_indices]
    test_files = [test_files[i] for i in test_indices]
    test_targets = [test_targets[i] for i in test_indices]
    
    # Process images and create features arrays
    def process_images(files, target_size):
        n_samples = len(files)
        features = np.zeros((target_size[0], target_size[1], n_samples), dtype=np.float32)
        
        for i, file in enumerate(tqdm(files) if SHOW_PROGRESS_BAR else files):
            img = Image.open(file).convert('L')  # Convert to grayscale
            img_resized = img.resize(target_size)
            features[:, :, i] = np.array(img_resized) / 255.0  # Normalize to [0,1]
        
        return features
    
    # Process train and test images
    print("Processing training images...")
    train_features = process_images(train_files, target_size)
    print("Processing test images...")
    test_features = process_images(test_files, target_size)
    
    # Create metadata
    train_metadata = {
        "n_samples": len(train_files),
        "n_polyp": len(polyp_train),
        "n_no_polyp": len(no_polyp_train),
    }
    
    test_metadata = {
        "n_samples": len(test_files),
        "n_polyp": len(polyp_test),
        "n_no_polyp": len(no_polyp_test),
    }
    
    # Create dataset structs
    train_dataset = {
        "metadata": train_metadata,
        "split": "train",
        "features": train_features,
        "targets": np.array(train_targets)
    }
    
    test_dataset = {
        "metadata": test_metadata,
        "split": "test",
        "features": test_features,
        "targets": np.array(test_targets)
    }
    
    return train_dataset, test_dataset

def flatten_images(data: np.ndarray, desc: str = "Flattening images") -> np.ndarray:
    """
    Flatten 3D image data into 2D matrix.
    
    Converts images from (height, width, n_samples) to (height*width, n_samples)
    for further processing.
    
    Parameters
    ----------
    data : np.ndarray
        Image data tensor of shape (height, width, n_samples)
    desc : str, optional
        Description for progress bar (default is "Flattening images")
        
    Returns
    -------
    np.ndarray
        Flattened data matrix of shape (height*width, n_samples)
    """
    dataset_length = data.shape[2]
    image_size = data.shape[0] * data.shape[1]
    
    flat_iterator = tqdm(range(dataset_length), desc=desc)
    data_flat = np.zeros((image_size, dataset_length))
    for i in flat_iterator:
        data_flat[:, i] = data[:, :, i].flatten()
    
    return data_flat

def show_sample_image(data: Dict[str, Any], index: Optional[int] = None) -> int:
    """
    Display a sample image from the dataset.
    
    Shows an image from the dataset with its corresponding label.
    
    Parameters
    ----------
    data : Dict[str, Any]
        Dataset containing 'features' and 'targets'
    index : Optional[int], optional
        Index of image to display (random if None)
        
    Returns
    -------
    int
        Index of displayed image
    """
    import matplotlib.pyplot as plt
    
    if index is None:
        index = random.randint(0, data["features"].shape[2] - 1)
    
    img = data["features"][:, :, index]
    plt.imshow(img, cmap='gray')
    plt.title(f"Label: {data['targets'][index]}")
    plt.show()
    
    return index