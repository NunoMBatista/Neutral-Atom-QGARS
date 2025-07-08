import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, List, Optional, Union, Callable
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Subset
    
# Global settings
SHOW_PROGRESS_BAR = True

class DatasetLoader:
    """
    Base class for dataset loaders.
    """
    def preprocess_image(self, image: Image.Image, target_size: Tuple[int, int], keep_rgb: bool = False) -> np.ndarray:
        """
        Preprocess a single image.
        
        Parameters
        ----------
        image : PIL.Image.Image
            The image to preprocess
        target_size : Tuple[int, int]
            Target size to resize image to
        keep_rgb : bool, optional
            Whether to keep RGB channels or convert to grayscale, by default False
            
        Returns
        -------
        np.ndarray
            Preprocessed image as numpy array
        """
        # Convert to grayscale if specified, otherwise maintain RGB
        if not keep_rgb and image.mode != 'L':
            image = image.convert('L')
        elif keep_rgb and image.mode != 'RGB':
            # Convert to RGB if it's not already
            image = image.convert('RGB')
            
        # Resize to target size
        img_resized = image.resize(target_size)
        
        # Convert to numpy array and normalize
        img_array = np.array(img_resized) / 255.0
        
        # Make sure single channel images have the right shape
        if not keep_rgb and len(img_array.shape) == 2:
            # Single channel images should have shape (height, width, 1)
            img_array = img_array[:, :, np.newaxis]
            
        return img_array

    def create_dataset_dict(self, features: np.ndarray, targets: np.ndarray, 
                           n_classes: int, split: str, **metadata) -> Dict[str, Any]:
        """
        Create a dictionary representation of a dataset.
        
        Parameters
        ----------
        features : np.ndarray
            Feature data with shape (height, width, n_samples) for images
        targets : np.ndarray
            Target labels
        n_classes : int
            Number of classes
        split : str
            Dataset split ('train' or 'test')
        **metadata
            Additional metadata to include
            
        Returns
        -------
        Dict[str, Any]
            Dataset dictionary
        """
        # Create metadata dictionary
        #print("\n\n\n\n" + str(features.shape) + "\n\n\n\n")
        
        meta_dict = {
            "n_samples": features.shape[-1],
            "n_classes": n_classes
        }
        
        # Add additional metadata
        meta_dict.update(metadata)
        
        # Create and return dataset dictionary
        return {
            "metadata": meta_dict,
            "split": split,
            "features": features,
            "targets": targets
        }

    def process_images(self, files: List[str], target_size: Tuple[int, int], keep_rgb: bool = False) -> np.ndarray:
        """
        Process images and resize them to the target size.
        
        Parameters
        ----------
        files : List[str]
            List of file paths to images
        target_size : Tuple[int, int]
            Target size to resize images to
        keep_rgb : bool, optional
            Whether to keep RGB channels, by default False
    
        Returns
        -------
        np.ndarray
            Array of processed images
        """
        n_samples = len(files)
    
        # Determine channels - 3 for RGB, 1 for grayscale
        channels = 3 if keep_rgb else 1
    
        # Initialize features array with correct shape (height, width, channels, n_samples)
        features = np.zeros((target_size[0], target_size[1], channels, n_samples), dtype=np.float32)
    
        for i, file in enumerate(tqdm(files, desc="Processing images") if SHOW_PROGRESS_BAR else files):
            try:
                img = Image.open(file)
                img_array = self.preprocess_image(img, target_size, keep_rgb)
            
                # Check if the image has the expected number of channels
                if img_array.shape[2] != channels:
                    print(f"Warning: Image {file} has unexpected channel count {img_array.shape[2]}, expected {channels}")
                    # Convert if needed
                    if channels == 1 and img_array.shape[2] == 3:
                        # Convert RGB to grayscale
                        img_array = np.mean(img_array, axis=2, keepdims=True)
                    elif channels == 3 and img_array.shape[2] == 1:
                        # Expand grayscale to RGB
                        img_array = np.repeat(img_array, 3, axis=2)
            
                features[:, :, :, i] = img_array
            except Exception as e:
                print(f"Error processing image {file}: {e}")
                # If image fails to load, use zeros
                features[:, :, :, i] = np.zeros((target_size[0], target_size[1], channels), dtype=np.float32)
                
        return features

    def load_image_folder_dataset(self, data_dir: str, 
                                 target_size: Tuple[int, int] = (28, 28),
                                 split_ratio: float = 0.8,
                                 num_examples: Optional[int] = None,
                                 num_test_examples: Optional[int] = None,
                                 keep_rgb: bool = False,
                                 **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load a dataset from a directory with class subfolders.
        
        Parameters
        ----------
        data_dir : str
            Directory containing class subfolders
        target_size : Tuple[int, int], optional
            Size to resize images to, by default (28, 28)
        split_ratio : float, optional
            Ratio for train/test split, by default 0.8
        num_examples : Optional[int], optional
            Maximum number of training examples to load, by default None (load all)
        num_test_examples : Optional[int], optional
            Maximum number of test examples to load, by default None (load all)
        keep_rgb : bool, optional
            Whether to keep RGB channels, by default False
            
        Returns
        -------
        Tuple[Dict[str, Any], Dict[str, Any]]
            train_dataset, test_dataset containing features and targets
        """
        # Get class folders (directories only)
        class_folders = [d for d in os.listdir(data_dir) 
                         if os.path.isdir(os.path.join(data_dir, d))]
        class_folders.sort()  # Sort to ensure consistent class indices
        
        # Map class names to indices
        class_to_idx = {cls_name: i for i, cls_name in enumerate(class_folders)}
        n_classes = len(class_folders)
        
        # Collect all files with their classes
        files = []
        targets = []
        
        for class_name in class_folders:
            class_dir = os.path.join(data_dir, class_name)
            class_idx = class_to_idx[class_name]
            
            # Get all image files in this class folder
            class_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            files.extend(class_files)
            targets.extend([class_idx] * len(class_files))
        
        # Create train/test split
        indices = list(range(len(files)))
        random.shuffle(indices)
        
        split = int(len(files) * split_ratio)
        train_indices = indices[:split]
        test_indices = indices[split:]
        
        # If num_examples or num_test_examples specified, only use that many samples
        if num_examples is not None and num_examples < len(train_indices):
            train_indices = train_indices[:num_examples]
            print(f"Using {num_examples} training examples out of {split} available")
        
        if num_test_examples is not None and num_test_examples < len(test_indices):
            test_indices = test_indices[:num_test_examples]
            print(f"Using {num_test_examples} test examples out of {len(indices) - split} available")
        
        train_files = [files[i] for i in train_indices]
        train_targets = [targets[i] for i in train_indices]
        test_files = [files[i] for i in test_indices]
        test_targets = [targets[i] for i in test_indices]
        
        # Process images
        train_features = self.process_images(train_files, target_size, keep_rgb)
        test_features = self.process_images(test_files, target_size, keep_rgb)
        
        # Count samples per class
        train_class_counts = {}
        test_class_counts = {}
        
        for cls_idx in range(n_classes):
            train_class_counts[class_folders[cls_idx]] = train_targets.count(cls_idx)
            test_class_counts[class_folders[cls_idx]] = test_targets.count(cls_idx)
        
        # Create dataset dictionaries
        train_dataset = self.create_dataset_dict(
            features=train_features, 
            targets=np.array(train_targets), 
            n_classes=n_classes, 
            split="train",
            class_names=class_folders,
            class_counts=train_class_counts
        )
        
        test_dataset = self.create_dataset_dict(
            features=test_features, 
            targets=np.array(test_targets), 
            n_classes=n_classes, 
            split="test",
            class_names=class_folders,
            class_counts=test_class_counts
        )
        
        return train_dataset, test_dataset

    def load_mnist_dataset(self, data_dir: str = './data/datasets',
                          mnist_type: str = 'mnist',
                          target_size: Tuple[int, int] = (28, 28),
                          num_examples: Optional[int] = None,
                          num_test_examples: Optional[int] = None,
                          **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load and preprocess the MNIST dataset.
        
        Parameters
        ----------
        data_dir : str, optional
            Directory to store the MNIST dataset, by default './data'
        target_size : Tuple[int, int], optional
            Size to resize images to, by default (28, 28)
        
        Returns
        -------
        Tuple[Dict[str, Any], Dict[str, Any]]
            train_dataset, test_dataset containing features and 3targets
        """
        try:
            # Import torchvision here to keep it as an optional dependency
            import torchvision
            import torchvision.transforms as transforms
        except ImportError:
            raise ImportError("torchvision is required to load MNIST dataset. "
                             "Install it with 'pip install torchvision'.")
        
        # Define transformations
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])

        dataset = None
        is_binary = False
        n_classes = 10
        
        if mnist_type == 'mnist':
            print("Loading MNIST dataset...")
            dataset = torchvision.datasets.MNIST
        elif mnist_type == 'binary_mnist':
            print("Loading Binary MNIST dataset (only classes 0 and 1)...")
            dataset = torchvision.datasets.MNIST
            is_binary = True
            n_classes = 2
        elif mnist_type == 'fashion_mnist':
            print("Loading Fashion MNIST dataset...")
            dataset = torchvision.datasets.FashionMNIST
        
        if dataset is None:
            raise ValueError(f"Unknown MNIST type: {mnist_type}. "
                             "Available types: 'mnist', 'binary_mnist', 'fashion_mnist'")
        
        # Load MNIST dataset
        full_train_data = dataset(
            root=data_dir,
            train=True,
            download=True,
            transform=transform
        )
        
        full_test_data = dataset(
            root=data_dir,
            train=False,
            download=True,
            transform=transform
        )
        
        # Filter for binary MNIST if needed
        if is_binary:
            
            # Get indices of classes 0 and 1
            train_indices = [i for i, (_, label) in enumerate(full_train_data) if label in [0, 1]]
            test_indices = [i for i, (_, label) in enumerate(full_test_data) if label in [0, 1]]
            
            # Create subsets with only classes 0 and 1
            full_train_data = Subset(full_train_data, train_indices)
            full_test_data = Subset(full_test_data, test_indices)
            
            print(f"Filtered to {len(full_train_data)} training examples and {len(full_test_data)} test examples with classes 0 and 1")
        
        # Only using a subset of the data if specified
        if num_examples is not None and num_examples < len(full_train_data):
            train_indices = random.sample(range(len(full_train_data)), num_examples)
            train_data = Subset(full_train_data, train_indices)
            print(f"Using {num_examples} training examples out of {len(full_train_data)} available")
        else:
            train_data = full_train_data
        
        if num_test_examples is not None and num_test_examples < len(full_test_data):
            test_indices = random.sample(range(len(full_test_data)), num_test_examples)
            test_data = Subset(full_test_data, test_indices)
            print(f"Using {num_test_examples} test examples out of {len(full_test_data)} available")
        else:
            test_data = full_test_data    
    
        
        # Convert to numpy arrays
        train_features = np.zeros((target_size[0], target_size[1], len(train_data)), dtype=np.float32)
        train_targets = np.zeros(len(train_data), dtype=np.int64)
        
        print("Processing training images...")
        for i, (img, target) in enumerate(tqdm(train_data) if SHOW_PROGRESS_BAR else train_data):
            # Convert img from tensor (1, H, W) to numpy array (H, W)
            img_np = img.squeeze().numpy()
            train_features[:, :, i] = img_np
            train_targets[i] = target
        
        test_features = np.zeros((target_size[0], target_size[1], len(test_data)), dtype=np.float32)
        test_targets = np.zeros(len(test_data), dtype=np.int64)
        
        print("Processing test images...")
        for i, (img, target) in enumerate(tqdm(test_data) if SHOW_PROGRESS_BAR else test_data):
            img_np = img.squeeze().numpy()
            test_features[:, :, i] = img_np
            test_targets[i] = target
        
        # Create dataset dictionaries with appropriate metadata
        train_dataset = self.create_dataset_dict(
            train_features, train_targets, 
            n_classes=n_classes, split="train"
        )
        
        test_dataset = self.create_dataset_dict(
            test_features, test_targets, 
            n_classes=n_classes, split="test"
        )
        
        return train_dataset, test_dataset

# Create a single loader instance
_dataset_loader = DatasetLoader()

def load_dataset(name: str = 'image_folder', **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load a dataset by name.
    
    Parameters
    ----------
    name : str, optional
        Name of the dataset to load, by default 'image_folder'
    **kwargs
        Additional arguments to pass to the dataset loader
        
    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any]]
        train_dataset, test_dataset
    """
    if name in ['mnist', 'fashion_mnist', 'binary_mnist']:
        return _dataset_loader.load_mnist_dataset(mnist_type=name, **kwargs)
    elif name == 'image_folder':
        return _dataset_loader.load_image_folder_dataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {name}. Available types: 'image_folder', 'mnist', 'binary_mnist', 'fashion_mnist'")

# Utility functions
def flatten_images(data: np.ndarray, desc: str = "Flattening images") -> np.ndarray:
    """
    Flatten 3D or 4D image data into 2D matrix, dynamically handling different image sizes.
    
    Parameters
    ----------
    data : np.ndarray
        Image data tensor, can be:
        - (height, width, n_samples) for grayscale
        - (height, width, channels, n_samples) for RGB
    desc : str, optional
        Description for progress bar, by default "Flattening images"
        
    Returns
    -------
    np.ndarray
        Flattened data matrix of shape (flattened_image_size, n_samples)
    """
    # Determine number of samples (always the last dimension)
    n_samples = data.shape[-1]
    
    # Calculate the flattened size (product of all dimensions except the last one)
    flattened_size = int(np.prod(data.shape[:-1]))
    
    print(f"Flattening images with shape {data.shape} to size ({flattened_size}, {n_samples})")
    
    # Create the output array with the correct dimensions
    data_flat = np.zeros((flattened_size, n_samples))
    
    # Process each sample
    flat_iterator = tqdm(range(n_samples), desc=desc)
    for i in flat_iterator:
        if len(data.shape) == 3:  # Grayscale: (H, W, N)
            data_flat[:, i] = data[:, :, i].flatten()
        elif len(data.shape) == 4:  # RGB: (H, W, C, N)
            data_flat[:, i] = data[:, :, :, i].flatten()
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}. Expected 3D or 4D array.")
    
    return data_flat

def show_sample_image(data: Dict[str, Any], index: Optional[int] = None) -> int:
    """
    Display a sample image from the dataset.
    
    Parameters
    ----------
    data : Dict[str, Any]
        Dataset containing 'features' and 'targets'
    index : Optional[int], optional
        Index of image to display (random if None), by default None
    
    Returns
    -------
    int
        Index of the displayed image
    """
    if index is None:
        index = random.randint(0, data["features"].shape[2] - 1)
    
    img = data["features"][:, :, index]
    plt.imshow(img, cmap='gray')
    
    # Get class name if available
    if 'class_names' in data['metadata'] and isinstance(data['targets'][index], (int, np.integer)):
        class_idx = data['targets'][index]
        class_name = data['metadata']['class_names'][class_idx]
        plt.title(f"Class: {class_name} (Label: {class_idx})")
    else:
        plt.title(f"Label: {data['targets'][index]}")
    
    plt.show()
    
    return index


def one_hot_encode(targets: np.ndarray, n_classes: int) -> Tuple[np.ndarray, OneHotEncoder]:
    """
    One-hot encode the target labels.
    
    Parameters
    ----------
    targets : np.ndarray
        Target labels
    n_classes : int
        Number of classes
        
    Returns
    -------
    Tuple[np.ndarray, OneHotEncoder]
        One-hot encoded targets and the encoder object
    """

    # Create encoder ensuring the correct number of categories
    encoder = OneHotEncoder(categories=[np.arange(n_classes)], sparse_output=False)
    
    # Reshape targets to required 2D array and fit_transform
    encoded_targets = encoder.fit_transform(targets.reshape(-1, 1))
    
    print(f"Encoded {len(targets)} targets into {n_classes} classes")
    
    return encoded_targets, encoder

def select_random_samples(data: Dict[str, Any], num_samples: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select random samples from a dataset.
    
    Parameters
    ----------
    data : Dict[str, Any]
        Dataset containing 'features' and 'targets'
    num_samples : int
        Number of samples to select
    seed : Optional[int], optional
        Random seed for reproducibility, by default None
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        selected_indices, selected_features, selected_targets
    """
    # Get total number of samples
    n_samples = data["features"].shape[2] if len(data["features"].shape) > 2 else len(data["features"])
    
    # Cap num_samples to available samples
    num_samples = min(num_samples, n_samples)
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Select random indices
    selected_indices = np.random.choice(n_samples, size=num_samples, replace=False)
    
    # Get corresponding features
    if len(data["features"].shape) > 2:
        # For image data with shape (height, width, n_samples)
        selected_features = data["features"][:, :, selected_indices]
    else:
        # For flattened data
        selected_features = data["features"][selected_indices]
    
    # Get corresponding targets
    selected_targets = data["targets"][selected_indices]
    
    return selected_indices, selected_features, selected_targets
