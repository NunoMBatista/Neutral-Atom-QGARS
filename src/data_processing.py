import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, List, Optional, Union, Callable
from sklearn.preprocessing import OneHotEncoder
    
# Global settings
SHOW_PROGRESS_BAR = True

class DatasetLoader:
    """
    Base class for dataset loaders.
    """
    def preprocess_image(self, image: Image.Image, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Preprocess a single image.
        
        Parameters
        ----------
        image : PIL.Image.Image
            The image to preprocess
        target_size : Tuple[int, int]
            Target size to resize image to
            
        Returns
        -------
        np.ndarray
            Preprocessed image as numpy array
        """
        # Convert to grayscale if it's not already
        if image.mode != 'L':
            image = image.convert('L')
            
        # Resize to target size
        img_resized = image.resize(target_size)
        
        # Convert to numpy array and normalize
        return np.array(img_resized) / 255.0

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
        meta_dict = {
            "n_samples": features.shape[2] if len(features.shape) > 2 else len(features),
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

    def process_images(self, files: List[str], target_size: Tuple[int, int]) -> np.ndarray:
        """
        Process images and resize them to the target size.
        
        Parameters
        ----------
        files : List[str]
            List of file paths to images
        target_size : Tuple[int, int]
            Target size to resize images to
        
        Returns
        -------
        np.ndarray
            Array of processed images
        """
        n_samples = len(files)
        features = np.zeros((target_size[0], target_size[1], n_samples), dtype=np.float32)
        
        for i, file in enumerate(tqdm(files, desc="Processing images") if SHOW_PROGRESS_BAR else files):
            try:
                img = Image.open(file)
                features[:, :, i] = self.preprocess_image(img, target_size)
            except Exception as e:
                print(f"Error processing image {file}: {e}")
                # If image fails to load, use zeros
                features[:, :, i] = np.zeros(target_size, dtype=np.float32)
                
        return features

    def load_image_folder_dataset(self, data_dir: str, 
                                 target_size: Tuple[int, int] = (28, 28),
                                 split_ratio: float = 0.8,
                                 num_examples: Optional[int] = None,
                                 num_test_examples: Optional[int] = None,
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
        train_features = self.process_images(train_files, target_size)
        test_features = self.process_images(test_files, target_size)
        
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

    def load_mnist_dataset(self, data_dir: str = './data',
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
        num_examples : Optional[int], optional
            Maximum number of training examples to load, by default None (load all)
        num_test_examples : Optional[int], optional
            Maximum number of test examples to load, by default None (load all)
        
        Returns
        -------
        Tuple[Dict[str, Any], Dict[str, Any]]
            train_dataset, test_dataset containing features and targets
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
        
        # Load MNIST dataset
        train_data = torchvision.datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform
        )
        
        test_data = torchvision.datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=transform
        )
        
        # Determine how many samples to use
        train_total = len(train_data)
        test_total = len(test_data)
        
        # If num_examples is specified, use only that many samples
        if num_examples is not None and num_examples < train_total:
            train_indices = np.random.choice(train_total, size=num_examples, replace=False)
            train_samples = num_examples
            print(f"Using {num_examples} training examples out of {train_total} available")
        else:
            train_indices = range(train_total)
            train_samples = train_total
        
        # If num_test_examples is specified, use only that many samples
        if num_test_examples is not None and num_test_examples < test_total:
            test_indices = np.random.choice(test_total, size=num_test_examples, replace=False)
            test_samples = num_test_examples
            print(f"Using {num_test_examples} test examples out of {test_total} available")
        else:
            test_indices = range(test_total)
            test_samples = test_total
        
        # Initialize arrays with the exact sizes needed
        train_features = np.zeros((target_size[0], target_size[1], train_samples), dtype=np.float32)
        train_targets = np.zeros(train_samples, dtype=np.int64)
        
        print("Processing training images...")
        for i, idx in enumerate(tqdm(train_indices) if SHOW_PROGRESS_BAR else train_indices):
            # Get the image and target at the selected index
            img, target = train_data[idx]
            # Convert img from tensor (1, H, W) to numpy array (H, W)
            img_np = img.squeeze().numpy()
            train_features[:, :, i] = img_np
            train_targets[i] = target
        
        test_features = np.zeros((target_size[0], target_size[1], test_samples), dtype=np.float32)
        test_targets = np.zeros(test_samples, dtype=np.int64)
        
        print("Processing test images...")
        for i, idx in enumerate(tqdm(test_indices) if SHOW_PROGRESS_BAR else test_indices):
            # Get the image and target at the selected index
            img, target = test_data[idx]
            img_np = img.squeeze().numpy()
            test_features[:, :, i] = img_np
            test_targets[i] = target
        
        # Create dataset dictionaries with appropriate metadata
        train_dataset = self.create_dataset_dict(
            train_features, train_targets, 
            n_classes=10, split="train"
        )
        
        test_dataset = self.create_dataset_dict(
            test_features, test_targets, 
            n_classes=10, split="test"
        )
        
        return train_dataset, test_dataset

    def load_cifar10_dataset(self, data_dir: str = './data',
                          target_size: Tuple[int, int] = (32, 32),
                          num_examples: Optional[int] = None,
                          num_test_examples: Optional[int] = None,
                          **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load and preprocess the CIFAR-10 dataset.
        
        Parameters
        ----------
        data_dir : str, optional
            Directory to store the CIFAR-10 dataset, by default './data'
        target_size : Tuple[int, int], optional
            Size to resize images to, by default (32, 32)
        num_examples : Optional[int], optional
            Maximum number of training examples to load, by default None (load all)
        num_test_examples : Optional[int], optional
            Maximum number of test examples to load, by default None (load all)
        
        Returns
        -------
        Tuple[Dict[str, Any], Dict[str, Any]]
            train_dataset, test_dataset containing features and targets
        """
        try:
            # Import torchvision here to keep it as an optional dependency
            import torchvision
            import torchvision.transforms as transforms
        except ImportError:
            raise ImportError("torchvision is required to load CIFAR-10 dataset. "
                             "Install it with 'pip install torchvision'.")
        
        # Define transformations
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])
        
        # Load CIFAR-10 dataset
        train_data = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform
        )
        
        test_data = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=transform
        )
        
        # Determine how many samples to use
        train_total = len(train_data)
        test_total = len(test_data)
        
        # If num_examples is specified, use only that many samples
        if num_examples is not None and num_examples < train_total:
            train_indices = np.random.choice(train_total, size=num_examples, replace=False)
            train_samples = num_examples
            print(f"Using {num_examples} training examples out of {train_total} available")
        else:
            train_indices = range(train_total)
            train_samples = train_total
        
        # If num_test_examples is specified, use only that many samples
        if num_test_examples is not None and num_test_examples < test_total:
            test_indices = np.random.choice(test_total, size=num_test_examples, replace=False)
            test_samples = num_test_examples
            print(f"Using {num_test_examples} test examples out of {test_total} available")
        else:
            test_indices = range(test_total)
            test_samples = test_total
        
        # Initialize arrays with the exact sizes needed
        train_features = np.zeros((target_size[0], target_size[1], train_samples), dtype=np.float32)
        train_targets = np.zeros(train_samples, dtype=np.int64)
        
        print("Processing training images...")
        for i, idx in enumerate(tqdm(train_indices) if SHOW_PROGRESS_BAR else train_indices):
            # Get the image and target at the selected index
            img, target = train_data[idx]
            # Convert img from tensor (3, H, W) to numpy array (H, W) - convert to grayscale
            img_np = img.mean(dim=0).numpy()  # Average the color channels for grayscale
            train_features[:, :, i] = img_np
            train_targets[i] = target
        
        test_features = np.zeros((target_size[0], target_size[1], test_samples), dtype=np.float32)
        test_targets = np.zeros(test_samples, dtype=np.int64)
        
        print("Processing test images...")
        for i, idx in enumerate(tqdm(test_indices) if SHOW_PROGRESS_BAR else test_indices):
            # Get the image and target at the selected index
            img, target = test_data[idx]
            # Convert to grayscale
            img_np = img.mean(dim=0).numpy()
            test_features[:, :, i] = img_np
            test_targets[i] = target
        
        # CIFAR-10 class names
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Create dataset dictionaries with appropriate metadata
        train_dataset = self.create_dataset_dict(
            train_features, train_targets, 
            n_classes=10, split="train",
            class_names=class_names
        )
        
        test_dataset = self.create_dataset_dict(
            test_features, test_targets, 
            n_classes=10, split="test",
            class_names=class_names
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
        Name of the dataset to load, supports 'image_folder', 'mnist', 'cifar10',
        by default 'image_folder'
    **kwargs
        Additional arguments to pass to the dataset loader
        
    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any]]
        train_dataset, test_dataset
    """
    # Handle torchvision datasets
    if name.lower() == 'mnist':
        return _dataset_loader.load_mnist_dataset(**kwargs)
    elif name.lower() == 'cifar10':
        return _dataset_loader.load_cifar10_dataset(**kwargs)
    # Handle custom image folder datasets - if name doesn't match predefined datasets
    elif name.lower() == 'image_folder' or os.path.isdir(name):
        # If the name is a directory path, use it directly
        data_dir = name if os.path.isdir(name) else kwargs.get('data_dir')
        if not data_dir:
            raise ValueError(f"When using image_folder, data_dir must be provided")
        return _dataset_loader.load_image_folder_dataset(data_dir=data_dir, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {name}. Available types: 'image_folder', 'mnist', 'cifar10'")

# Utility functions
def flatten_images(data: np.ndarray, desc: str = "Flattening images") -> np.ndarray:
    """
    Flatten 3D image data into 2D matrix.
    
    Parameters
    ----------
    data : np.ndarray
        Image data tensor of shape (height, width, n_samples)
    desc : str, optional
        Description for progress bar, by default "Flattening images"
        
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
