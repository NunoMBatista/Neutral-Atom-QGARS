import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional
import argparse
import os
import sys

# Add the project root to the Python path if running as main script
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

def visualize_reconstructions(model: nn.Module, 
                             data: torch.Tensor, 
                             device: str = 'cpu',
                             num_samples: int = 5,
                             image_shape: Optional[Tuple[int, int, int]] = None) -> None:
    """
    Visualize original images and their reconstructions.
    
    Parameters
    ----------
    model : nn.Module
        Trained autoencoder model
    data : torch.Tensor
        Input data tensor
    device : str, optional
        Device to use, by default 'cpu'
    num_samples : int, optional
        Number of samples to visualize, by default 5
    image_shape : Optional[Tuple[int, int, int]], optional
        Shape of the input image (height, width, channels), by default None
    """
    model.eval()
    
    # Use a subset of data
    subset = data[:num_samples].to(device)
    
    # Get reconstructions
    with torch.no_grad():
        _, reconstructions = model(subset)
    
    # Convert to numpy for plotting
    originals = subset.cpu().numpy()
    reconstructions = reconstructions.cpu().numpy()
    
    # If we know the image shape, reshape for visualization
    if image_shape:
        height, width, channels = image_shape
        
        # Function to reshape flat vector to image
        def reshape_to_image(flat_vec):
            if channels == 1:
                return flat_vec.reshape(height, width)
            else:
                return flat_vec.reshape(height, width, channels)
        
        # Reshape each sample
        originals_reshaped = [reshape_to_image(orig) for orig in originals]
        recons_reshaped = [reshape_to_image(recon) for recon in reconstructions]
        
        # Plot
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 5))
        
        for i in range(num_samples):
            # Original
            if channels == 1:
                axes[0, i].imshow(originals_reshaped[i], cmap='gray')
            else:
                axes[0, i].imshow(originals_reshaped[i])
            axes[0, i].set_title(f"Original {i+1}")
            axes[0, i].axis('off')
            
            # Reconstruction
            if channels == 1:
                axes[1, i].imshow(recons_reshaped[i], cmap='gray')
            else:
                axes[1, i].imshow(recons_reshaped[i])
            axes[1, i].set_title(f"Reconstruction {i+1}")
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()
    else:
        # Just show the flattened vectors
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 5))
        
        for i in range(num_samples):
            axes[0, i].plot(originals[i])
            axes[0, i].set_title(f"Original {i+1}")
            
            axes[1, i].plot(reconstructions[i])
            axes[1, i].set_title(f"Reconstruction {i+1}")
        
        plt.tight_layout()
        plt.show()

def print_layer_shapes(model: nn.Module, input_shape: Tuple[int, ...]) -> None:
    """
    Print the shapes of tensors as they pass through each layer.
    
    Parameters
    ----------
    model : nn.Module
        Model to analyze
    input_shape : Tuple[int, ...]
        Shape of input tensor (batch_size, ...)
    """
    # Create a dummy input tensor
    x = torch.randn(input_shape)
    
    # Hook to print layer output shapes
    def hook_fn(module, input, output):
        print(f"{module.__class__.__name__}: Input shape {input[0].shape}, Output shape {output.shape}")
    
    # Register hooks for all layers
    hooks = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
            hooks.append(module.register_forward_hook(hook_fn))
    
    # Forward pass
    try:
        model(x)
    except Exception as e:
        print(f"Error during forward pass: {e}")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()

def visualize_rgb_vs_grayscale(original_image_path: str) -> None:
    """
    Visualize the difference between RGB and grayscale processing of an image.
    
    Parameters
    ----------
    original_image_path : str
        Path to the original image
    """
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    from src.data_processing.data_processing import DatasetLoader
    
    # Load the image
    img = Image.open(original_image_path)
    
    # Create dataset loader
    loader = DatasetLoader()
    
    # Process image in RGB and grayscale
    target_size = (128, 128)  # Example size
    rgb_img = loader.preprocess_image(img, target_size, keep_rgb=True)
    gray_img = loader.preprocess_image(img, target_size, keep_rgb=False)
    
    # Plot original and processed images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(np.array(img))
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # RGB processed
    axes[1].imshow(rgb_img)
    axes[1].set_title(f"RGB Image ({rgb_img.shape})")
    axes[1].axis('off')
    
    # Grayscale processed
    axes[2].imshow(gray_img.squeeze(), cmap='gray')
    axes[2].set_title(f"Grayscale Image ({gray_img.shape})")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print flattened dimensions
    rgb_flat = rgb_img.reshape(-1)
    gray_flat = gray_img.reshape(-1)
    
    print(f"RGB flattened dimension: {rgb_flat.shape}")
    print(f"Grayscale flattened dimension: {gray_flat.shape}")
    print(f"Ratio: {rgb_flat.shape[0] / gray_flat.shape[0]:.1f}x")

def debug_train_convolutional_autoencoder(
        dataset_type: str = 'mnist',
        data_dir: Optional[str] = None,
        target_size: Tuple[int, int] = (28, 28),
        num_examples: int = 500,
        num_test_examples: int = 50,
        encoding_dim: int = 12,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        keep_rgb: bool = True,
        visualize_samples: int = 5
    ) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train a convolutional autoencoder and visualize results.
    
    Parameters
    ----------
    dataset_type : str, optional
        Type of dataset to use, by default 'mnist'
    data_dir : Optional[str], optional
        Path to data directory, by default None
    target_size : Tuple[int, int], optional
        Size to resize images to, by default (28, 28)
    num_examples : int, optional
        Number of training examples, by default 500
    num_test_examples : int, optional
        Number of test examples, by default 50
    encoding_dim : int, optional
        Dimension of encoded representation, by default 12
    epochs : int, optional
        Number of training epochs, by default 10
    batch_size : int, optional
        Batch size for training, by default 32
    learning_rate : float, optional
        Learning rate for training, by default 0.001
    keep_rgb : bool, optional
        Whether to keep RGB channels, by default True
    visualize_samples : int, optional
        Number of samples to visualize, by default 5
        
    Returns
    -------
    Tuple[nn.Module, Dict[str, Any]]
        Trained autoencoder model and test dataset
    """
    from src.data_processing.data_processing import load_dataset, flatten_images
    from src.feature_reduction.autoencoder.autoencoder import Autoencoder, train_autoencoder
    
    print(f"Loading dataset: {dataset_type}")
    
    # Handle special datasets
    if dataset_type == 'cvc_clinic_db_patches':
        # Default path for CVC-ClinicDB if not specified
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                   "../../../data/datasets/cvc_clinic_db_patches")
            print(f"Using default path for CVC-ClinicDB: {data_dir}")
            
        # Load as image folder dataset
        train_data, test_data = load_dataset(
            name='image_folder',
            data_dir="./data/datasets/cvc_clinic_db_patches",
            target_size=target_size,
            num_examples=num_examples,
            num_test_examples=num_test_examples,
            keep_rgb=keep_rgb
        )
    else:
        # Regular dataset loading
        train_data, test_data = load_dataset(
            name=dataset_type,
            data_dir=data_dir,
            target_size=target_size,
            num_examples=num_examples,
            num_test_examples=num_test_examples,
            keep_rgb=keep_rgb
        )
    
    print(f"Dataset loaded: {dataset_type}")
    print(f"Number of training samples: {train_data['metadata']['n_samples']}")
    print(f"Number of test samples: {test_data['metadata']['n_samples']}")
    
    # Flatten images
    print("Flattening images...")
    train_features = train_data["features"]
    
    # Check the actual shape of the data
    if len(train_features.shape) == 3:  # This is grayscale data (H, W, N)
        if keep_rgb:
            # Convert grayscale to RGB by repeating the channel 3 times
            # Create a new array with RGB channels
            rgb_features = np.zeros((train_features.shape[0], train_features.shape[1], 3, train_features.shape[2]), dtype=np.float32)
            for i in range(train_features.shape[2]):
                # Repeat the grayscale channel 3 times
                for c in range(3):
                    rgb_features[:, :, c, i] = train_features[:, :, i]
            train_features = rgb_features
            print(f"Converted grayscale data to RGB with shape {train_features.shape}")
    
    data_flat = flatten_images(train_features)
    
    # Determine image shape for convolutional autoencoder
    height, width = train_features.shape[0:2]
    channels = 3 if keep_rgb else 1
    image_shape = (height, width, channels)
    print(f"Image shape: {image_shape}")
    
    # Train autoencoder
    print("Training convolutional autoencoder...")
    input_dim = data_flat.shape[0]
    
    model, spectral = train_autoencoder(
        data=data_flat,
        encoding_dim=encoding_dim,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        device='cpu',
        verbose=True,
        use_batch_norm=True,
        dropout=0.1,
        autoencoder_regularization=1e-5,
        ae_type='convolutional',
        image_shape=image_shape
    )
    
    # Prepare visualization data
    print("Preparing data for visualization...")
    test_features = test_data["features"]
    
    # Apply the same RGB conversion to test data if needed
    if len(test_features.shape) == 3 and keep_rgb:  # Grayscale data but RGB mode
        rgb_test_features = np.zeros((test_features.shape[0], test_features.shape[1], 3, test_features.shape[2]), dtype=np.float32)
        for i in range(test_features.shape[2]):
            for c in range(3):
                rgb_test_features[:, :, c, i] = test_features[:, :, i]
        test_features = rgb_test_features
        print(f"Converted test grayscale data to RGB with shape {test_features.shape}")
    
    test_flat = flatten_images(test_features)
    
    # Convert to torch tensor
    test_tensor = torch.tensor(test_flat.T, dtype=torch.float32)
    
    # Visualize reconstructions
    print("Visualizing reconstructions...")
    visualize_reconstructions(
        model=model,
        data=test_tensor[:visualize_samples],
        num_samples=visualize_samples,
        image_shape=image_shape
    )
    
    return model, test_data

def main():
    """Command-line interface for debug training and visualization."""
    parser = argparse.ArgumentParser(description="Debug and visualize convolutional autoencoder")
    
    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="mnist", 
                        help="Dataset to use (mnist, fashion_mnist, image_folder)")
    parser.add_argument("--data-dir", type=str, default=None, 
                        help="Path to dataset directory")
    parser.add_argument("--target-size", type=int, nargs=2, default=[28, 28],
                        help="Target image size (height width)")
    parser.add_argument("--num-examples", type=int, default=500,
                        help="Number of training examples")
    parser.add_argument("--num-test-examples", type=int, default=50,
                        help="Number of test examples")
    
    # Autoencoder parameters
    parser.add_argument("--encoding-dim", type=int, default=12,
                        help="Dimension of encoded representation")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate for training")
    
    # Visualization parameters
    parser.add_argument("--visualize-samples", type=int, default=5,
                        help="Number of samples to visualize")
    parser.add_argument("--keep-rgb", action="store_true",
                        help="Keep RGB channels instead of converting to grayscale")
    
    args = parser.parse_args()
    
    # Run debug training and visualization
    debug_train_convolutional_autoencoder(
        dataset_type=args.dataset,
        data_dir=args.data_dir,
        target_size=tuple(args.target_size),
        num_examples=args.num_examples,
        num_test_examples=args.num_test_examples,
        encoding_dim=args.encoding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        keep_rgb=args.keep_rgb,
        visualize_samples=args.visualize_samples
    )

# Example usage
if __name__ == "__main__":
    """
    Example of how to use the debug helpers
    
    Usage:
    1. For inspecting model shapes:
    
       from src.feature_reduction.autoencoder.debug_helpers import print_layer_shapes
       from src.feature_reduction.autoencoder.autoencoder import Autoencoder
       
       # Create model with any image dimensions
       model = Autoencoder(input_dim=150*150*3, encoding_dim=32, ae_type='convolutional', 
                         image_shape=(150, 150, 3))  # Example with 150x150 RGB images
       
       # Print shapes for a batch of images
       print_layer_shapes(model, (10, 150*150*3))
    
    2. For visualizing reconstructions:
    
       from src.feature_reduction.autoencoder.debug_helpers import visualize_reconstructions
       import torch
       
       # Load some sample data
       data = torch.randn(5, 150*150*3)  # 5 sample images at 150x150 RGB
       
       # Visualize reconstructions
       visualize_reconstructions(model, data, image_shape=(150, 150, 3))
    """
    print("This module provides helper functions for debugging autoencoders.")
    print("Import and use these functions in your code as shown in the docstring.")
    
    # Example with larger images (original CVC dataset resolution is often higher)
    model, test_data = debug_train_convolutional_autoencoder(
        dataset_type='cvc_clinic_db_patches',
        encoding_dim=12,
        epochs=100,
        batch_size=32,
        num_examples=200,
        num_test_examples=10,
        target_size=(64, 64),  # Using 64x64 resolution for better quality
        keep_rgb=True,
        visualize_samples=5
    )
