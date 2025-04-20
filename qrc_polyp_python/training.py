from typing import Tuple, List, Any, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from models import LinearClassifier, NeuralNetwork

def train(
    x_train: np.ndarray, 
    y_train: np.ndarray, 
    x_test: np.ndarray, 
    y_test: np.ndarray, 
    regularization: float = 0.0, 
    nepochs: int = 100, 
    batchsize: int = 100, 
    learning_rate: float = 0.01, 
    verbose: bool = True, 
    nonlinear: bool = False
) -> Tuple[List[float], List[float], List[float], Union[LinearClassifier, NeuralNetwork]]:
    """
    Train a model on the given data.
    
    Trains either a linear classifier or neural network on the provided 
    training data and evaluates performance on both training and test data.
    
    Parameters
    ----------
    x_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels (one-hot encoded)
    x_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    regularization : float, optional
        Weight decay for regularization (default is 0.0)
    nepochs : int, optional
        Number of training epochs (default is 100)
    batchsize : int, optional
        Batch size for training (default is 100)
    learning_rate : float, optional
        Learning rate for optimizer (default is 0.01)
    verbose : bool, optional
        Whether to show progress bars (default is True)
    nonlinear : bool, optional
        Whether to use a neural network (True) or linear classifier (False)
        (default is False)
        
    Returns
    -------
    Tuple[List[float], List[float], List[float], Union[LinearClassifier, NeuralNetwork]]
        A tuple containing:
        - losses: training loss history
        - accs_train: training accuracy history
        - accs_test: test accuracy history
        - model: the trained model
    """
    # Convert numpy arrays to PyTorch tensors
    x_train_tensor = torch.FloatTensor(x_train.T)  # Transpose to match PyTorch's expected shape
    y_train_tensor = torch.FloatTensor(y_train.T)
    
    # Print shapes for debugging
    if verbose:
        print(f"Training data shapes - X: {x_train.shape}, Y: {y_train.shape}")
        print(f"Tensor shapes - X: {x_train_tensor.shape}, Y: {y_train_tensor.shape}")
    
    # Ensure tensors have compatible dimensions for TensorDataset
    if x_train_tensor.shape[0] != y_train_tensor.shape[0]:
        raise ValueError(f"Tensor dimension mismatch: x_train_tensor has {x_train_tensor.shape[0]} samples, "
                         f"y_train_tensor has {y_train_tensor.shape[0]} samples")
    
    x_test_tensor = torch.FloatTensor(x_test.T)
    
    # Convert test labels to one-hot encoding if they're not already
    if len(y_test.shape) == 1:
        # One-hot encode if we have a 1D array of class indices
        n_classes = y_train.shape[0]
        y_test_one_hot = np.zeros((n_classes, y_test.shape[0]))
        for i, label in enumerate(y_test):
            y_test_one_hot[int(label), i] = 1.0
        y_test = y_test_one_hot
    
    y_test_tensor = torch.FloatTensor(y_test.T)
    
    # Create model
    input_dim = x_train.shape[0]
    output_dim = y_train.shape[0]
    
    if nonlinear:
        model = NeuralNetwork(input_dim, output_dim)
    else:
        model = LinearClassifier(input_dim, output_dim)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularization)
    
    # Create DataLoader
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    
    # Training loop
    if verbose:
        print("Training...")
    
    losses = []
    accs_train = []
    accs_test = []
    
    # Enhanced progress bar with description
    for epoch in tqdm(range(nepochs), desc="Training epochs", unit="epoch") if verbose else range(nepochs):
        model.train()
        epoch_loss = 0.0
        
        # Add progress bar for batches within each epoch if there are many batches
        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{nepochs}", leave=False) if verbose and len(train_loader) > 10 else train_loader
        for x_batch, y_batch in batch_iterator:
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(x_batch)
            
            # Compute loss - CrossEntropyLoss expects class indices
            loss = criterion(outputs, torch.argmax(y_batch, dim=1))
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            # Training accuracy
            train_outputs = model(x_train_tensor)
            _, train_pred = torch.max(train_outputs, 1)
            train_targets = torch.argmax(y_train_tensor, dim=1)
            train_acc = (train_pred == train_targets).sum().item() / train_targets.size(0)
            
            # Test accuracy
            test_outputs = model(x_test_tensor)
            _, test_pred = torch.max(test_outputs, 1)
            test_targets = torch.argmax(y_test_tensor, dim=1)
            test_acc = (test_pred == test_targets).sum().item() / test_targets.size(0)
        
        losses.append(epoch_loss / len(train_loader))
        accs_train.append(train_acc)
        accs_test.append(test_acc)
        
        # Update the progress bar with current metrics
        if verbose and epoch % 10 == 0:
            tqdm.write(f"Epoch {epoch+1}/{nepochs} - Loss: {losses[-1]:.4f} - Train Acc: {train_acc:.4f} - Test Acc: {test_acc:.4f}")
    
    return losses, accs_train, accs_test, model
