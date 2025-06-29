from calendar import c
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from typing import Tuple, List, Dict, Any, Optional, Union
from sklearn.metrics import confusion_matrix, f1_score
from src.globals import ResultsDict, TrainingResults
from src.classification_models.models import LinearClassifier, NeuralNetwork


def train(x_train: np.ndarray, y_train: np.ndarray, 
          x_test: np.ndarray, y_test: np.ndarray, 
          regularization: float = 0.0, nepochs: int = 100, 
          batchsize: int = 100, learning_rate: float = 0.01, 
          verbose: bool = True, nonlinear: bool = False
          ) -> TrainingResults:
    """
    Train a model on the given data.
    
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
        Weight decay for regularization, by default 0.0
    nepochs : int, optional
        Number of training epochs, by default 100
    batchsize : int, optional
        Batch size for training, by default 100
    learning_rate : float, optional
        Learning rate for optimizer, by default 0.01
    verbose : bool, optional
        Whether to show progress bars, by default True
    nonlinear : bool, optional
        Whether to use a neural network (True) or linear classifier (False), by default False
    
    Returns
    -------
    Tuple[List[float], List[float], List[float], nn.Module, 
           np.ndarray, np.ndarray, float, float]
        - losses: Training losses per epoch
        - accs_train: Training accuracies per epoch
        - accs_test: Test accuracies per epoch
        - model: Trained model
        - confusion_matrix_train: Confusion matrix for training set
        - confusion_matrix_test: Confusion matrix for test set
        - f1_train: F1 score for training set
        - f1_test: F1 score for test set
    """
    # Convert numpy arrays to PyTorch tensors
    x_train_tensor = torch.FloatTensor(x_train.T)  # Transpose to match PyTorch's expected shape
    y_train_tensor = torch.FloatTensor(y_train.T)
    x_test_tensor = torch.FloatTensor(x_test.T)
    y_test_tensor = torch.LongTensor(np.argmax(y_test, axis=0) if len(y_test.shape) > 1 else y_test)
    
    # Convert batchsize to Python native int to avoid PyTorch DataLoader errors
    batchsize = int(batchsize)
    
    # Adjust batch size if it's larger than dataset size
    train_samples = x_train_tensor.shape[0]
    adjusted_batchsize = min(batchsize, train_samples)
    if adjusted_batchsize != batchsize and verbose:
        print(f"Warning: Reducing batch size from {batchsize} to {adjusted_batchsize} to match dataset size")
    
    # Create model
    input_dim = x_train.shape[0]
    output_dim = y_train.shape[0] if len(y_train.shape) > 1 else len(np.unique(y_train))
    
    if nonlinear:
        model = NeuralNetwork(input_dim, output_dim)
    else:
        model = LinearClassifier(input_dim, output_dim)
    
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
                        params=model.parameters(), 
                        lr=learning_rate, 
                        weight_decay=regularization
                    )
    
    
    # Create DataLoader
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(
                            dataset=train_dataset, 
                            batch_size=adjusted_batchsize, 
                            shuffle=True
                        )
    
    # Training loop
    if verbose:
        print("Training...")
    
    losses = []
    accs_train = []
    accs_test = []
    train_targets: Optional[torch.Tensor] = None
    train_pred: Optional[torch.Tensor] = None
    test_pred: Optional[torch.Tensor] = None
    confusion_matrix_train: Optional[np.ndarray] = None
    confusion_matrix_test: Optional[np.ndarray] = None
    f1_train: Optional[float] = None
    f1_test: Optional[float] = None
    
    # Enhanced progress bar with description
    for epoch in tqdm(range(nepochs), desc="Training epochs", unit="epoch") if verbose else range(nepochs):
        model.train() # Set the model to training mode
        epoch_loss = 0.0
        
        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{nepochs}", leave=False) if (verbose and len(train_loader) > 10) else train_loader
        for x_batch, y_batch in batch_iterator:
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(x_batch)
            
            # Compute loss
            loss = criterion(outputs, torch.argmax(y_batch, dim=1))
            
            # Backward pass and optimize
            loss.backward() # Compute gradients
            optimizer.step() # Update weights
            
            epoch_loss += loss.item()
        
        # Evaluate
        model.eval() # Set the model to evaluation mode
        with torch.no_grad():
            # Training accuracy
            train_outputs = model(x_train_tensor)
            _, train_pred = torch.max(train_outputs, 1)
            train_targets = torch.argmax(y_train_tensor, dim=1)
            train_acc = (train_pred == train_targets).sum().item() / train_targets.size(0)
            # Update confusion matrix for training
            confusion_matrix_train = confusion_matrix(
                train_targets.numpy(), 
                train_pred.numpy(), 
                labels=range(output_dim)
            )          
            
            # Test accuracy
            test_outputs = model(x_test_tensor)
            _, test_pred = torch.max(test_outputs, 1)
            test_acc = (test_pred == y_test_tensor).sum().item() / y_test_tensor.size(0)
            # Update confusion matrix for testing
            confusion_matrix_test = confusion_matrix(
                y_test_tensor.numpy(), 
                test_pred.numpy(), 
                labels=range(output_dim)
            )
        
        losses.append(epoch_loss / len(train_loader))
        accs_train.append(train_acc)
        accs_test.append(test_acc)
        
        # Update the progress bar with current metrics
        if verbose:
            tqdm.write(f"Epoch {epoch+1}/{nepochs} - Loss: {losses[-1]:.4f} - Train Acc: {train_acc:.4f} - Test Acc: {test_acc:.4f}")
    
    # Calculate final F1 scores
    if output_dim == 2:
        average_type = 'binary'
    else:
        average_type = 'weighted'


    f1_train = f1_score(
        train_targets.numpy(),
        train_pred.numpy(),
        labels=range(output_dim),
        average=average_type
    )
    f1_test = f1_score(
        y_test_tensor.numpy(),
        test_pred.numpy(),
        labels=range(output_dim),
        average=average_type
    )
    
    
    return losses, accs_train, accs_test, model, confusion_matrix_train, confusion_matrix_test, f1_train, f1_test
