from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np

class QuantumBackend(ABC):
    """Abstract base class defining the interface for quantum backends."""
    
    @abstractmethod
    def build_task(self, qrc_parameters: Dict[str, Any], detunings: np.ndarray) -> Any:
        """
        Build a quantum task for the backend.
        
        Parameters
        ----------
        qrc_parameters : Dict[str, Any]
            Dictionary with QRC parameters
        detunings : np.ndarray
            Array of detunings for each atom
        
        Returns
        -------
        Any
            A backend-specific task ready to be executed
        """
        pass
    
    @abstractmethod
    def get_embeddings(self, xs: np.ndarray, qrc_params: Dict[str, Any], 
                      num_examples: int, n_shots: int = 1000) -> np.ndarray:
        """
        Get quantum embeddings for input data.
        
        Parameters
        ----------
        xs : np.ndarray
            Input data
        qrc_params : Dict[str, Any]
            Dictionary with QRC parameters
        num_examples : int
            Number of examples to process
        n_shots : int, optional
            Number of shots for simulation, by default 1000
        
        Returns
        -------
        np.ndarray
            Array with quantum embeddings
        """
        pass

# Helper functions shared across backends
def process_results(QRC_parameters: Dict[str, Any], report: Any) -> np.ndarray:
    """
    Process the results from a quantum task.
    
    Parameters
    ----------
    QRC_parameters : Dict[str, Any]
        Dictionary with QRC parameters
    report : Any
        Report from the quantum task
    
    Returns
    -------
    np.ndarray
        Array of expectation values
    """
    # Initialize array for embedding vector
    embedding = []
    atom_number = QRC_parameters["atom_number"]
    readout_type = QRC_parameters.get("readouts", "ZZ")
    time_steps = QRC_parameters["time_steps"]
    
    # Process bitstrings for each time step
    for t in range(time_steps):
        # Convert bit values (0,1) to spin values (-1,+1)
        spin_values = -1.0 + 2.0 * (report.bitstrings()[t])
        n_shots = spin_values.shape[0]
        
        # Process based on readout type
        
        # Calculate Z expectation values for each atom
        for i in range(atom_number):
            embedding.append(np.sum(spin_values[:, i])/n_shots)
        
        # Add ZZ correlators if specified
        if readout_type == "ZZ" or readout_type == "all":
            for i in range(atom_number):
                for j in range(i+1, atom_number):
                    embedding.append(np.sum(spin_values[:, i]*spin_values[:, j])/n_shots)
        
        # Add other correlators if needed
        if readout_type == "all":
            # Add three-body ZZZ correlators
            for i in range(atom_number):
                for j in range(i+1, atom_number):
                    for k in range(j+1, atom_number):
                        embedding.append(np.sum(spin_values[:, i]*spin_values[:, j]*spin_values[:, k])/n_shots)
    
    return np.array(embedding)
