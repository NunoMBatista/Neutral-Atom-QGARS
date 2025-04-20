from typing import Dict, Any
import numpy as np
import bloqade
from bloqade.analog.ir.location import Chain

from tqdm import tqdm
from quantum_task import get_embeddings_emulation

class DetuningLayer:
    """
    Bloqade implementation of the quantum dynamics simulation.
    
    A class that handles the quantum reservoir computing layer using
    Rydberg atom dynamics simulated with Bloqade.
    
    Attributes
    ----------
    qrc_params : Dict[str, Any]
        Dictionary containing quantum reservoir computing parameters
    
    Methods
    -------
    apply_layer(x)
        Apply quantum dynamics to input data
    """
    def __init__(self, n_atoms: int, rabi_freq: float, t_end: float, n_steps: int) -> None:
        """
        Initialize the detuning layer.
        
        Parameters
        ----------
        n_atoms : int
            Number of atoms in the chain
        rabi_freq : float
            Rabi frequency for the atomic transitions
        t_end : float
            Total evolution time
        n_steps : int
            Number of time steps for the evolution
        """
        # Create an atom chain with the specified number of atoms
        atom_chain = Chain(n_atoms, lattice_spacing=10)
        
        # Define QRC parameters
        self.qrc_params = {
            "atom_number": n_atoms,
            "geometry_spec": atom_chain,
            "encoding_scale": 9.0,
            "rabi_frequency": rabi_freq,
            "total_time": t_end,
            "time_steps": n_steps,
            "readouts": "ZZ"
        }
        
    def apply_layer(self, x: np.ndarray) -> np.ndarray:
        """
        Apply quantum dynamics to input data.
        
        Transforms input features through quantum simulation to create
        quantum embeddings.
        
        Parameters
        ----------
        x : np.ndarray
            Input features (dims × samples or dims)
            
        Returns
        -------
        np.ndarray
            Quantum embeddings (features × samples)
        """
        if len(x.shape) == 1:
            # Single sample
            return get_embeddings_emulation(x.reshape(-1, 1), self.qrc_params, 1)
        else:
            # Batch of samples
            print(f"Processing {x.shape[1]} samples...")
            return get_embeddings_emulation(x, self.qrc_params, x.shape[1])
