import numpy as np
from bloqade.analog.ir.location import Chain
from tqdm import tqdm
from typing import Dict, Any
from quantum_task import get_embeddings_emulation

class DetuningLayer:
    """
    Bloqade implementation of the quantum dynamics simulation.
    
    Parameters
    ----------
    n_atoms : int
        Number of atoms in the chain
    rabi_freq : float
        Rabi frequency for the quantum dynamics
    t_end : float
        Total evolution time
    n_steps : int
        Number of time steps for readout
    """
    def __init__(self, n_atoms: int, rabi_freq: float, t_end: float, n_steps: int):
        # Create an atom chain with the specified number of atoms
        atom_chain = Chain(n_atoms, lattice_spacing=10)
        
        # Define QRC parameters
        self.qrc_params = {
            "atom_number": n_atoms,
            "geometry_spec": atom_chain,
            "encoding_scale": 9.0,  # Use same encoding scale as MNIST example
            "rabi_frequency": rabi_freq,
            "total_time": t_end,
            "time_steps": n_steps,
            "readouts": "ZZ"
        }
        
    def apply_layer(self, x: np.ndarray) -> np.ndarray:
        """
        Apply quantum dynamics to input data.
        
        Parameters
        ----------
        x : np.ndarray
            Input features (dims × samples or dims)
            
        Returns
        -------
        np.ndarray
            Quantum embeddings
        """
        if len(x.shape) == 1:
            # Single sample
            return get_embeddings_emulation(x.reshape(-1, 1), self.qrc_params, 1)
        else:
            # Batch of samples
            print(f"Processing {x.shape[1]} samples...")
            outputs = []
            iterator = tqdm(range(x.shape[1]), desc="Quantum simulation", unit="sample")
            for i in iterator:
                outputs.append(get_embeddings_emulation(x[:, i].reshape(-1, 1), self.qrc_params, 1)[0])
            return np.column_stack(outputs)
