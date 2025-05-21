import numpy as np
from bloqade.analog.ir.location import Chain
from tqdm import tqdm
from typing import Dict, Any, Union, Optional, List, Callable, Any

from src.quantum_layer.quantum_task import get_embeddings_emulation
from src.utils.cli_printing import print_quantum_layer

class DetuningLayer:
    """
    Bloqade implementation of the quantum dynamics simulation.
    
    Parameters
    ----------
    geometry : str, optional
        Geometry specification (only 'chain' supported)
    n_atoms : int, optional
        Number of atoms in the chain
    lattice_spacing : float, optional
        Spacing between atoms (in μm)
    rabi_freq : float, optional
        Rabi frequency for the quantum dynamics
    t_end : float, optional
        Total evolution time
    n_steps : int, optional
        Number of time steps for readout
    readout_type : str, optional
        Type of readout measurements ("Z", "ZZ", or "all")
    encoding_scale : float, optional
        Scale for encoding features as detunings
    """
    def __init__(self, 
                 geometry: str = 'chain', 
                 n_atoms: int = 12, 
                 lattice_spacing: float = 10.0,
                 rabi_freq: float = 2*np.pi, 
                 t_end: float = 4.0, 
                 n_steps: int = 12,
                 readout_type: str = "all",
                 encoding_scale: float = 9.0):
        
        # Create chain geometry (only supported option)
        if geometry.lower() != 'chain':
            print(f"Warning: Only 'chain' geometry is supported. Ignoring requested geometry: {geometry}")
        
        # Convert numpy types to native Python types to avoid type errors
        n_atoms_int = int(n_atoms)  # Convert numpy.int64 to Python int
        lattice_spacing_float = float(lattice_spacing)  # Convert any numpy float types
        
        atom_geometry = Chain(n_atoms_int, lattice_spacing=lattice_spacing_float)

        # Define QRC parameters
        self.qrc_params = {
            "atom_number": n_atoms_int,
            "geometry_spec": atom_geometry,
            "encoding_scale": float(encoding_scale),
            "rabi_frequency": float(rabi_freq),
            "total_time": float(t_end),
            "time_steps": int(n_steps),
            "readouts": readout_type,
        }
        
       
    def __str__(self, use_colors: bool = True) -> str:
        return print_quantum_layer(self, use_colors=use_colors)
        
    def apply_layer(self, x: np.ndarray, n_shots: int = 1000, show_progress: bool = True) -> np.ndarray:
        """
        Apply quantum dynamics to input data.
        
        Parameters
        ----------
        x : np.ndarray
            Input features (dims × samples or dims)
        n_shots : int, optional
            Number of shots for quantum simulation
        show_progress : bool, optional
            Whether to show progress bar
            
        Returns
        -------
        np.ndarray
            Quantum embeddings
        """
        if len(x.shape) == 1:
            # Single sample
            return get_embeddings_emulation(x.reshape(-1, 1), self.qrc_params, 1, n_shots)
        else:
            outputs = get_embeddings_emulation(
                xs=x, 
                qrc_params=self.qrc_params, 
                num_examples=x.shape[1], 
                n_shots=n_shots
            )
            
            return outputs