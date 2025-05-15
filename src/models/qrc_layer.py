import numpy as np
from bloqade.analog.ir.location import Chain
from tqdm import tqdm
from typing import Dict, Any, Union, Optional, List, Callable
from quantum_backends import get_backend  # Direct import from backends package

class DetuningLayer:
    """
    High-level interface to quantum simulation backends.
    
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
    backend : str, optional
        Quantum simulation backend ('bloqade' or 'qutip')
    use_gpu : bool, optional
        Whether to use GPU acceleration with QuTiP
    """
    def __init__(self, 
                 geometry: str = 'chain', 
                 n_atoms: int = 12, 
                 lattice_spacing: float = 10.0,
                 rabi_freq: float = 2*np.pi, 
                 t_end: float = 4.0, 
                 n_steps: int = 12,
                 readout_type: str = "all",
                 encoding_scale: float = 9.0,
                 backend: str = "bloqade",
                 use_gpu: bool = False,
                 print_params: bool = True):
        
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
            "lattice_spacing": lattice_spacing_float,
            "encoding_scale": float(encoding_scale),
            "rabi_frequency": float(rabi_freq),
            "total_time": float(t_end),
            "time_steps": int(n_steps),
            "readouts": readout_type,
            "backend": backend,
            "use_gpu": use_gpu
        }
        
        # Initialize the backend
        self.backend = get_backend(backend)
        
        if print_params:
            print(f"""
                    *******************************************
                    *          Quantum Reservoir Layer        *
                    *******************************************
                    *                                           
                    *    Geometry: {geometry}                   
                    *    Number of atoms: {n_atoms_int}           
                    *    Lattice spacing: {lattice_spacing_float} μm
                    *    Rabi frequency: {float(rabi_freq)} Hz
                    *    Total evolution time: {float(t_end)} s
                    *    Number of time steps: {int(n_steps)}
                    *    Readout type: {readout_type}
                    *    Encoding scale: {float(encoding_scale)}
                    *    Backend: {backend}
                    *    Use GPU: {use_gpu}
                    *    
                    *******************************************
                  """)
        
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
            return self.backend.get_embeddings(x.reshape(-1, 1), self.qrc_params, 1, n_shots)
        else:
            # Batch of samples
            return self.backend.get_embeddings(x, self.qrc_params, x.shape[1], n_shots)
    
    def get_embeddings_with_checkpoint(self, xs: np.ndarray, 
                                     num_examples: int, n_shots: int = 1000, 
                                     checkpoint_file: str = 'quantum_embeddings.joblib') -> np.ndarray:
        """
        Get embeddings with checkpoint support to avoid recomputing.
        
        Parameters
        ----------
        xs : np.ndarray
            Training set
        num_examples : int
            Number of examples to process
        n_shots : int, optional
            Number of shots for the quantum task, by default 1000
        checkpoint_file : str, optional
            Filename to save/load embeddings, by default 'quantum_embeddings.joblib'
        
        Returns
        -------
        np.ndarray
            Array with the embeddings
        """
        import os
        from joblib import dump, load
        
        # Check if checkpoint exists
        if os.path.exists(checkpoint_file):
            print(f"Loading embeddings from checkpoint: {checkpoint_file}")
            return load(checkpoint_file)
        
        # If not, compute embeddings
        print("Computing embeddings...")
        embeddings = self.apply_layer(xs, n_shots)
        
        # Save checkpoint
        print(f"Saving embeddings checkpoint: {checkpoint_file}")
        dump(embeddings, checkpoint_file)
        
        return embeddings