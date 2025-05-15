from typing import Dict, Any, Optional
from .backend_interface import QuantumBackend

def get_backend(backend_name: str) -> QuantumBackend:
    """
    Factory function to get the appropriate backend.
    
    Parameters
    ----------
    backend_name : str
        Name of the backend ('bloqade' or 'qutip')
    
    Returns
    -------
    QuantumBackend
        An instance of the requested backend
    """
    if backend_name.lower() == 'bloqade':
        from .bloqade.simulator import BloqadeBackend
        return BloqadeBackend()
    elif backend_name.lower() == 'qutip':
        from .qutip.simulator import QuTiPBackend
        return QuTiPBackend()
    else:
        raise ValueError(f"Unknown backend: {backend_name}. Available backends: 'bloqade', 'qutip'")
