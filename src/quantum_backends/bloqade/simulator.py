import numpy as np
from bloqade.analog.ir.location import Chain
from typing import Dict, Any
from tqdm import tqdm
from ..backend_interface import QuantumBackend, process_results

class BloqadeBackend(QuantumBackend):
    """Bloqade backend implementation."""
    
    def build_task(self, QRC_parameters: Dict[str, Any], detunings: np.ndarray):
        """
        Build a quantum task using Bloqade.
        
        Parameters
        ----------
        QRC_parameters : Dict[str, Any]
            Dictionary with QRC parameters
        detunings : np.ndarray
            Array of detunings for each atom
        
        Returns
        -------
            A Bloqade Program ready to be executed
        """
        # Get parameters
        atom_geometry = QRC_parameters["geometry_spec"]
        rabi_frequency = float(QRC_parameters["rabi_frequency"])  # Convert to standard float
        total_time = float(QRC_parameters["total_time"])  # Convert to standard float
        encoding_scale = float(QRC_parameters["encoding_scale"])  # Convert to standard float
        
        # Time between consecutive probes
        delta_t = total_time / int(QRC_parameters["time_steps"])  # Convert to standard int
        
        # Create the bloqade program using the flexible API

        # Initialize the program with the atom geometry
        program = atom_geometry
            
        # We can customize the rabi amplitude and detuning profiles (constant for now)
        # Add Rabi amplitude based on the constant profile
        program = program.rydberg.rabi.amplitude.uniform.constant(
            duration="run_time",
            value=rabi_frequency
        )
            
        # Add detuning based on the constant profile
        program = program.detuning.uniform.constant(
            duration="run_time",
            value=encoding_scale/2
        ).scale(list(detunings)).constant(
            duration="run_time",
            value=-encoding_scale
        )
        
        # Batch assign to probe the quantum system at multiple timesteps
        program_job = program.batch_assign(
            run_time=np.arange(1, int(QRC_parameters["time_steps"])+1, 1) * delta_t
        )
        
        return program_job
    
    def get_embeddings(self, xs: np.ndarray, qrc_params: Dict[str, Any], 
                       num_examples: int, n_shots: int = 1000) -> np.ndarray:
        """
        Function to get the embeddings from the Bloqade quantum task.
        
        Parameters
        ----------
        xs : np.ndarray
            Training set
        qrc_params : Dict[str, Any]
            Dictionary with QRC parameters
        num_examples : int
            Number of examples to process
        n_shots : int, optional
            Number of shots for the quantum task, by default 1000
        
        Returns
        -------
        np.ndarray
            Array with the embeddings of the training set
        """    
        embeddings = []
        
        iterator = tqdm(range(num_examples), desc="Bloqade simulation", unit="sample", position=2, leave=False) 
        
        # Process each example one at a time
        for i in iterator:   
            # Extract features for current example 
            features = xs[:, i] if len(xs.shape) > 1 else xs
            
            # Build and run quantum task
            task = self.build_task(
                        QRC_parameters=qrc_params, 
                        detunings=features
                    )
            result = task.bloqade.python().run(shots=n_shots).report()
            
            # Process results and add to embeddings
            embedding = process_results(qrc_params, result)
            embeddings.append(embedding)
        
        return np.column_stack(embeddings)
