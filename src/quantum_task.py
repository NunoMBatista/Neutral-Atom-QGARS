import numpy as np
from bloqade.analog.ir.location import Chain
from typing import Dict, Any
from tqdm import tqdm

def build_task(QRC_parameters: Dict[str, Any], detunings: np.ndarray):
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

def get_embeddings_emulation(xs: np.ndarray, qrc_params: Dict[str, Any], 
                            num_examples: int, n_shots: int = 1000) -> np.ndarray:
    """
    Function to get the embeddings from the quantum task.
    
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
    
    iterator = tqdm(range(num_examples), desc="Quantum simulation", unit="sample", position=2, leave=False) 
    
    # Process each example one at a time
    for i in iterator:   
        # Extract features for current example 
        features = xs[:, i] if len(xs.shape) > 1 else xs
        
        # Build and run quantum task
        task = build_task(
                    QRC_parameters=qrc_params, 
                    detunings=features
                )
        result = task.bloqade.python().run(shots=n_shots).report()
        
        # Process results and add to embeddings
        embedding = process_results(qrc_params, result)
        embeddings.append(embedding)
    
    return np.column_stack(embeddings)

def get_embeddings_with_checkpoint(xs: np.ndarray, qrc_params: Dict[str, Any], 
                                 num_examples: int, n_shots: int = 1000, 
                                 checkpoint_file: str = 'quantum_embeddings.joblib') -> np.ndarray:
    """
    Get embeddings with checkpoint support to avoid recomputing.
    
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
    embeddings = get_embeddings_emulation(xs, qrc_params, num_examples, n_shots)
    
    # Save checkpoint
    print(f"Saving embeddings checkpoint: {checkpoint_file}")
    dump(embeddings, checkpoint_file)
    
    return embeddings
