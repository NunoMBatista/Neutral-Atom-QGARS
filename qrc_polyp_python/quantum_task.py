from typing import Dict, Any, List, Union, Optional
import numpy as np
import bloqade
from bloqade.analog.ir.location import Chain

from tqdm import tqdm

def build_task(QRC_parameters: Dict[str, Any], detunings: np.ndarray) -> Any:
    """
    Build a quantum task using Bloqade.
    
    Creates a Bloqade program for simulating Rydberg atom dynamics with
    the given parameters and detunings.
    
    Parameters
    ----------
    QRC_parameters : Dict[str, Any]
        Dictionary with QRC parameters including:
        - geometry_spec: atom chain geometry
        - rabi_frequency: Rabi frequency
        - total_time: total evolution time
        - encoding_scale: scaling factor for data encoding
        - time_steps: number of time steps
    detunings : np.ndarray
        Array of detunings for each atom
    
    Returns
    -------
    Any
        A Bloqade Program ready to be executed
    """
    # Get parameters
    atom_chain = QRC_parameters["geometry_spec"]
    rabi_frequency = QRC_parameters["rabi_frequency"]
    total_time = QRC_parameters["total_time"]
    encoding_scale = QRC_parameters["encoding_scale"]
    
    # Time between consecutive probes
    delta_t = total_time / QRC_parameters["time_steps"]
    
    # Create the bloqade program using the API style from the MNIST example
    rabi_oscillations_program = (
        atom_chain
        .rydberg.rabi.amplitude.uniform.constant(
            duration="run_time",
            value=rabi_frequency
        )
        .detuning.uniform.constant(
            duration="run_time",
            value=encoding_scale/2
        )
        .scale(list(detunings)).constant(
            duration="run_time",
            value=-encoding_scale
        )
    )
    
    # Batch assign to probe the quantum system at multiple timesteps
    rabi_oscillations_job = rabi_oscillations_program.batch_assign(
        run_time=np.arange(1, QRC_parameters["time_steps"]+1, 1) * delta_t
    )
    
    return rabi_oscillations_job

def process_results(QRC_parameters: Dict[str, Any], report: Any) -> np.ndarray:
    """
    Process the results from a quantum task.
    
    Extracts and processes measurement results from a quantum simulation,
    calculating expectation values for observables.
    
    Parameters
    ----------
    QRC_parameters : Dict[str, Any]
        Dictionary with QRC parameters including:
        - atom_number: number of atoms
        - time_steps: number of time steps
        - readouts: type of readout ("Z" or "ZZ")
    report : Any
        Report from the quantum task containing measurement results
    
    Returns
    -------
    np.ndarray
        Array of expectation values forming the quantum embedding
    """
    # Initialize array for embedding vector
    embedding = []
    atom_number = QRC_parameters["atom_number"]
    
    try: 
        # Process bitstrings for each time step
        for t in range(QRC_parameters['time_steps']):
            # Convert bit values (0,1) to spin values (-1,+1)
            ar1 = -1.0 + 2.0 * ((report.bitstrings())[t])
            nsh1 = ar1.shape[0]
            
            # Calculate Z expectation values for each atom
            for i in range(atom_number):
                embedding.append(np.sum(ar1[:, i])/nsh1)
            
            # Add ZZ correlators if needed
            if QRC_parameters.get("readouts") == "ZZ":
                for i in range(atom_number):
                    for j in range(i+1, atom_number):
                        embedding.append(np.sum(ar1[:, i]*ar1[:, j])/nsh1)
                        
    except Exception as e:
        print(f"Error processing results: {e}")
        # Fallback to zeros if no results obtained
        for t in range(QRC_parameters["time_steps"]):
            for i in range(atom_number):
                embedding.append(0.0)
            if QRC_parameters.get("readouts") == "ZZ":
                for i in range(atom_number):
                    for j in range(i+1, atom_number):
                        embedding.append(0.0)
    
    return np.array(embedding)

def get_embeddings_emulation(
    xs: np.ndarray, 
    qrc_params: Dict[str, Any], 
    num_examples: int, 
    n_shots: int = 1000
) -> np.ndarray:
    """
    Get embeddings from quantum tasks.
    
    Processes input data through quantum simulation to obtain
    quantum feature embeddings.
    
    Parameters
    ----------
    xs : np.ndarray
        Training set features
    qrc_params : Dict[str, Any]
        Dictionary with QRC parameters
    num_examples : int
        Number of examples to process
    n_shots : int, optional
        Number of shots for the quantum task (default is 1000)
    
    Returns
    -------
    np.ndarray
        Array with the embeddings of the training set
    """    
    embeddings = []
    
    # Process each example one at a time with progress bar
    for i in tqdm(range(num_examples), desc="Processing quantum samples", unit="sample"):
        try:
            # Extract features for current example 
            features = xs[:, i] if len(xs.shape) > 1 else xs
            
            # Build quantum task
            task = build_task(qrc_params, features)
            
            # Run simulation
            #result = task.bloqade.python().run(shots=n_shots).report()
            result = task.bloqade.python().run(shots=n_shots).report()
            
            # Process results and add to embeddings
            embedding = process_results(qrc_params, result)
            embeddings.append(embedding)
            
            # Update progress bar with completion percentage
            if i % 10 == 0 or i == num_examples - 1:
                tqdm.write(f"Completed {i+1}/{num_examples} samples ({(i+1)/num_examples*100:.1f}%)")
                
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            # Create dummy embedding of the right size if processing fails
            dim = qrc_params["atom_number"] * qrc_params["time_steps"]
            if qrc_params.get("readouts") == "ZZ":
                dim += qrc_params["atom_number"] * (qrc_params["atom_number"] - 1) // 2 * qrc_params["time_steps"]
            embeddings.append(np.zeros(dim))
    
    # Return as a transposed array where each column is a sample (features × samples)
    return np.array(embeddings).T
