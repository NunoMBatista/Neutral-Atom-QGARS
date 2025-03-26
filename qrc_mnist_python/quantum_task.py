from itertools import chain
import numpy as np
import bloqade
from bloqade.ir.location import Chain, start

def build_task(QRC_parameters, xs1):
    """
        Build the quantum task for the QRC model.
    
        Input: 
            QRC_parameters: dictionary containing the parameters of the QRC model.
            xs1: list of input data.
        
        Output:
            rabi_oscillation_job: quantum job for the QRC model.
    """
    atom_number = QRC_parameters['atom_number']
    encoding_scale = QRC_parameters['encoding_scale']
    
    # Time between two consecutive probes
    delta_t = QRC_parameters['total_time'] / QRC_parameters['time_steps'] 
    
    # Get the atom chain 
    chain: Chain
    chain = QRC_parameters["geometry_spec"]
    
    rabi_oscillations_program = (
        #QRC_parameters["geometry_spec"]
        chain
        .rydberg.rabi.amplitude.uniform.constant(
            duration="run_time",
            value=QRC_parameters["rabi_frequency"]
        )
        .detuning.uniform.constant(
            duration="run_time",
            value=encoding_scale/2
        )
        .scale(list(xs1)).constant(
            duration="run_time",
            value=-encoding_scale
        )
    )
    
    # batch_assign is used to probe the quantum reservoir at a set number of timesteps
    rabi_oscillations_job = rabi_oscillations_program.batch_assign(
        run_time=np.arange(1, QRC_parameters["time_steps"]+1, 1) * delta_t
    )
    
    return rabi_oscillations_job


def process_results(QRC_parameters, report):
    """
        Process the results of the quantum task for the QRC model.
        
        Input:
            QRC_parameters: dictionary containing the parameters of the QRC model.
            report: quantum report containing the results of the quantum task.
            
        Output:
            embedding: list containing the embedding of the input data.
    """

    # In this context, embedding is the output of the quantum reservoir
    embedding = []
    atom_number = QRC_parameters['atom_number']
    try: 
        for t in range(QRC_parameters['time_steps']):
            ar1 = -1.0+2.0*((report.bitstrings())[t])
            nsh1 = ar1.shape[0]
            for i in range(atom_number):
                embedding.append(np.sum(ar1[:, i])/nsh1) # Z expectation values
            if QRC_parameters["readouts"] == "ZZ":
                for i in range(atom_number):
                    for j in range(i+1, atom_number):
                        embedding.append(np.sum(ar1[:, i]*ar1[:, j])/nsh1) # ZZ expectation values
                        
    except: # No experimental results were obtained
        print("No results were obtained.")
        for t in range(QRC_parameters["time_steps"]):
            for i in range(atom_number):
                embedding.append(0.0)
            if QRC_parameters["readouts"] == "ZZ":
                for i in range(atom_number):
                    for j in range(i+1, atom_number):
                        embedding.append(0.0)
    
    return embedding
        
def process_results_samples(QRC_parameters, report):
    """
        Process the results of the quantum task for the QRC model.
        
        Input:
            QRC_parameters: dictionary containing the parameters of the QRC model.
            report: quantum report containing the results of the quantum task.
            
        Output:
            embedding: list containing the embedding of the input data.
    """
    embedding = []
    atom_number = QRC_parameters['atom_number']
    try: 
        embedding=report.bitstrings()
    except: # No experimental results were obtained
        print("No results were obtained.")
        for t in range(QRC_parameters["time_steps"]):
            for i in range(atom_number):
                embedding.append(0.0)
            if QRC_parameters["readouts"] == "ZZ":
                for i in range(atom_number):
                    for j in range(i+1, atom_number):
                        embedding.append(0.0)
    
    return embedding    