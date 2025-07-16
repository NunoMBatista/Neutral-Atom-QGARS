import os
import argparse
from datetime import datetime
from tkinter.tix import AUTO
from typing import Dict, Any, List, Optional
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from parameter_sweep import ParameterSweep
from utils.config_manager import ConfigManager

# Define experiment profiles with parameter ranges
EXPERIMENT_PROFILES = {
    # DONE, 1e-5 and 1-e3 yielded the best results
    "autoencoder_regularization": {
        "description": "Sweep over autoencoder regularization parameters",
        "param_grid": {
            "reduction_method": ["guided_autoencoder"],
            "autoencoder_regularization": [0.0, 0.00001, 0.0001, 0.001, 0.01],
            "quantum_update_frequency": [5],
            "dim_reduction": [12]
        }
    },
    
    # DONE, 0.7 yielded the best results
    "guided_autoencoder_lambda": {
        "description": "Sweep over guided autoencoder lambda parameter",
        "param_grid": {
            "reduction_method": ["guided_autoencoder"],
           # "guided_lambda": [1, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1, 0.05, 0], # SWEEP IN LOG SCALE!
           
           
           # ADD SYMETRIC
            "guided_lambda": [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.0,
                              0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                              1-0.01, 1-0.001, 1-0.0001, 1-0.00001, 1-0.000001, 1],
            "quantum_update_frequency": [5],
            "dim_reduction": [12]
        }
    },
    
    # DONE
    # SHOULD SHOW THAT USING MORE QUBITS IS BETTER
    "encoding_dimensions": {
        "description": "Sweep over encoding dimensions and methods",
        "param_grid": {
            "dim_reduction": [4, 6, 8, 10, 12],
            "reduction_method": ["pca", "autoencoder", "guided_autoencoder"],
            "n_shots": [1000],
            "autoencoder_regularization": [1e-5], # FIXED FROM THE PREVIOUS STEPS
            "guided_lambda": [0.7], # FIXED FROM THE PREVIOUS STEPS
            "quantum_update_frequency": [1] # SHOWS QUANTUM GUIDED AUTOENCODER PEAK PERFORMANCE
        }
    },
    
    # DONE 
    "guided_autoencoder_update_frequency": {
        "description": "Sweep over guided autoencoder update frequency",
        "param_grid": {
            "reduction_method": ["guided_autoencoder"],
            "guided_lambda": [0.7], # FIX WITH THE ONE THAT WORKED BEST IN THE PREVIOUS SWEEP
            "quantum_update_frequency": [1, 3, 5, 7, 10, 15, 25, 0],
            "dim_reduction": [12], # PEAK PERFORMANCE
            "autoencoder_regularization": [1e-5] # FIXED FROM THE PREVIOUS STEPS
        }
    },
    
    # NOT DONE
    "quantum_readouts": {
        "description": "Sweep over quantum readout types and time steps",
        "param_grid": {
            "readout_type": ["Z", "ZZ", "all"],
            "time_steps": [4, 8, 12, 16],
            "reduction_method": ["guided_autoencoder"],
            "dim_reduction": [12]
        }
    },
    
    # NOT DONE
    "encoding_scale": {
        "description": "Sweep over encoding scale parameter",
        "param_grid": {
            "encoding_scale": [3.0, 6.0, 9.0, 12.0],
            "detuning_max": [3.0, 6.0, 9.0],
            "reduction_method": ["guided_autoencoder"],
            "dim_reduction": [12]
        }
    },
    
    # RUNNING ON CRAY-4
    "shot_noise": {
        "description": "Sweep over number of quantum shots",
        "param_grid": {
            "n_shots": [100, 500, 1000, 2000],
            "readout_type": ["all"],
            "reduction_method": ["guided_autoencoder"],
            "quantum_update_frequency": [5],
            "guided_lambda": [0.7], 
            "dim_reduction": [12]
        }
    },
    
    
    # RUNNING ON CRAY-4
    "time_steps_evolution_time": {
        "description": "Sweep over quantum evolution time and steps",
        "param_grid": {
            "evolution_time": [2.0, 4.0, 6.0],
            "time_steps": [8, 12, 16], # FIX WITH THE ONE THAT WORKED BEST IN THE PREVIOUS SWEEP
            "reduction_method": ["guided_autoencoder"],
            "dim_reduction": [12]
        }
    },
    
    
    # NOT DONE
    "full_dataset": {
        "description": "Compare performance with different dataset sizes",
        "param_grid": {
            "num_examples": [1000, 2000, 5000, 10000],
            "num_test_examples": [200, 400, 1000],
            "reduction_method": ["guided_autoencoder"],
            "dim_reduction": [12]
        }
    },
    
    # NOT DONE
    # RUN THIS ONE NEXT TO FIND THE LEARNING RATE AND REGULARIZATION
    "training_parameters": {
        "description": "Sweep over training hyperparameters",
        "param_grid": {
            "nepochs": [50, 100, 200],
            "learning_rate": [0.001, 0.01, 0.05],
            "regularization": [0.0, 0.0001, 0.001, 0.01],
            "reduction_method": ["guided_autoencoder"],
            "dim_reduction": [12],
        }
    },
    
    # MOST IMPORTANT FOR THE PAPER
    "ae_pca_qgars_5_runs": {
        "description": "Run a general test with fixed parameters",
        "param_grid": {
            "dataset_type": ["cvc_clinic_db_patches"],
            "num_examples": [2000],
            "num_test_examples": [400],
            "reduction_method": ["guided_autoencoder", "autoencoder", "pca"],
            "dim_reduction": [12],
            "seed": [42, 43, 44, 45, 46]
        }
    },
    
    "convolutional_ae": {
        "description": "Run a general test with fixed parameters",
        "param_grid": {
            "seed": [42, 43, 44, 45, 46],
            "dataset_type": ["cvc_clinic_db_patches"],
            "num_examples": [2000],
            "num_test_examples": [400],
            "reduction_method": ["guided_autoencoder", "autoencoder"],
            "ae_type": ["convolutional", "default"],
            "quantum_update_frequency": [5],
            "dim_reduction": [12]
        }
    },
    
    "test": {
        "description": "asdf",
        "param_grid": {
            "seed": [42],
            "dataset_type": ["generated_polyp_dataset"],
            "num_examples": [5],
            "num_test_examples": [2],
            "quantum_update_frequency": [25],
            "reduction_method": ["guided_autoencoder"],
            "dim_reduction": [4]
        }
    }
    
}

def run_experiment_profile(profile_name: str, 
                          base_config: Optional[Dict[str, Any]] = None, 
                          output_dir: Optional[str] = None,
                          parallel: bool = False,
                          n_workers: int = 4) -> pd.DataFrame:
    """
    Run an experiment profile.
    
    Parameters
    ----------
    profile_name : str
        Name of the experiment profile to run
    base_config : Optional[Dict[str, Any]], optional
        Base configuration to use, by default None
    output_dir : Optional[str], optional
        Directory to save results, by default None
    parallel : bool, optional
        Whether to run experiments in parallel, by default False
    n_workers : int, optional
        Number of workers for parallel execution, by default 4
        
    Returns
    -------
    pd.DataFrame
        Results dataframe
    """
    if profile_name not in EXPERIMENT_PROFILES:
        print(f"Error: Experiment profile '{profile_name}' not found. Available profiles:")
        for name, profile in EXPERIMENT_PROFILES.items():
            print(f"  - {name}: {profile['description']}")
        return None
    
    # Get profile
    profile = EXPERIMENT_PROFILES[profile_name]
    param_grid = profile["param_grid"]
    
    # Use default config if not provided
    if base_config is None:
        from src.utils.config_manager import get_config_args
        base_args = get_config_args()
        base_config = vars(base_args)
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join("results", "sweep", profile_name)
    
    # Configure sweep
    sweep = ParameterSweep(base_config, output_dir)
    
    # Run sweep
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{profile_name}_{timestamp}"
    
    print(f"Running experiment profile: {profile_name}")
    print(f"Description: {profile['description']}")
    print(f"Parameter grid: {param_grid}")
    print(f"Parallel execution: {parallel} with {n_workers} workers")
    
    # Run the sweep
    results = sweep.run_sweep(param_grid, experiment_name, run_parallel=parallel, n_workers=n_workers)
    
    return results

def main():
    """Main function to parse arguments and run experiments"""
    parser = argparse.ArgumentParser(description="Run parameter sweeps for QRC experiments")
    
    # Define subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Profile command
    profile_parser = subparsers.add_parser("profile", help="Run an experiment profile")
    profile_parser.add_argument("profile_name", type=str, choices=list(EXPERIMENT_PROFILES.keys()),
                              help="Name of the experiment profile to run")
    profile_parser.add_argument("--config", type=str, default=None,
                              help="Path to base configuration JSON file")
    profile_parser.add_argument("--output-dir", type=str, default=None,
                              help="Directory to save results")
    profile_parser.add_argument("--parallel", action="store_true",
                              help="Run experiments in parallel")
    profile_parser.add_argument("--workers", type=int, default=5,
                              help="Number of workers for parallel execution")
    
    # List profiles command
    list_parser = subparsers.add_parser("list", help="List available experiment profiles")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == "profile":
        # Load base config if provided
        base_config = None
        if args.config:
            base_config = ConfigManager.load_config(args.config)
        
        # Run experiment profile with parallel option
        run_experiment_profile(args.profile_name, base_config, args.output_dir, 
                              parallel=args.parallel, n_workers=args.workers)
        
    elif args.command == "list":
        # List available profiles
        print("Available experiment profiles:")
        for name, profile in EXPERIMENT_PROFILES.items():
            print(f"  - {name}: {profile['description']}")
            print(f"    Parameters: {list(profile['param_grid'].keys())}")
            print()
            
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
