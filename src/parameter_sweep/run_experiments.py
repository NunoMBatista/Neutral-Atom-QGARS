import os
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parameter_sweep import ParameterSweep
from config_manager import ConfigManager

# Define experiment profiles with parameter ranges
EXPERIMENT_PROFILES = {
    "encoding_dimensions": {
        "description": "Sweep over encoding dimensions and methods",
        "param_grid": {
            "dim_reduction": [4, 6, 8, 10, 12],
            "reduction_method": ["pca", "autoencoder", "guided_autoencoder"],
            "n_shots": [500]
        }
    },
    "quantum_readouts": {
        "description": "Sweep over quantum readout types and time steps",
        "param_grid": {
            "readout_type": ["Z", "ZZ", "all"],
            "time_steps": [4, 8, 12, 16],
            "n_shots": [500],
            "reduction_method": ["guided_autoencoder"],
            "dim_reduction": [12]
        }
    },
    "shot_noise": {
        "description": "Sweep over number of quantum shots",
        "param_grid": {
            "n_shots": [100, 500, 1000, 2000],
            "readout_type": ["all"],
            "reduction_method": ["guided_autoencoder"],
            "dim_reduction": [12]
        }
    },
    "guided_autoencoder": {
        "description": "Sweep over guided autoencoder parameters",
        "param_grid": {
            "reduction_method": ["guided_autoencoder"],
            "guided_alpha": [0.3, 0.5, 0.7],
            "guided_beta": [0.7, 0.5, 0.3],
            "quantum_update_frequency": [1, 3, 5, 10],
            "dim_reduction": [12]
        }
    },
    "evolution_time": {
        "description": "Sweep over quantum evolution time and steps",
        "param_grid": {
            "evolution_time": [2.0, 4.0, 6.0],
            "time_steps": [8, 12, 16],
            "reduction_method": ["guided_autoencoder"],
            "dim_reduction": [12]
        }
    },
    "encoding_scale": {
        "description": "Sweep over encoding scale parameter",
        "param_grid": {
            "encoding_scale": [3.0, 6.0, 9.0, 12.0],
            "detuning_max": [3.0, 6.0, 9.0],
            "reduction_method": ["guided_autoencoder"],
            "dim_reduction": [12]
        }
    },
    "full_dataset": {
        "description": "Compare performance with different dataset sizes",
        "param_grid": {
            "num_examples": [1000, 2000, 5000, 10000],
            "num_test_examples": [200, 400, 1000],
            "reduction_method": ["guided_autoencoder"],
            "dim_reduction": [12]
        }
    },
    "training_parameters": {
        "description": "Sweep over training hyperparameters",
        "param_grid": {
            "nepochs": [50, 100, 200],
            "learning_rate": [0.001, 0.01, 0.05],
            "regularization": [0.0, 0.0001, 0.001, 0.01],
            "reduction_method": ["guided_autoencoder"],
            "dim_reduction": [12]
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
        from config_manager import get_config_args
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
    profile_parser.add_argument("--workers", type=int, default=4,
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
