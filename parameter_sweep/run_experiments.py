import os
import argparse
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import sys

# Make the repository root importable when this file is run as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parameter_sweep import ParameterSweep
from src.utils.config_manager import ConfigManager

# Sweep profiles behind the paper's results. Each grid entry overrides the
# defaults in src/utils/config_manager.py; every combination is one run.
EXPERIMENT_PROFILES = {
    # Picked the autoencoder weight decay used everywhere else (1e-5)
    "autoencoder_regularization": {
        "description": "Sweep over autoencoder regularization parameters",
        "param_grid": {
            "reduction_method": ["guided_autoencoder"],
            "autoencoder_regularization": [0.0, 0.00001, 0.0001, 0.001, 0.01],
            "quantum_update_frequency": [5],
            "dim_reduction": [12]
        }
    },

    # Paper figure: accuracy against lambda, log-spaced near 0 and 1 (picked 0.7)
    "guided_autoencoder_lambda": {
        "description": "Sweep over guided autoencoder lambda parameter",
        "param_grid": {
            "reduction_method": ["guided_autoencoder"],
            "guided_lambda": [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.0,
                              0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                              1-0.01, 1-0.001, 1-0.0001, 1-0.00001, 1-0.000001, 1],
            "quantum_update_frequency": [5],
            "dim_reduction": [12]
        }
    },

    # Paper figure: accuracy against the number of atoms (= latent dimension)
    "encoding_dimensions": {
        "description": "Sweep over encoding dimensions and methods",
        "param_grid": {
            "dim_reduction": [4, 6, 8, 10, 12],
            "reduction_method": ["pca", "autoencoder", "guided_autoencoder"],
            "n_shots": [1000],
            "autoencoder_regularization": [1e-5],
            "guided_lambda": [0.7],
            "quantum_update_frequency": [1]
        }
    },

    # How often the surrogate is refreshed from the reservoir (0 = never, a plain autoencoder)
    "guided_autoencoder_update_frequency": {
        "description": "Sweep over guided autoencoder update frequency",
        "param_grid": {
            "reduction_method": ["guided_autoencoder"],
            "guided_lambda": [0.7],
            "quantum_update_frequency": [1, 3, 5, 7, 10, 15, 25, 0],
            "dim_reduction": [12],
            "autoencoder_regularization": [1e-5]
        }
    },

    # Paper tables I and II: five seeds per method (set dataset_type for each dataset)
    "ae_pca_qgars_5_runs": {
        "description": "Five seeds of PCA, autoencoder and QGARS",
        "param_grid": {
            "dataset_type": ["cvc_clinic_db_patches"],
            "num_examples": [2000],
            "num_test_examples": [400],
            "reduction_method": ["guided_autoencoder", "autoencoder", "pca"],
            "dim_reduction": [12],
            "seed": [42, 43, 44, 45, 46]
        }
    },

    "test": {
        "description": "Tiny run to check the pipeline end to end",
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
