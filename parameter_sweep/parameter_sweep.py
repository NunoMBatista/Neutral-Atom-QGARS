import os
import numpy as np
import itertools
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm
import json
import datetime
import argparse
import sys

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


from main import main
from utils.config_manager import ConfigManager
from utils.statistics_tracking import save_all_statistics, extract_metrics  # Import the functions directly

class ParameterSweep:
    """
    Manages parameter sweeps for QRC experiments.
    
    Parameters
    ----------
    base_config : Dict[str, Any]
        Base configuration to use for all experiments
    output_dir : str
        Directory to save results
    """
    def __init__(self, 
                 base_config: Optional[Dict[str, Any]] = None, 
                 output_dir: str = DEFAULT_OUTPUT_DIR):
        """Initialize parameter sweep with base configuration"""
        self.base_config = base_config if base_config is not None else ConfigManager.get_default_config()
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate timestamp for this sweep
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_counter = 0
        
    def _create_experiment_configs(self, 
                                  param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Create experiment configurations from parameter grid.
        
        Parameters
        ----------
        param_grid : Dict[str, List[Any]]
            Dictionary mapping parameter names to lists of values
            
        Returns
        -------
        List[Dict[str, Any]]
            List of experiment configurations
        """
        # Get parameter names and values
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Generate all combinations
        configs = []
        for combo in itertools.product(*param_values):
            config = self.base_config.copy()
            # Update with parameter values
            for name, value in zip(param_names, combo):
                config[name] = value
            configs.append(config)
            
        return configs
    
    def run_sweep(self, 
                 param_grid: Dict[str, List[Any]], 
                 experiment_name: str,
                 run_parallel: bool = False,
                 n_workers: int = 4) -> pd.DataFrame:
        """
        Run parameter sweep experiments.
        
        Parameters
        ----------
        param_grid : Dict[str, List[Any]]
            Dictionary mapping parameter names to lists of values
        experiment_name : str
            Name for this experiment sweep
        run_parallel : bool, optional
            Whether to run experiments in parallel, by default False
        n_workers : int, optional
            Number of workers for parallel execution, by default 4
            
        Returns
        -------
        pd.DataFrame
            Results dataframe
        """
        # Create experiment configurations
        configs = self._create_experiment_configs(param_grid)
        
        print(f"Starting parameter sweep with {len(configs)} configurations")
        
        # Create directory for this sweep
        sweep_dir = os.path.join(self.output_dir, f"{experiment_name}")
        os.makedirs(sweep_dir, exist_ok=True)
        
        # Save parameter grid for reference
        with open(os.path.join(sweep_dir, "param_grid.json"), "w") as f:
            grid_serializable = {k: v if not isinstance(v, np.ndarray) else v.tolist() 
                                for k, v in param_grid.items()}
            json.dump(grid_serializable, f, indent=4)
        
        # Run experiments
        all_results = []
        
        if run_parallel and n_workers > 1:
            import concurrent.futures
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = []
                for i, config in enumerate(configs):
                    futures.append(executor.submit(self._run_single_experiment, 
                                                  config, sweep_dir, f"exp_{i}"))
                
                # Process results as they complete
                for future in tqdm(concurrent.futures.as_completed(futures), 
                                  total=len(futures), 
                                  desc="Running experiments"):
                    all_results.append(future.result())
        else:
            # Run sequentially
            for i, config in enumerate(tqdm(configs, desc="Running experiments")):
                result = self._run_single_experiment(config, sweep_dir, f"exp_{i}")
                all_results.append(result)
        
        # Compile results into dataframe
        results_df = pd.DataFrame(all_results)
        
        # Save compiled results
        results_df.to_csv(os.path.join(sweep_dir, "all_results.csv"), index=False)
        
        return results_df
    
    def _run_single_experiment(self, 
                              config: Dict[str, Any], 
                              sweep_dir: str,
                              exp_id: str) -> Dict[str, Any]:
        """
        Run a single experiment with the given configuration.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Experiment configuration
        sweep_dir : str
            Directory to save results
        exp_id : str
            Experiment identifier
            
        Returns
        -------
        Dict[str, Any]
            Experiment results with configuration parameters
        """
        # Convert config to argparse Namespace
        args = ConfigManager.config_to_args(config)
        
        # Mark this as a parameter sweep run to prevent duplicate saving
        setattr(args, '_parameter_sweep', True)
        
        # Run the experiment
        try:
            self.experiment_counter += 1
            print(f"\nExperiment {self.experiment_counter}: Running with {exp_id}\n")
            
            # Create experiment directory
            exp_dir = os.path.join(sweep_dir, exp_id)
            os.makedirs(exp_dir, exist_ok=True)
            
            # Run main function with args
            results, guided_losses = main(args, results_dir=exp_dir)
            
            # Save all statistics using the same function as main.py
            save_all_statistics(results, guided_losses, exp_dir, args)
            
            # Extract metrics using the common function
            metrics = extract_metrics(results)
            
            # Add configuration parameters to metrics for the DataFrame
            for key, value in config.items():
                # Skip large configurations
                if key not in ['autoencoder_hidden_dims', 'target_size']:
                    metrics[key] = value
            
            # Add guided_autoencoder metrics if available
            if guided_losses is not None:
                metrics["guided_autoencoder_final_total_loss"] = float(guided_losses["total_loss"][-1])
                metrics["guided_autoencoder_final_recon_loss"] = float(guided_losses["recon_loss"][-1])
                metrics["guided_autoencoder_final_class_loss"] = float(guided_losses["class_loss"][-1])
                
                # Add surrogate metrics if available
                if 'surrogate_loss' in guided_losses and any(x is not None for x in guided_losses['surrogate_loss']):
                    # Get the last valid surrogate loss
                    valid_surrogate_losses = [x for x in guided_losses['surrogate_loss'] if x is not None]
                    if valid_surrogate_losses:
                        metrics["surrogate_final_loss"] = float(valid_surrogate_losses[-1])
                        metrics["surrogate_min_loss"] = float(min(valid_surrogate_losses))
            
            return metrics
            
        except Exception as e:
            print(f"Error in experiment {exp_id}: {str(e)}")
            # Return error metrics
            return {
                "error": str(e),
                "status": "failed",
                **{k: v for k, v in config.items() if k not in ['autoencoder_hidden_dims', 'target_size']}
            }
