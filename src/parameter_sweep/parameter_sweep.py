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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import main
from config_manager import ConfigManager

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
                 output_dir: str = "../results/parameter_sweep"):
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
        sweep_dir = os.path.join(self.output_dir, f"{experiment_name}_{self.timestamp}")
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
        
        # Mark this as a parameter sweep run to handle statistics properly
        setattr(args, '_parameter_sweep', True)
        
        # Create experiment directory
        exp_dir = os.path.join(sweep_dir, exp_id)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Run the experiment
        try:
            self.experiment_counter += 1
            print(f"\nExperiment {self.experiment_counter}: Running with {exp_id}\n")
            
            # Import statistics tracking here to avoid circular imports
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
            from statistics_tracking import save_all_statistics
            
            # Run main function with args
            results = main(args)
            
            # Save statistics in the experiment directory
            guided_losses = None
            
            # Check if guided autoencoder data is available in the global scope
            import __main__
            if hasattr(__main__, 'guided_autoencoder_losses') and __main__.guided_autoencoder_losses is not None:
                guided_losses = __main__.guided_autoencoder_losses
            
            # Save statistics
            save_all_statistics(results, guided_losses, exp_dir)
            
            # Extract metrics
            metrics = self._extract_metrics(results)
            
            # Add configuration parameters to metrics
            for key, value in config.items():
                # Skip large configurations
                if key not in ['autoencoder_hidden_dims', 'target_size']:
                    metrics[key] = value
                    
            # Save configuration
            with open(os.path.join(exp_dir, "config.json"), "w") as f:
                config_serializable = {k: str(v) if isinstance(v, np.ndarray) else v 
                                     for k, v in config.items()}
                json.dump(config_serializable, f, indent=4)
                
            # Save metrics
            with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)
            
            return metrics
            
        except Exception as e:
            print(f"Error in experiment {exp_id}: {str(e)}")
            # Return error metrics
            return {
                "error": str(e),
                "status": "failed",
                **{k: v for k, v in config.items() if k not in ['autoencoder_hidden_dims', 'target_size']}
            }
    
    def _extract_metrics(self, results: Dict[str, Tuple]) -> Dict[str, Any]:
        """
        Extract metrics from experiment results.
        
        Parameters
        ----------
        results : Dict[str, Tuple]
            Results dictionary from main function
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of metrics
        """
        metrics = {
            "status": "success",
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Extract final metrics from each model
        for model_name, (losses, accs_train, accs_test, _) in results.items():
            metrics[f"{model_name}_final_train_acc"] = float(accs_train[-1])
            metrics[f"{model_name}_final_test_acc"] = float(accs_test[-1])
            metrics[f"{model_name}_final_loss"] = float(losses[-1])
            
            # Calculate stability (std dev of last 10% of training)
            if len(accs_test) > 5:
                stability_window = max(1, int(len(accs_test) * 0.1))
                metrics[f"{model_name}_stability"] = float(np.std(accs_test[-stability_window:]))
        
        return metrics
