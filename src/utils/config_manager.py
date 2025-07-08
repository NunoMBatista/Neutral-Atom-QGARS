import os
import json
import argparse
import numpy as np
from typing import Dict, Any, Optional

from src.globals import DEFAULT_RESULTS_DIR

#DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "results")

class ConfigManager:
    """
    Manages configuration settings
    Allows loading from JSON configs and converting to argparse Namespace.
    """
    
    DEFAULT_CONFIG_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "default_config.json"
    )
    
    @staticmethod
    def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from a JSON file.
        
        Parameters
        ----------
        config_path : Optional[str], optional
            Path to config file, by default None which uses DEFAULT_CONFIG_PATH
            
        Returns
        -------
        Dict[str, Any]
            Configuration dictionary
        """
        if config_path is None:
            config_path = ConfigManager.DEFAULT_CONFIG_PATH
            
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        return config
    

    @staticmethod
    def config_to_args(config: Dict[str, Any]) -> argparse.Namespace:
        """
        Convert configuration dictionary to argparse Namespace.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary
            
        Returns
        -------
        argparse.Namespace
            Command line arguments
        """
        args = argparse.Namespace()
        for key, value in config.items():
            setattr(args, key, value)
        return args
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        Get default configuration values.
        
        Returns
        -------
        Dict[str, Any]
            Default configuration dictionary
        """
        return {
            "dataset_type": "cvc_clinic_db_patches",
            "data_dir": None,
            "target_size": [28, 28],
            "split_ratio": 0.8,
            "reduction_method": "autoencoder",
            "dim_reduction": 12,
            "num_examples": 10000,
            "num_test_examples": 400,
            "guided_lambda": 0.7,
            "quantum_update_frequency": 1,
            "guided_batch_size": 32,
            "autoencoder_epochs": 50,
            "autoencoder_batch_size": 64,
            "autoencoder_learning_rate": 0.001,
            "autoencoder_hidden_dims": None,
            "autoencoder_regularization": 1e-5,
            "gpu": False,
            "geometry": "chain",
            "lattice_spacing": 10.0,
            "rabi_freq": 2*np.pi,
            "evolution_time": 4.0,
            "time_steps": 16,
            "readout_type": "all",
            "n_shots": 1000,
            "detuning_max": 6.0,
            "encoding_scale": 9.0,
            "classifier_regularization": 0.0005,
            "nepochs": 100,
            "batchsize": 1000,
            "learning_rate": 0.01,
            "seed": 42,
            "no_progress": False,
            "no_plot": False,
            "ae_type": "convolutional",
            "results_dir": DEFAULT_RESULTS_DIR,
        }
