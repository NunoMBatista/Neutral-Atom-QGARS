import os
import sys
import json
import time
import itertools
import argparse
import numpy as np
import random  # Add import for Python's standard random module
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import uuid
import concurrent.futures
import multiprocessing

# Add both the parent directory and the project root to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)  # Add the parent directory to Python path
project_root = os.path.dirname(parent_dir)
sys.path.insert(0, project_root)  # Add the project root directory

# Use absolute imports from the project
from qrc_polyp_python.main import main
from qrc_polyp_python.cli_utils import AVAILABLE_READOUT_TYPES

# Create results directory if it doesn't exist
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Files to store the best results by category
BEST_RESULTS_FILE = os.path.join(RESULTS_DIR, "best_results.json")
BEST_QRC_RESULTS_FILE = os.path.join(RESULTS_DIR, "best_qrc_results.json")
BEST_AUTOENCODER_RESULTS_FILE = os.path.join(RESULTS_DIR, "best_autoencoder_results.json")
BEST_GUIDED_AUTOENCODER_RESULTS_FILE = os.path.join(RESULTS_DIR, "best_guided_autoencoder_results.json")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the benchmarking script."""
    parser = argparse.ArgumentParser(description="Benchmark QRC with different parameter combinations")
    
    # Test configuration
    parser.add_argument("--num-combinations", type=int, default=10,
                        help="Number of random parameter combinations to test")
    parser.add_argument("--random-search", action="store_true",
                        help="Use random search instead of grid search")
    parser.add_argument("--test-name", type=str, default=None,
                        help="Name for this test run (default: timestamp)")
    
    # Parallel processing parameters
    parser.add_argument("--parallel", action="store_true",
                        help="Run parameter tests in parallel")
    parser.add_argument("--n-processes", type=int, default=None,
                        help="Number of parallel processes to use (default: number of CPU cores)")
    
    # Parameter ranges for testing
    parser.add_argument("--dim-pca-range", type=int, nargs=3, default=[6, 12, 2],
                        help="Range for PCA dimensions [min, max, step]")
    parser.add_argument("--rabi-freq-range", type=float, nargs=3, 
                        default=[4.0, 8.0, 2.0],
                        help="Range for Rabi frequencies [min, max, step]")
    parser.add_argument("--time-steps-range", type=int, nargs=3, default=[4, 16, 4],
                        help="Range for time steps [min, max, step]")
    parser.add_argument("--readout-types", type=str, nargs="+", 
                        default=AVAILABLE_READOUT_TYPES,
                        help="Readout types to test")
    
    parser.add_argument("--dataset-types", type=str, nargs="+", 
                       default=["cvc_clinic_db_patches"],
                       help="Dataset types to test")
    
    # Reduction method parameter
    parser.add_argument("--reduction-methods", type=str, nargs="+", 
                       default=["pca", "autoencoder", "guided_autoencoder"],
                       help="Feature reduction methods to test")
    
    # Autoencoder specific parameters
    parser.add_argument("--ae-hidden-dims", type=str, nargs="+", 
                       default=["[256,128]", "[512,256,128]", "[1024,512,256]"],
                       help="Hidden layer configurations for autoencoder as JSON strings")
    parser.add_argument("--ae-learning-rate-range", type=float, nargs=3, 
                       default=[0.0001, 0.01, 10],  # Using multiplier for log scale
                       help="Range for autoencoder learning rates [min, max, multiplier]")
    parser.add_argument("--ae-batch-size-range", type=int, nargs=3, 
                       default=[32, 256, 2],  # Using multiplier for powers of 2
                       help="Range for autoencoder batch sizes [min, max, multiplier]")
    parser.add_argument("--ae-epochs-range", type=int, nargs=3, 
                       default=[20, 100, 20],
                       help="Range for autoencoder training epochs [min, max, step]")
    parser.add_argument("--ae-dropout-range", type=float, nargs=3, 
                       default=[0.0, 0.5, 0.1],
                       help="Range for autoencoder dropout rates [min, max, step]")
    parser.add_argument("--batch-norm-options", type=str, nargs="+", 
                       default=["True", "False"],
                       help="Whether to use batch normalization in autoencoder")
    
    # Guided autoencoder specific parameters
    parser.add_argument("--guided-alpha-range", type=float, nargs=3, 
                       default=[0.3, 0.9, 0.2],
                       help="Range for guided autoencoder alpha values [min, max, step]. Beta will be calculated as (1-alpha).")
    parser.add_argument("--guided-beta-range", type=float, nargs=3, 
                       default=[0.1, 0.7, 0.2],
                       help="DEPRECATED: Use guided-alpha-range instead. Beta will be calculated as (1-alpha).")
    parser.add_argument("--quantum-update-freq-range", type=int, nargs=3, 
                       default=[1, 10, 3],
                       help="Range for quantum update frequency [min, max, step]")
    
    # Classifier specific parameters
    parser.add_argument("--learning-rate-range", type=float, nargs=3, 
                       default=[0.001, 0.1, 10],  # Using multiplier for log scale
                       help="Range for classifier learning rates [min, max, multiplier]")
    parser.add_argument("--regularization-range", type=float, nargs=3, 
                       default=[0.0, 0.01, 10],  # Using multiplier for log scale
                       help="Range for classifier regularization [min, max, multiplier]")
    
    # Constraints for faster testing
    parser.add_argument("--num-examples", type=int, default=100,
                        help="Number of examples to use for training")
    parser.add_argument("--num-test-examples", type=int, default=50,
                        help="Number of examples to use for testing")
    parser.add_argument("--nepochs", type=int, default=20,
                        help="Number of training epochs")
    

    
    return parser.parse_args()

def convert_numpy_types(obj):
    """
    Convert NumPy types to standard Python types for JSON serialization.
    
    Parameters
    ----------
    obj : Any
        Object to convert
        
    Returns
    -------
    Any
        Object with NumPy types converted to standard Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):  # Add handling for numpy boolean type
        return bool(obj)
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

def create_parameter_grid(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """
    Create a grid of parameter combinations to test.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
        
    Returns
    -------
    List[Dict[str, Any]]
        List of parameter dictionaries to test
    """
    # Define parameter ranges
    param_grid = {
        "geometry": ["chain"],  # Fixed to chain only
        "dim_reduction": list(range(args.dim_pca_range[0], args.dim_pca_range[1] + 1, args.dim_pca_range[2])),
        "rabi_freq": list(np.arange(args.rabi_freq_range[0], args.rabi_freq_range[1] + 0.01, args.rabi_freq_range[2])),
        "time_steps": list(range(args.time_steps_range[0], args.time_steps_range[1] + 1, args.time_steps_range[2])),
        "readout_type": args.readout_types,
        
        # NEW: Add dataset type options
        "dataset_type": args.dataset_types,
        
        # Add reduction method options
        "reduction_method": args.reduction_methods,
        
        # Autoencoder parameters
        "autoencoder_hidden_dims": [eval(dims) for dims in args.ae_hidden_dims],
        "autoencoder_learning_rate": [args.ae_learning_rate_range[0] * (args.ae_learning_rate_range[2] ** i) 
                                     for i in range(int(np.log(args.ae_learning_rate_range[1]/args.ae_learning_rate_range[0]) / 
                                                     np.log(args.ae_learning_rate_range[2])) + 1)],
        "autoencoder_batch_size": [args.ae_batch_size_range[0] * (args.ae_batch_size_range[2] ** i) 
                                  for i in range(int(np.log(args.ae_batch_size_range[1]/args.ae_batch_size_range[0]) / 
                                                  np.log(args.ae_batch_size_range[2])) + 1)],
        "autoencoder_epochs": list(range(args.ae_epochs_range[0], args.ae_epochs_range[1] + 1, args.ae_epochs_range[2])),
        "dropout": list(np.arange(args.ae_dropout_range[0], args.ae_dropout_range[1] + 0.01, args.ae_dropout_range[2])),
        "use_batch_norm": [eval(bn) for bn in args.batch_norm_options],
        
        # Guided autoencoder parameters
        "guided_alpha": list(np.arange(args.guided_alpha_range[0], args.guided_alpha_range[1] + 0.01, args.guided_alpha_range[2])),
        # Beta is no longer a separately chosen parameter
        "quantum_update_frequency": list(range(args.quantum_update_freq_range[0], 
                                              args.quantum_update_freq_range[1] + 1, 
                                              args.quantum_update_freq_range[2])),
        
        # Classifier parameters
        "learning_rate": [args.learning_rate_range[0] * (args.learning_rate_range[2] ** i) 
                         for i in range(int(np.log(args.learning_rate_range[1]/args.learning_rate_range[0]) / 
                                         np.log(args.learning_rate_range[2])) + 1)],
    }
    
    # Handle regularization separately to avoid division by zero
    if args.regularization_range[0] == 0.0:
        # If min regularization is 0, start with 0 and then add log-spaced values if max > 0
        reg_values = [0.0]
        if args.regularization_range[1] > 0:
            # Start from a small value instead of 0 for log spacing
            min_reg = max(1e-5, args.regularization_range[0])
            reg_values.extend([
                args.regularization_range[2] ** i 
                for i in range(int(np.log(args.regularization_range[1]/min_reg) / 
                                  np.log(args.regularization_range[2])) + 1)
            ])
    else:
        # Original calculation if min regularization is not 0
        reg_values = [args.regularization_range[0] * (args.regularization_range[2] ** i) 
                     for i in range(int(np.log(args.regularization_range[1]/args.regularization_range[0]) / 
                                     np.log(args.regularization_range[2])) + 1)]
    
    # Add regularization values to parameter grid
    param_grid["regularization"] = reg_values
    
    # Create combinations
    if args.random_search:
        # Random search
        combinations = []
        for _ in range(args.num_combinations):
            params = {
                "geometry": "chain",  # Fixed to chain
                "dim_reduction": np.random.choice(param_grid["dim_reduction"]),
                "rabi_freq": np.random.choice(param_grid["rabi_freq"]),
                "time_steps": np.random.choice(param_grid["time_steps"]),
                "readout_type": np.random.choice(param_grid["readout_type"]),
                
                # NEW: Randomly select dataset type
                "dataset_type": np.random.choice(param_grid["dataset_type"]),
                
                # Randomly select reduction method
                "reduction_method": np.random.choice(param_grid["reduction_method"]),
            }
            
            # Add autoencoder-specific parameters based on reduction method
            if params["reduction_method"] in ["autoencoder", "guided_autoencoder"]:
                params.update({
                    # Use random.choice() for nested list parameters instead of np.random.choice()
                    "autoencoder_hidden_dims": random.choice(param_grid["autoencoder_hidden_dims"]),
                    "autoencoder_learning_rate": np.random.choice(param_grid["autoencoder_learning_rate"]),
                    "autoencoder_batch_size": np.random.choice(param_grid["autoencoder_batch_size"]),
                    "autoencoder_epochs": np.random.choice(param_grid["autoencoder_epochs"]),
                    "dropout": np.random.choice(param_grid["dropout"]),
                    "use_batch_norm": np.random.choice(param_grid["use_batch_norm"]),
                })
            
            # Add guided autoencoder-specific parameters if applicable
            if params["reduction_method"] == "guided_autoencoder":
                alpha = np.random.choice(param_grid["guided_alpha"])
                params.update({
                    "guided_alpha": alpha,
                    "guided_beta": round(1.0 - alpha, 10),  # Ensure precision doesn't cause issues
                    "quantum_update_frequency": np.random.choice(param_grid["quantum_update_frequency"]),
                })
            
            combinations.append(params)
    else:
        # For grid search, we need to be careful not to create too many combinations
        # We'll handle reduction method separately to avoid combinatorial explosion
        base_param_grid = {
            "dim_reduction": param_grid["dim_reduction"],
            "rabi_freq": param_grid["rabi_freq"],
            "time_steps": param_grid["time_steps"],
            "readout_type": param_grid["readout_type"],
            # NEW: Add dataset type to base parameters
            "dataset_type": param_grid["dataset_type"],
            "learning_rate": param_grid["learning_rate"][:2],  # Limit options to prevent explosion
            "regularization": param_grid["regularization"][:2],  # Limit options to prevent explosion
        }
        
        # Create base combinations for quantum parameters (without reduction method specifics)
        keys = list(base_param_grid.keys())
        values = list(base_param_grid.values())
        base_combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
        
        # Add fixed chain geometry to all combinations
        for combo in base_combinations:
            combo["geometry"] = "chain"
        
        # Limit number of base combinations if there are too many
        max_base = min(20, len(base_combinations))
        if len(base_combinations) > max_base:
            np.random.shuffle(base_combinations)
            base_combinations = base_combinations[:max_base]
        
        # Now create final combinations with reduction method specifics
        combinations = []
        
        # For each base combination, create variants with different reduction methods
        for base_combo in base_combinations:
            for reduction_method in param_grid["reduction_method"]:
                combo = base_combo.copy()
                combo["reduction_method"] = reduction_method
                
                # Add autoencoder-specific parameters for autoencoder methods
                if reduction_method in ["autoencoder", "guided_autoencoder"]:
                    # Just pick a few combinations for these to avoid explosion
                    for hidden_dims in param_grid["autoencoder_hidden_dims"][:2]:  # Limit to first 2
                        for learning_rate in param_grid["autoencoder_learning_rate"][:2]:  # Limit to first 2
                            for use_batch_norm in param_grid["use_batch_norm"]:
                                ae_combo = combo.copy()
                                ae_combo.update({
                                    "autoencoder_hidden_dims": hidden_dims,
                                    "autoencoder_learning_rate": learning_rate,
                                    "autoencoder_batch_size": param_grid["autoencoder_batch_size"][1],  # Middle value
                                    "autoencoder_epochs": param_grid["autoencoder_epochs"][0],  # First value
                                    "dropout": param_grid["dropout"][1],  # Middle value
                                    "use_batch_norm": use_batch_norm,
                                })
                                
                                # Add guided autoencoder-specific parameters if applicable
                                if reduction_method == "guided_autoencoder":
                                    for alpha in param_grid["guided_alpha"][:2]:  # Limit to first 2
                                        guided_combo = ae_combo.copy()
                                        guided_combo.update({
                                            "guided_alpha": alpha,
                                            "guided_beta": round(1.0 - alpha, 10),  # Ensure precision doesn't cause issues
                                            "quantum_update_frequency": param_grid["quantum_update_frequency"][0],  # First value
                                        })
                                        combinations.append(guided_combo)
                                else:
                                    combinations.append(ae_combo)
                else:
                    # For PCA, just use the base combination
                    combinations.append(combo)
        
        # Limit number of combinations if specified
        if args.num_combinations and args.num_combinations < len(combinations):
            np.random.shuffle(combinations)
            combinations = combinations[:args.num_combinations]
    
    # Add fixed parameters to all combinations
    for params in combinations:
        params.update({
            "num_examples": args.num_examples,
            "num_test_examples": args.num_test_examples,
            "nepochs": args.nepochs,
            "no_progress": False,  # Disable progress bars for benchmarking
            "no_plot": True,      # Disable plotting for benchmarking
        })
    
    return combinations

def run_parameter_test(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a test with a specific parameter combination.
    
    Parameters
    ----------
    params : Dict[str, Any]
        Parameters to test
        
    Returns
    -------
    Dict[str, Any]
        Results and parameters of the test
    """
    # Validate that guided_alpha and guided_beta sum to 1
    if "reduction_method" in params and params["reduction_method"] == "guided_autoencoder":
        if abs(params["guided_alpha"] + params["guided_beta"] - 1.0) > 1e-9:
            # Correct beta value to ensure sum is exactly 1
            params["guided_beta"] = round(1.0 - params["guided_alpha"], 10)
            print(f"Adjusted guided_beta to {params['guided_beta']} to ensure guided_alpha + guided_beta = 1")
    
    class Args:
        pass
    
    # Convert dictionary to Namespace object for main
    args = Args()
    for key, value in params.items():
        setattr(args, key, value)
    
    # Set defaults for any missing parameters
    if not hasattr(args, "seed"):
        args.seed = 42
    if not hasattr(args, "data_dir"):
        args.data_dir = None
    if not hasattr(args, "target_size"):
        args.target_size = [128, 128]
    if not hasattr(args, "split_ratio"):
        args.split_ratio = 0.8
    if not hasattr(args, "lattice_spacing"):
        args.lattice_spacing = 10.0
    if not hasattr(args, "evolution_time"):
        args.evolution_time = 4.0
    if not hasattr(args, "n_shots"):
        args.n_shots = 1000
    if not hasattr(args, "detuning_max"):
        args.detuning_max = 6.0
    if not hasattr(args, "encoding_scale"):
        args.encoding_scale = 9.0
    if not hasattr(args, "gpu"):
        args.gpu = False
    
    # Set defaults for autoencoder parameters if not present
    if not hasattr(args, "autoencoder_batch_size") and hasattr(args, "reduction_method") and args.reduction_method in ["autoencoder", "guided_autoencoder"]:
        args.autoencoder_batch_size = 64
    if not hasattr(args, "autoencoder_epochs") and hasattr(args, "reduction_method") and args.reduction_method in ["autoencoder", "guided_autoencoder"]:
        args.autoencoder_epochs = 30
    if not hasattr(args, "autoencoder_learning_rate") and hasattr(args, "reduction_method") and args.reduction_method in ["autoencoder", "guided_autoencoder"]:
        args.autoencoder_learning_rate = 0.001
    
    # Set defaults for guided autoencoder parameters if not present
    if not hasattr(args, "guided_alpha") and hasattr(args, "reduction_method") and args.reduction_method == "guided_autoencoder":
        args.guided_alpha = 0.7
    if not hasattr(args, "guided_beta") and hasattr(args, "reduction_method") and args.reduction_method == "guided_autoencoder":
        args.guided_beta = 0.3
    if not hasattr(args, "guided_batch_size") and hasattr(args, "reduction_method") and args.reduction_method == "guided_autoencoder":
        args.guided_batch_size = 32
    if not hasattr(args, "quantum_update_frequency") and hasattr(args, "reduction_method") and args.reduction_method == "guided_autoencoder":
        args.quantum_update_frequency = 5
    
    if not hasattr(args, "batchsize"):
        args.batchsize = 100
    
    # Run the main function with these parameters
    start_time = time.time()
    try:
        results = main(args)
        
        # Extract final test accuracies
        test_accuracies = {
            model_name: accs_test[-1] * 100 
            for model_name, (_, _, accs_test, _) in results.items()
        }
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Prepare results dictionary
        result_dict = {
            "parameters": params,
            "test_accuracies": test_accuracies,
            "qrc_accuracy": test_accuracies.get("QRC", 0),
            "best_model": max(test_accuracies.items(), key=lambda x: x[1])[0],
            "best_accuracy": max(test_accuracies.values()),
            "execution_time": execution_time,
            "status": "success"
        }
        
    except Exception as e:
        # Log error information if the test fails
        execution_time = time.time() - start_time
        result_dict = {
            "parameters": params,
            "error": str(e),
            "execution_time": execution_time,
            "status": "error",
            "test_accuracies": {},
            "qrc_accuracy": 0
        }
    
    return result_dict

def save_results(result: Dict[str, Any], args: argparse.Namespace) -> str:
    """
    Save test results to a JSON file.
    
    Parameters
    ----------
    result : Dict[str, Any]
        Test results to save
    args : argparse.Namespace
        Command line arguments
        
    Returns
    -------
    str
        Path to the saved file
    """
    # Generate timestamp and unique identifier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = str(uuid.uuid4())[:8]
    
    # Create filename
    test_name = args.test_name or timestamp
    filename = f"{test_name}_{uid}.json"
    filepath = os.path.join(RESULTS_DIR, filename)
    
    # Add metadata
    result["metadata"] = {
        "timestamp": timestamp,
        "test_name": test_name,
        "id": uid
    }
    
    # Convert NumPy types to standard Python types for JSON serialization
    result_converted = convert_numpy_types(result)
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(result_converted, f, indent=2)
    
    return filepath

def update_best_results(result: Dict[str, Any]) -> Dict[str, bool]:
    """
    Update different best result files based on the current test result.
    
    Parameters
    ----------
    result : Dict[str, Any]
        Current test result
        
    Returns
    -------
    Dict[str, bool]
        Dictionary indicating whether the result was the new best for each category
    """
    # Initialize results
    is_new_best = {
        "overall": False,
        "qrc": False,
        "autoencoder": False,
        "guided_autoencoder": False
    }
    
    # Convert NumPy types to standard Python types for consistency
    result_converted = convert_numpy_types(result)
    
    # Update overall best results (based on QRC accuracy)
    if os.path.exists(BEST_RESULTS_FILE):
        with open(BEST_RESULTS_FILE, 'r') as f:
            try:
                best_results = json.load(f)
                best_qrc_accuracy = best_results.get("qrc_accuracy", 0)
            except json.JSONDecodeError:
                best_qrc_accuracy = 0
    else:
        best_qrc_accuracy = 0
    
    # Check if current QRC result is better for overall
    current_qrc_accuracy = result["qrc_accuracy"]
    if current_qrc_accuracy > best_qrc_accuracy:
        is_new_best["overall"] = True
        with open(BEST_RESULTS_FILE, 'w') as f:
            json.dump(result_converted, f, indent=2)
    
    # Update best QRC results (also based on QRC accuracy)
    if os.path.exists(BEST_QRC_RESULTS_FILE):
        with open(BEST_QRC_RESULTS_FILE, 'r') as f:
            try:
                best_qrc_results = json.load(f)
                best_qrc_specific_accuracy = best_qrc_results.get("qrc_accuracy", 0)
            except json.JSONDecodeError:
                best_qrc_specific_accuracy = 0
    else:
        best_qrc_specific_accuracy = 0
    
    # Check if current QRC result is better for QRC specifically
    if current_qrc_accuracy > best_qrc_specific_accuracy:
        is_new_best["qrc"] = True
        with open(BEST_QRC_RESULTS_FILE, 'w') as f:
            json.dump(result_converted, f, indent=2)
    
    # Update best results for specific reduction methods
    reduction_method = result["parameters"].get("reduction_method", "")
    
    # For autoencoder method
    if reduction_method == "autoencoder":
        if os.path.exists(BEST_AUTOENCODER_RESULTS_FILE):
            with open(BEST_AUTOENCODER_RESULTS_FILE, 'r') as f:
                try:
                    best_ae_results = json.load(f)
                    best_ae_accuracy = best_ae_results.get("qrc_accuracy", 0)
                except json.JSONDecodeError:
                    best_ae_accuracy = 0
        else:
            best_ae_accuracy = 0
        
        if current_qrc_accuracy > best_ae_accuracy:
            is_new_best["autoencoder"] = True
            with open(BEST_AUTOENCODER_RESULTS_FILE, 'w') as f:
                json.dump(result_converted, f, indent=2)
    
    # For guided autoencoder method
    if reduction_method == "guided_autoencoder":
        if os.path.exists(BEST_GUIDED_AUTOENCODER_RESULTS_FILE):
            with open(BEST_GUIDED_AUTOENCODER_RESULTS_FILE, 'r') as f:
                try:
                    best_guided_ae_results = json.load(f)
                    best_guided_ae_accuracy = best_guided_ae_results.get("qrc_accuracy", 0)
                except json.JSONDecodeError:
                    best_guided_ae_accuracy = 0
        else:
            best_guided_ae_accuracy = 0
        
        if current_qrc_accuracy > best_guided_ae_accuracy:
            is_new_best["guided_autoencoder"] = True
            with open(BEST_GUIDED_AUTOENCODER_RESULTS_FILE, 'w') as f:
                json.dump(result_converted, f, indent=2)
    
    return is_new_best

def run_parameter_test_wrapper(params_and_args):
    """
    Wrapper function for parallel processing of parameter tests.
    
    Parameters
    ----------
    params_and_args : tuple
        Tuple containing (params, args, test_index, total_tests)
        
    Returns
    -------
    Dict[str, Any]
        Results and parameters of the test
    """
    params, args, test_index, total_tests = params_and_args
    print(f"\nTest {test_index}/{total_tests}")
    print(f"Parameters: {params}")
    
    try:
        # Run test
        result = run_parameter_test(params)
        
        # Save individual result
        result_file = save_results(result, args)
        print(f"Results saved to: {result_file}")
        
        # Check if it's the best result in any category
        is_best_results = update_best_results(result)
        if is_best_results["overall"]:
            print(f"✓ New best overall QRC result! Accuracy: {result['qrc_accuracy']:.2f}%")
        if is_best_results["qrc"]:
            print(f"✓ New best QRC-specific result! Accuracy: {result['qrc_accuracy']:.2f}%")
        if is_best_results["autoencoder"]:
            print(f"✓ New best autoencoder result! Accuracy: {result['qrc_accuracy']:.2f}%")
        if is_best_results["guided_autoencoder"]:
            print(f"✓ New best guided autoencoder result! Accuracy: {result['qrc_accuracy']:.2f}%")
        
        # Print summary
        print(f"Test accuracies: {result['test_accuracies']}")
        print(f"QRC model accuracy: {result['qrc_accuracy']:.2f}%")
        print(f"Execution time: {result['execution_time']:.2f} seconds")
        
        return result
        
    except Exception as e:
        print(f"Error in test {test_index}: {e}")
        return {
            "parameters": params,
            "error": str(e),
            "status": "error",
            "test_accuracies": {},
            "qrc_accuracy": 0
        }

def run_benchmark(args: argparse.Namespace) -> None:
    """
    Run benchmark tests with different parameter combinations.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    """
    search_type = 'random' if args.random_search else 'grid'
    print(f"Starting benchmark with {search_type} search")
    
    # Create parameter combinations
    param_combinations = create_parameter_grid(args)
    print(f"Testing {len(param_combinations)} parameter combinations")
    
    # Determine if using parallel execution
    if args.parallel:
        # Determine number of processes
        n_processes = args.n_processes or multiprocessing.cpu_count()
        n_processes = min(n_processes, len(param_combinations))
        
        print(f"Running tests in parallel with {n_processes} processes")
        
        # Prepare parameters for parallel execution
        params_and_args = [
            (params, args, i+1, len(param_combinations)) 
            for i, params in enumerate(param_combinations)
        ]
        
        # Run tests in parallel
        all_results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_processes) as executor:
            # Submit all tasks and get futures
            futures = [executor.submit(run_parameter_test_wrapper, p_and_a) 
                      for p_and_a in params_and_args]
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    print(f"Error in worker process: {e}")
    else:
        # Run tests sequentially
        all_results = []
        for i, params in enumerate(param_combinations):
            print(f"\nTest {i+1}/{len(param_combinations)}")
            print(f"Parameters: {params}")
            
            try:
                # Run test
                result = run_parameter_test(params)
                
                # Save individual result
                result_file = save_results(result, args)
                print(f"Results saved to: {result_file}")
                
                # Check if it's the best result in any category
                is_best_results = update_best_results(result)
                if is_best_results["overall"]:
                    print(f"✓ New best overall QRC result! Accuracy: {result['qrc_accuracy']:.2f}%")
                if is_best_results["qrc"]:
                    print(f"✓ New best QRC-specific result! Accuracy: {result['qrc_accuracy']:.2f}%")
                if is_best_results["autoencoder"]:
                    print(f"✓ New best autoencoder result! Accuracy: {result['qrc_accuracy']:.2f}%")
                if is_best_results["guided_autoencoder"]:
                    print(f"✓ New best guided autoencoder result! Accuracy: {result['qrc_accuracy']:.2f}%")
                
                # Add to all results
                all_results.append(result)
                
                # Print summary
                print(f"Test accuracies: {result['test_accuracies']}")
                print(f"QRC model accuracy: {result['qrc_accuracy']:.2f}%")
                print(f"Execution time: {result['execution_time']:.2f} seconds")
                
            except Exception as e:
                print(f"Error in test: {e}")
    
    # Save all results together
    all_results_file = os.path.join(RESULTS_DIR, f"all_results_{args.test_name or datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    # Convert NumPy types to standard Python types
    all_results_converted = convert_numpy_types(all_results)
    
    with open(all_results_file, 'w') as f:
        json.dump(all_results_converted, f, indent=2)
    print(f"\nAll results saved to: {all_results_file}")
    
    # Print best results for each category
    print("\nBest Results by Category:")

    if os.path.exists(BEST_RESULTS_FILE):
        with open(BEST_RESULTS_FILE, 'r') as f:
            best_results = json.load(f)
        print("\nBest Overall Result:")
        print(f"QRC model accuracy: {best_results['qrc_accuracy']:.2f}%")
        print(f"Parameters: {best_results['parameters']}")

    if os.path.exists(BEST_QRC_RESULTS_FILE):
        with open(BEST_QRC_RESULTS_FILE, 'r') as f:
            best_qrc_results = json.load(f)
        print("\nBest QRC-Specific Result:")
        print(f"QRC model accuracy: {best_qrc_results['qrc_accuracy']:.2f}%")
        print(f"Parameters: {best_qrc_results['parameters']}")

    if os.path.exists(BEST_AUTOENCODER_RESULTS_FILE):
        with open(BEST_AUTOENCODER_RESULTS_FILE, 'r') as f:
            best_ae_results = json.load(f)
        print("\nBest Autoencoder Result:")
        print(f"QRC model accuracy: {best_ae_results['qrc_accuracy']:.2f}%")
        print(f"Parameters: {best_ae_results['parameters']}")

    if os.path.exists(BEST_GUIDED_AUTOENCODER_RESULTS_FILE):
        with open(BEST_GUIDED_AUTOENCODER_RESULTS_FILE, 'r') as f:
            best_guided_ae_results = json.load(f)
        print("\nBest Guided Autoencoder Result:")
        print(f"QRC model accuracy: {best_guided_ae_results['qrc_accuracy']:.2f}%")
        print(f"Parameters: {best_guided_ae_results['parameters']}")

if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
