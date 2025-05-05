import os
import sys
import json
import time
import itertools
import argparse
import numpy as np
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
from qrc_polyp_python.cli import AVAILABLE_READOUT_TYPES

# Create results directory if it doesn't exist
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# File to store the best results
BEST_RESULTS_FILE = os.path.join(RESULTS_DIR, "best_results.json")

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
    
    # Parameter ranges for testing (removed geometries parameter)
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
    
    # Constraints for faster testing
    parser.add_argument("--num-examples", type=int, default=100,
                        help="Number of examples to use for training")
    parser.add_argument("--num-test-examples", type=int, default=50,
                        help="Number of examples to use for testing")
    parser.add_argument("--nepochs", type=int, default=20,
                        help="Number of training epochs")
    
    return parser.parse_args()

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
    # Define parameter ranges (fixed to chain geometry only)
    param_grid = {
        "geometry": ["chain"],  # Fixed to chain only
        "dim_pca": list(range(args.dim_pca_range[0], args.dim_pca_range[1] + 1, args.dim_pca_range[2])),
        "rabi_freq": list(np.arange(args.rabi_freq_range[0], args.rabi_freq_range[1] + 0.01, args.rabi_freq_range[2])),
        "time_steps": list(range(args.time_steps_range[0], args.time_steps_range[1] + 1, args.time_steps_range[2])),
        "readout_type": args.readout_types,
    }
    
    # Create combinations
    if args.random_search:
        # Random search
        combinations = []
        for _ in range(args.num_combinations):
            params = {
                "geometry": "chain",  # Fixed to chain
                "dim_pca": np.random.choice(param_grid["dim_pca"]),
                "rabi_freq": np.random.choice(param_grid["rabi_freq"]),
                "time_steps": np.random.choice(param_grid["time_steps"]),
                "readout_type": np.random.choice(param_grid["readout_type"]),
            }
            combinations.append(params)
    else:
        # Grid search (with fixed chain geometry)
        param_grid.pop("geometry")  # Remove geometry from grid search
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
        
        # Add fixed chain geometry to all combinations
        for combo in combinations:
            combo["geometry"] = "chain"
        
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
            "no_progress": True,  # Disable progress bars for benchmarking
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
    if not hasattr(args, "regularization"):
        args.regularization = 0.0005
    if not hasattr(args, "batchsize"):
        args.batchsize = 100
    if not hasattr(args, "learning_rate"):
        args.learning_rate = 0.01
    
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
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)
    
    return filepath

def update_best_results(result: Dict[str, Any]) -> bool:
    """
    Update best results file if current QRC result is better.
    
    Parameters
    ----------
    result : Dict[str, Any]
        Current test result
        
    Returns
    -------
    bool
        Whether the QRC result was the new best
    """
    # Load existing best results if available
    if os.path.exists(BEST_RESULTS_FILE):
        with open(BEST_RESULTS_FILE, 'r') as f:
            try:
                best_results = json.load(f)
                best_qrc_accuracy = best_results.get("qrc_accuracy", 0)
            except json.JSONDecodeError:
                best_qrc_accuracy = 0
    else:
        best_qrc_accuracy = 0
    
    # Check if current QRC result is better
    current_qrc_accuracy = result["qrc_accuracy"]
    is_new_best = current_qrc_accuracy > best_qrc_accuracy
    
    # Update if better
    if is_new_best:
        with open(BEST_RESULTS_FILE, 'w') as f:
            json.dump(result, f, indent=2)
    
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
        
        # Check if it's the best QRC result so far
        is_best = update_best_results(result)
        if is_best:
            print(f"✓ New best QRC result! Accuracy: {result['qrc_accuracy']:.2f}%")
        
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
                
                # Check if it's the best QRC result so far
                is_best = update_best_results(result)
                if is_best:
                    print(f"✓ New best QRC result! Accuracy: {result['qrc_accuracy']:.2f}%")
                
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
    with open(all_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to: {all_results_file}")
    
    # Print best result
    if os.path.exists(BEST_RESULTS_FILE):
        with open(BEST_RESULTS_FILE, 'r') as f:
            best_results = json.load(f)
        print("\nBest Result:")
        print(f"QRC model accuracy: {best_results['qrc_accuracy']:.2f}%")
        print(f"Parameters: {best_results['parameters']}")

if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
