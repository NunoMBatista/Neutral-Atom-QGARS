# Benchmark Results

This folder contains JSON files with results from parameter benchmarking experiments:

- Individual experiment files (format: `{test_name}_{uuid}.json`)
- Summary files combining multiple experiments (`all_results_{timestamp}.json`)
- The best result discovered so far (`best_results.json`)

Each result file contains:
- Parameter values used
- Test accuracies for all models
- Best model and accuracy
- Execution time
- Metadata (timestamp, test name, ID)
