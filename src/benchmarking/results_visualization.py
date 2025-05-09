import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the results from the JSON file
def load_results(file_path):
    with open(file_path, 'r') as f:
        results = json.load(f)
    return results

# Convert the results to a pandas DataFrame for easier analysis
def results_to_dataframe(results):
    data = []
    for result in results:
        row = {
            'dim_pca': result['parameters']['dim_pca'],
            'rabi_freq': result['parameters']['rabi_freq'],
            'time_steps': result['parameters']['time_steps'],
            'readout_type': result['parameters']['readout_type'],
            'PCA+linear': result['test_accuracies']['PCA+linear'],
            'QRC': result['test_accuracies']['QRC'],
            'PCA+NN': result['test_accuracies']['PCA+NN'],
            'best_model': result['best_model'],
            'best_accuracy': result['best_accuracy'],
            'execution_time': result['execution_time']
        }
        data.append(row)
    return pd.DataFrame(data)

# Create visualizations for parameter analysis
def create_visualizations(df):
    # Set the style for all visualizations
    sns.set(style="whitegrid")
    plt.figure(figsize=(15, 10))
    
    # 1. Overall model comparison
    plt.subplot(2, 2, 1)
    model_accuracies = df[['PCA+linear', 'QRC', 'PCA+NN']].mean()
    sns.barplot(x=model_accuracies.index, y=model_accuracies.values)
    plt.title('Average Accuracy by Model')
    plt.ylabel('Accuracy (%)')
    
    # 2. Effect of dim_pca on model accuracy
    plt.subplot(2, 2, 2)
    pivot_dim = df.pivot_table(
        index='dim_pca', 
        values=['PCA+linear', 'QRC', 'PCA+NN'], 
        aggfunc='mean'
    )
    pivot_dim.plot(marker='o')
    plt.title('Effect of PCA Dimension on Accuracy')
    plt.xlabel('PCA Dimension')
    plt.ylabel('Accuracy (%)')
    plt.xticks(df['dim_pca'].unique())
    
    # 3. Effect of rabi_freq on model accuracy
    plt.subplot(2, 2, 3)
    pivot_rabi = df.pivot_table(
        index='rabi_freq', 
        values=['PCA+linear', 'QRC', 'PCA+NN'], 
        aggfunc='mean'
    )
    pivot_rabi.plot(marker='o')
    plt.title('Effect of Rabi Frequency on Accuracy')
    plt.xlabel('Rabi Frequency')
    plt.ylabel('Accuracy (%)')
    
    # 4. Effect of time_steps on model accuracy
    plt.subplot(2, 2, 4)
    pivot_time = df.pivot_table(
        index='time_steps', 
        values=['PCA+linear', 'QRC', 'PCA+NN'], 
        aggfunc='mean'
    )
    pivot_time.plot(marker='o')
    plt.title('Effect of Time Steps on Accuracy')
    plt.xlabel('Time Steps')
    plt.ylabel('Accuracy (%)')
    plt.xticks(df['time_steps'].unique())
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()
    
    # 5. Readout type comparison
    plt.figure(figsize=(12, 8))
    pivot_readout = df.pivot_table(
        index='readout_type', 
        values=['PCA+linear', 'QRC', 'PCA+NN'], 
        aggfunc='mean'
    )
    pivot_readout.plot(kind='bar')
    plt.title('Effect of Readout Type on Accuracy')
    plt.xlabel('Readout Type')
    plt.ylabel('Accuracy (%)')
    plt.savefig('readout_comparison.png')
    plt.close()
    
    # 6. QRC model heatmaps for parameter interactions
    param_pairs = [
        ('dim_pca', 'rabi_freq'),
        ('dim_pca', 'time_steps'),
        ('rabi_freq', 'time_steps')
    ]
    
    for i, (param1, param2) in enumerate(param_pairs):
        plt.figure(figsize=(10, 8))
        pivot = df.pivot_table(
            index=param1, 
            columns=param2, 
            values='QRC', 
            aggfunc='mean'
        )
        sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.1f')
        plt.title(f'QRC Accuracy: {param1} vs {param2}')
        plt.savefig(f'heatmap_{param1}_{param2}.png')
        plt.close()
    
    # 7. Best parameter combinations for QRC
    top_qrc = df.sort_values(by='QRC', ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    top_params = top_qrc[['dim_pca', 'rabi_freq', 'time_steps', 'readout_type', 'QRC']]
    
    param_str = [f"dim={row['dim_pca']}, rabi={row['rabi_freq']}, steps={row['time_steps']}, readout={row['readout_type']}" 
                 for _, row in top_params.iterrows()]
    
    plt.barh(range(len(top_params)), top_params['QRC'])
    plt.yticks(range(len(top_params)), param_str)
    plt.xlabel('QRC Accuracy (%)')
    plt.title('Top 10 Parameter Combinations for QRC')
    plt.tight_layout()
    plt.savefig('top_qrc_params.png')
    plt.close()

if __name__ == "__main__":
    # Set the file path
    file_path = "results/all_results_20250422_060200.json"
    
    # Load the results
    results = load_results(file_path)
    
    # Convert to DataFrame
    df = results_to_dataframe(results)
    
    # Create visualizations
    create_visualizations(df)
    
    print("Visualizations created successfully!")
    
    # Print some basic statistics
    print("\nBasic Statistics:")
    print(f"Number of experiments: {len(df)}")
    print(f"Average PCA+linear accuracy: {df['PCA+linear'].mean():.2f}%")
    print(f"Average QRC accuracy: {df['QRC'].mean():.2f}%")
    print(f"Average PCA+NN accuracy: {df['PCA+NN'].mean():.2f}%")
    
    # Count best models
    best_model_counts = df['best_model'].value_counts()
    print("\nBest model distribution:")
    for model, count in best_model_counts.items():
        print(f"{model}: {count} times ({count/len(df)*100:.1f}%)")