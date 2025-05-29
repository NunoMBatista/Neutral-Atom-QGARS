import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import silhouette_score
import seaborn as sns
import os
import json
from typing import Dict, List, Tuple, Any, Optional

from src.data_processing.data_processing import load_dataset, select_random_samples
from src.feature_reduction.feature_reduction import (
    apply_pca, apply_autoencoder, apply_guided_autoencoder,
    scale_to_detuning_range
)
from src.quantum_layer.qrc_layer import DetuningLayer

N_FEATURES = 12  # Number of features for the quantum layer

def generate_quantum_embeddings(
    data_train: Dict[str, Any],
    quantum_layer: DetuningLayer,
    method_name: str,
    encoding_dim: int = N_FEATURES,
    n_examples: int = 100,
    guided_lambda: float = 0.3,
    epochs: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate quantum embeddings using the specified reduction method"""
    print(f"\n== Processing {method_name} ==")
    
    # Select a subset of samples
    selected_indices, selected_features, selected_targets = select_random_samples(
        data_train, num_samples=n_examples, seed=42
    )
    
    # Apply dimensionality reduction
    if method_name == "pca":
        xs, _, spectral = apply_pca(
            data_train, 
            dim_pca=encoding_dim,
            selected_indices=selected_indices,
            selected_features=selected_features
        )
    elif method_name == "autoencoder":
        xs, _, spectral = apply_autoencoder(
            data_train, 
            encoding_dim=encoding_dim,
            epochs=epochs,
            selected_indices=selected_indices,
            selected_features=selected_features
        )
    elif method_name == "guided_autoencoder":
        xs, _, spectral, _ = apply_guided_autoencoder(
            data_train,
            quantum_layer=quantum_layer,
            encoding_dim=encoding_dim,
            epochs=epochs,
            guided_lambda=guided_lambda,
            selected_indices=selected_indices,
            selected_features=selected_features,
            selected_targets=selected_targets
        )
    
    # Scale to detuning range
    xs_scaled = scale_to_detuning_range(xs, spectral)
    
    # Generate quantum embeddings
    print(f"Computing quantum embeddings for {method_name}...")
    embeddings = quantum_layer.apply_layer(
        xs_scaled, 
        n_shots=1000, 
        show_progress=True
    )
    
    return embeddings, selected_targets

def visualize_embeddings(
    results: Dict[str, Dict[str, np.ndarray]], 
    output_dir: str = "../results/embedding_visualizations",
    method: str = "tsne",
    n_plot_samples: int = 100  # Number of samples to actually plot
):
    """Create visualization plots for the embeddings"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each method and create individual 3D plots
    for method_name, data in results.items():
        embeddings = data["embeddings"]
        labels = data["labels"]
        
        # Randomly select subset for visualization if we have more than n_plot_samples
        if embeddings.shape[1] > n_plot_samples:
            np.random.seed(42)  # For reproducibility
            sample_indices = np.random.choice(embeddings.shape[1], n_plot_samples, replace=False)
            vis_embeddings = embeddings[:, sample_indices]
            vis_labels = labels[sample_indices]
            print(f"Randomly selected {n_plot_samples} samples from {embeddings.shape[1]} for visualization")
        else:
            vis_embeddings = embeddings
            vis_labels = labels
        
        # Apply secondary dimensionality reduction with 3 components
        if method.lower() == "umap":
            print(f"Applying 3D UMAP to {method_name} embeddings...")
            reducer = umap.UMAP(n_components=3, random_state=42)
            reduced_embeddings = reducer.fit_transform(vis_embeddings.T)
        else:
            print(f"Applying 3D t-SNE to {method_name} embeddings...")
            reducer = TSNE(n_components=3, random_state=42, perplexity=min(30, len(vis_labels)-1))
            reduced_embeddings = reducer.fit_transform(vis_embeddings.T)
        
        # Create individual 3D scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            reduced_embeddings[:, 0], 
            reduced_embeddings[:, 1],
            reduced_embeddings[:, 2],
            c=vis_labels, 
            cmap='viridis', 
            alpha=0.7
        )
        
        ax.set_title(f"{method_name.replace('_', ' ').title()}")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_zlabel("Dimension 3")
        
        # Add legend
        legend = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend)
        
        # Save individual plot
        plt.tight_layout()
        plt.savefig(f"{output_dir}/quantum_embeddings_{method}_{method_name}.pdf")
        plt.savefig(f"{output_dir}/quantum_embeddings_{method}_{method_name}.png", dpi=300)
        plt.close()
    
    # Calculate and visualize silhouette scores
    calculate_metrics(results, output_dir)

def calculate_metrics(
    results: Dict[str, Dict[str, np.ndarray]],
    output_dir: str
):
    """Calculate and visualize quantitative metrics for embeddings separation"""
    metrics = {}
    
    for method_name, data in results.items():
        embeddings = data["embeddings"]
        labels = data["labels"]
        
        # Silhouette score (higher is better)
        # We need to transpose embeddings to have samples as rows
        sil_score = silhouette_score(embeddings.T, labels)
        
        metrics[method_name] = {
            "silhouette_score": sil_score
        }
        
        print(f"{method_name} silhouette score: {sil_score:.4f}")
    
    # Plot metrics
    plt.figure(figsize=(10, 6))
    methods = list(metrics.keys())
    sil_scores = [metrics[m]["silhouette_score"] for m in methods]
    
    plt.bar(methods, sil_scores)
    plt.title("Silhouette Score by Reduction Method")
    plt.ylabel("Silhouette Score (higher is better)")
    plt.ylim(0, max(sil_scores) * 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(f"{output_dir}/quantum_embeddings_metrics.pdf")
    plt.savefig(f"{output_dir}/quantum_embeddings_metrics.png", dpi=300)
    plt.show()

def save_embeddings_to_json(
    results: Dict[str, Dict[str, np.ndarray]],
    output_dir: str
):
    """Save embeddings and their labels to a JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for JSON serialization (convert numpy arrays to lists)
    json_data = {}
    for method_name, data in results.items():
        embeddings = data["embeddings"].tolist()  # Convert to Python list
        labels = data["labels"].tolist()  # Convert to Python list
        
        json_data[method_name] = {
            "embeddings": embeddings,
            "labels": labels
        }
    
    # Save to JSON file
    json_path = os.path.join(output_dir, "quantum_embeddings.json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f)
    
    print(f"Saved embeddings to {json_path}")

def main():
    global N_FEATURES 

    # Load dataset
    data_train, data_test = load_dataset(
        name="binary_mnist",  # or "cvc_clinic_db_patches" for polyp data
        num_examples=1000,
        num_test_examples=200
    )
    
    # Create quantum layer
    quantum_layer = DetuningLayer(
        geometry="chain",
        n_atoms=N_FEATURES,
        rabi_freq=2*np.pi,
        t_end=4.0,
        n_steps=16,
        readout_type="all"
    )
    
    # Define reduction methods to compare
    reduction_methods = ["PCA", "autoencoder", "guided_autoencoder"]
    
    # Dictionary to store results
    results = {}
    n_examples = 1000  # Process 1000 examples through the pipeline
    
    # Generate embeddings for each method
    for method in reduction_methods:
        embeddings, labels = generate_quantum_embeddings(
            data_train=data_train,
            quantum_layer=quantum_layer,
            method_name=method.lower(),
            n_examples=n_examples
        )
        
        results[method] = {
            "embeddings": embeddings,
            "labels": labels
        }
    
    # Save embeddings to JSON
    save_embeddings_to_json(
        results,
        output_dir="/home/nbatista/GIC-quAI-QRC/results/embeddings"
    )
    
    # Visualize embeddings
    visualize_embeddings(
        results,
        method="tsne",
        output_dir="/home/nbatista/GIC-quAI-QRC/results/figures/generated",
        n_plot_samples=100  # Only plot 100 points
    )

    visualize_embeddings(
        results,
        method="umap",
        output_dir="/home/nbatista/GIC-quAI-QRC/results/figures/generated",
        n_plot_samples=100  # Only plot 100 points
    )


if __name__ == "__main__":
    main()
