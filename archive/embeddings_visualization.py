import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import seaborn as sns
import os
import json
from typing import Dict, List, Tuple, Any, Optional
from scipy.spatial.distance import pdist, squareform
import plotly.graph_objects as go  # For interactive 3D plots

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

def save_probability_distributions(
    results: Dict[str, Dict[str, np.ndarray]],
    output_dir: str
):
    """Save KDE plots of the probability distributions for each embedding dimension"""
    os.makedirs(output_dir, exist_ok=True)
    
    for method_name, data in results.items():
        embeddings = data["embeddings"]
        labels = data["labels"]
        unique_labels = np.unique(labels)
        
        # Create a figure with a subplot for each dimension
        n_dims = min(8, embeddings.shape[0])  # Limit to first 8 dimensions for readability
        fig, axes = plt.subplots(n_dims, 1, figsize=(10, 2*n_dims), sharex=True)
        
        if n_dims == 1:
            axes = [axes]  # Make it iterable when there's only one subplot
            
        for i in range(n_dims):
            ax = axes[i]
            # Plot KDE for each class
            for label in unique_labels:
                class_indices = np.where(labels == label)[0]
                sns.kdeplot(
                    embeddings[i, class_indices],
                    ax=ax,
                    label=f"Class {label}",
                    fill=True,
                    alpha=0.3
                )
            
            ax.set_title(f"Dimension {i+1}")
            ax.set_ylabel("Density")
            
            # Only add legend to the first subplot
            if i == 0:
                ax.legend(title="Classes")
        
        axes[-1].set_xlabel("Embedding Value")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/prob_dist_{method_name}.pdf")
        plt.savefig(f"{output_dir}/prob_dist_{method_name}.png", dpi=300)
        plt.close()

def visualize_2d_embeddings(
    results: Dict[str, Dict[str, np.ndarray]],
    output_dir: str,
    method: str = "tsne",
    n_plot_samples: int = 100
):
    """Create 2D visualization plots for the embeddings"""
    os.makedirs(output_dir, exist_ok=True)
    
    for method_name, data in results.items():
        embeddings = data["embeddings"]
        labels = data["labels"]
        
        # Randomly select subset for visualization if we have more than n_plot_samples
        if embeddings.shape[1] > n_plot_samples:
            np.random.seed(42)
            sample_indices = np.random.choice(embeddings.shape[1], n_plot_samples, replace=False)
            vis_embeddings = embeddings[:, sample_indices]
            vis_labels = labels[sample_indices]
        else:
            vis_embeddings = embeddings
            vis_labels = labels
        
        # Apply dimensionality reduction with 2 components
        if method.lower() == "umap":
            reducer = umap.UMAP(n_components=2, random_state=42)
            reduced_embeddings = reducer.fit_transform(vis_embeddings.T)
        else:
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(vis_labels)-1))
            reduced_embeddings = reducer.fit_transform(vis_embeddings.T)
        
        # Create 2D scatter plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            c=vis_labels,
            cmap='viridis',
            alpha=0.7,
            s=50
        )
        
        plt.title(f"{method_name.replace('_', ' ').title()} - 2D {method.upper()}")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        
        # Add legend
        legend = plt.legend(*scatter.legend_elements(), title="Classes")
        plt.gca().add_artist(legend)
        
        # Add density contours
        for label in np.unique(vis_labels):
            mask = vis_labels == label
            if sum(mask) > 2:  # Need at least 3 points for KDE
                sns.kdeplot(
                    x=reduced_embeddings[mask, 0],
                    y=reduced_embeddings[mask, 1],
                    levels=5,
                    alpha=0.5,
                    linewidths=1
                )
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/quantum_embeddings_2d_{method}_{method_name}.pdf")
        plt.savefig(f"{output_dir}/quantum_embeddings_2d_{method}_{method_name}.png", dpi=300)
        plt.close()

def visualize_class_separation(
    results: Dict[str, Dict[str, np.ndarray]],
    output_dir: str
):
    """Visualize the separation between classes in the embedding space"""
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_data = {}
    
    for method_name, data in results.items():
        embeddings = data["embeddings"]
        labels = data["labels"]
        unique_labels = np.unique(labels)
        
        # Calculate pairwise distances between class centroids
        centroids = []
        for label in unique_labels:
            class_indices = np.where(labels == label)[0]
            centroid = np.mean(embeddings[:, class_indices], axis=1)
            centroids.append(centroid)
        
        centroids = np.array(centroids)
        distances = squareform(pdist(centroids))
        
        # Plot heatmap of distances between class centroids
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            distances,
            annot=True,
            cmap="YlGnBu",
            xticklabels=[f"Class {l}" for l in unique_labels],
            yticklabels=[f"Class {l}" for l in unique_labels]
        )
        plt.title(f"{method_name.replace('_', ' ').title()} - Class Centroid Distances")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/class_distances_{method_name}.pdf")
        plt.savefig(f"{output_dir}/class_distances_{method_name}.png", dpi=300)
        plt.close()
        
        # Calculate additional clustering metrics
        metrics_data[method_name] = {
            "avg_centroid_distance": np.mean(distances[distances > 0]),
            "min_centroid_distance": np.min(distances[distances > 0]) if distances.size > 1 else 0
        }
    
    # Plot comparison of metrics
    methods = list(metrics_data.keys())
    avg_distances = [metrics_data[m]["avg_centroid_distance"] for m in methods]
    min_distances = [metrics_data[m]["min_centroid_distance"] for m in methods]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(methods))
    width = 0.35
    
    ax.bar(x - width/2, avg_distances, width, label='Avg Distance')
    ax.bar(x + width/2, min_distances, width, label='Min Distance')
    
    ax.set_ylabel('Distance')
    ax.set_title('Class Separation Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/class_separation_metrics.pdf")
    plt.savefig(f"{output_dir}/class_separation_metrics.png", dpi=300)
    plt.close()

def visualize_embedding_correlations(
    results: Dict[str, Dict[str, np.ndarray]],
    output_dir: str,
    max_dims: int = 16
):
    """Visualize correlations between embedding dimensions"""
    os.makedirs(output_dir, exist_ok=True)
    
    for method_name, data in results.items():
        embeddings = data["embeddings"]
        
        # Limit to first max_dims dimensions for readability
        n_dims = min(max_dims, embeddings.shape[0])
        embeddings_subset = embeddings[:n_dims, :]
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(embeddings_subset)
        
        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            xticklabels=[f"Dim {i+1}" for i in range(n_dims)],
            yticklabels=[f"Dim {i+1}" for i in range(n_dims)]
        )
        plt.title(f"{method_name.replace('_', ' ').title()} - Embedding Dimension Correlations")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/embedding_correlations_{method_name}.pdf")
        plt.savefig(f"{output_dir}/embedding_correlations_{method_name}.png", dpi=300)
        plt.close()

def calculate_metrics(
    results: Dict[str, Dict[str, np.ndarray]],
    output_dir: str
):
    """Calculate and visualize quantitative metrics for embeddings separation"""
    metrics = {}
    
    for method_name, data in results.items():
        embeddings = data["embeddings"]
        labels = data["labels"]
        
        # Transpose embeddings to have samples as rows
        embeddings_t = embeddings.T
        
        # Silhouette score (higher is better)
        sil_score = silhouette_score(embeddings_t, labels)
        
        # Davies-Bouldin Index (lower is better)
        try:
            db_score = davies_bouldin_score(embeddings_t, labels)
        except:
            db_score = float('nan')  # In case of error (e.g., single cluster)
        
        # Calinski-Harabasz Index (higher is better)
        try:
            ch_score = calinski_harabasz_score(embeddings_t, labels)
        except:
            ch_score = float('nan')  # In case of error
        
        metrics[method_name] = {
            "silhouette_score": sil_score,
            "davies_bouldin_score": db_score,
            "calinski_harabasz_score": ch_score
        }
        
        print(f"{method_name} silhouette score: {sil_score:.4f}")
        print(f"{method_name} Davies-Bouldin index: {db_score:.4f}")
        print(f"{method_name} Calinski-Harabasz index: {ch_score:.4f}")
    
    # Plot metrics
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    methods = list(metrics.keys())
    
    # Silhouette Score (higher is better)
    sil_scores = [metrics[m]["silhouette_score"] for m in methods]
    axes[0].bar(methods, sil_scores)
    axes[0].set_title("Silhouette Score (higher is better)")
    axes[0].set_ylabel("Score")
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Davies-Bouldin Index (lower is better)
    db_scores = [metrics[m]["davies_bouldin_score"] for m in methods]
    axes[1].bar(methods, db_scores)
    axes[1].set_title("Davies-Bouldin Index (lower is better)")
    axes[1].set_ylabel("Score")
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Calinski-Harabasz Index (higher is better)
    ch_scores = [metrics[m]["calinski_harabasz_score"] for m in methods]
    axes[2].bar(methods, ch_scores)
    axes[2].set_title("Calinski-Harabasz Index (higher is better)")
    axes[2].set_ylabel("Score")
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/quantum_embeddings_metrics.pdf")
    plt.savefig(f"{output_dir}/quantum_embeddings_metrics.png", dpi=300)
    plt.close()

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

def save_method_embeddings_to_json(
    method_name: str,
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_dir: str
):
    """Save embeddings and labels for a single method to a JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for JSON serialization
    json_data = {
        "embeddings": embeddings.tolist(),
        "labels": labels.tolist()
    }
    
    # Save to JSON file
    json_path = os.path.join(output_dir, f"quantum_embeddings_{method_name}.json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f)
    
    print(f"Saved {method_name} embeddings to {json_path}")

def load_method_embeddings_from_json(
    method_name: str,
    input_dir: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Load embeddings and labels for a single method from a JSON file"""
    json_path = os.path.join(input_dir, f"quantum_embeddings_{method_name}.json")
    
    if not os.path.exists(json_path):
        return None, None
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        embeddings = np.array(data["embeddings"])
        labels = np.array(data["labels"])
        
        print(f"Loaded {method_name} embeddings from {json_path}")
        return embeddings, labels
    except Exception as e:
        print(f"Error loading {method_name} embeddings: {str(e)}")
        return None, None

def load_embeddings_from_json(
    json_path: str
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load all embeddings and labels from a combined JSON file
    
    Args:
        json_path: Path to the JSON file containing all embeddings
        
    Returns:
        Dictionary with method names as keys, each containing embeddings and labels
    """
    if not os.path.exists(json_path):
        print(f"JSON file not found: {json_path}")
        return {}
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to numpy arrays
        results = {}
        for method_name, method_data in data.items():
            results[method_name] = {
                "embeddings": np.array(method_data["embeddings"]),
                "labels": np.array(method_data["labels"])
            }
        
        print(f"Successfully loaded embeddings from {json_path}")
        print(f"Loaded {len(results)} methods: {', '.join(results.keys())}")
        return results
    except Exception as e:
        print(f"Error loading embeddings from {json_path}: {str(e)}")
        return {}

def save_combined_distributions(
    results: Dict[str, Dict[str, np.ndarray]],
    output_dir: str
):
    """Save KDE plots of the combined probability distributions across all dimensions"""
    os.makedirs(output_dir, exist_ok=True)
    
    for method_name, data in results.items():
        embeddings = data["embeddings"]
        labels = data["labels"]
        unique_labels = np.unique(labels)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Calculate pairwise distances for each class
        for label in unique_labels:
            class_indices = np.where(labels == label)[0]
            class_embeddings = embeddings[:, class_indices]
            
            # Flatten the embeddings to get overall distribution
            flattened_embeddings = class_embeddings.flatten()
            
            # Plot KDE of the flattened distribution
            sns.kdeplot(
                flattened_embeddings,
                label=f"Class {label}",
                fill=True,
                alpha=0.3
            )
        
        plt.title(f"{method_name.replace('_', ' ').title()} - Combined Dimension Distribution")
        plt.xlabel("Embedding Value")
        plt.ylabel("Density")
        plt.legend(title="Classes")
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/combined_dist_{method_name}.pdf")
        plt.savefig(f"{output_dir}/combined_dist_{method_name}.png", dpi=300)
        plt.close()
        
        # Alternative visualization: pairwise distances
        plt.figure(figsize=(12, 8))
        
        for label in unique_labels:
            class_indices = np.where(labels == label)[0]
            class_embeddings = embeddings[:, class_indices]
            
            # Calculate pairwise distances between points
            if len(class_indices) > 1:  # Need at least 2 points
                # Transpose to get points in rows
                distances = pdist(class_embeddings.T, metric='euclidean')
                
                # Plot distribution of distances
                sns.kdeplot(
                    distances,
                    label=f"Class {label}",
                    fill=True,
                    alpha=0.3
                )
        
        plt.title(f"{method_name.replace('_', ' ').title()} - Pairwise Distance Distribution")
        plt.xlabel("Pairwise Distance")
        plt.ylabel("Density")
        plt.legend(title="Classes")
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pairwise_dist_{method_name}.pdf")
        plt.savefig(f"{output_dir}/pairwise_dist_{method_name}.png", dpi=300)
        plt.close()

def create_interactive_3d_plots(
    results: Dict[str, Dict[str, np.ndarray]],
    output_dir: str,
    method: str = "tsne",
    n_plot_samples: int = 100
):
    """Create interactive 3D plots and save them as HTML files"""
    os.makedirs(output_dir, exist_ok=True)
    
    for method_name, data in results.items():
        embeddings = data["embeddings"]
        labels = data["labels"]
        
        # Randomly select subset for visualization if we have more than n_plot_samples
        if embeddings.shape[1] > n_plot_samples:
            np.random.seed(42)  # For reproducibility
            sample_indices = np.random.choice(embeddings.shape[1], n_plot_samples, replace=False)
            vis_embeddings = embeddings[:, sample_indices]
            vis_labels = labels[sample_indices]
            print(f"Randomly selected {n_plot_samples} samples from {embeddings.shape[1]} for interactive visualization")
        else:
            vis_embeddings = embeddings
            vis_labels = labels
        
        # Apply secondary dimensionality reduction with 3 components
        if method.lower() == "umap":
            print(f"Applying 3D UMAP to {method_name} embeddings for interactive plot...")
            reducer = umap.UMAP(n_components=3, random_state=42)
            reduced_embeddings = reducer.fit_transform(vis_embeddings.T)
        else:
            print(f"Applying 3D t-SNE to {method_name} embeddings for interactive plot...")
            reducer = TSNE(n_components=3, random_state=42, perplexity=min(30, len(vis_labels)-1))
            reduced_embeddings = reducer.fit_transform(vis_embeddings.T)
        
        # Create a color map for classes
        unique_labels = np.unique(vis_labels)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        color_map = {label: f'rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},{c[3]})' 
                    for label, c in zip(unique_labels, colors)}
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add traces for each class
        for label in unique_labels:
            mask = vis_labels == label
            fig.add_trace(go.Scatter3d(
                x=reduced_embeddings[mask, 0],
                y=reduced_embeddings[mask, 1],
                z=reduced_embeddings[mask, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=color_map[label],
                    opacity=0.7
                ),
                name=f'Class {label}'
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{method_name.replace('_', ' ').title()} - 3D {method.upper()}",
                font=dict(size=20)
            ),
            scene=dict(
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                zaxis_title="Dimension 3"
            ),
            width=900,
            height=700,
            margin=dict(l=0, r=0, b=0, t=40),
            legend=dict(
                title="Classes",
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        
        # Save as HTML
        html_path = f"{output_dir}/interactive_3d_{method}_{method_name}.html"
        fig.write_html(html_path, include_plotlyjs='cdn')
        print(f"Saved interactive 3D plot to {html_path}")

def main():
    global N_FEATURES 

    # Create output directories with correct paths
    base_output_dir = "/home/nbatista/GIC-quAI-QRC/results/embeddings"
    embeddings_dir = f"{base_output_dir}/embeddings"
    figures_dir = f"{base_output_dir}/figures/generated"
    
    # Create directories
    os.makedirs(embeddings_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Path to the combined JSON file (separate from directory)
    combined_json_path = os.path.join(embeddings_dir, "quantum_embeddings.json")
    
    # Define reduction methods to compare
    reduction_methods = ["PCA", "autoencoder", "guided_autoencoder"]
    
    # Check if the user wants to load from a specific JSON file
    import argparse
    parser = argparse.ArgumentParser(description='Quantum embeddings visualization')
    parser.add_argument('--load-json', type=str, default=None, 
                        help='Path to JSON file containing embeddings to load')
    parser.add_argument('--generate', action='store_true',
                        help='Force regeneration of embeddings even if files exist')
    parser.add_argument('--methods', nargs='+', default=reduction_methods,
                        help='Methods to process (if generating)')
    args = parser.parse_args()
    
    # Dictionary to store results
    results = {}
    
    # If a specific JSON file is provided, load from it
    if args.load_json and os.path.exists(args.load_json):
        results = load_embeddings_from_json(args.load_json)
        if not results:
            print("Failed to load embeddings from specified JSON. Falling back to default behavior.")
    
    # If results weren't loaded from a specific file, try the combined file
    if not results and os.path.exists(combined_json_path) and not args.generate:
        results = load_embeddings_from_json(combined_json_path)
    
    # If still no results, try loading individual method files or generate new ones
    if not results or args.generate:
        methods_to_process = args.methods if args.generate else reduction_methods
        
        for method in methods_to_process:
            # Skip if we're not forcing regeneration and already have this method in results
            if not args.generate and method in results:
                continue
                
            # Try to load from individual method file
            if not args.generate:
                embeddings, labels = load_method_embeddings_from_json(
                    method_name=method.lower(),
                    input_dir=embeddings_dir
                )
                
                if embeddings is not None and labels is not None:
                    results[method] = {
                        "embeddings": embeddings,
                        "labels": labels
                    }
                    continue
            
            # If we get here, we need to generate new embeddings
            print(f"Generating new embeddings for {method}...")
            
            # Load dataset (only if we need to generate embeddings)
            if 'data_train' not in locals():
                data_train, data_test = load_dataset(
                    "image_folder",  # or appropriate dataset name
                    data_dir="/home/nbatista/GIC-quAI-QRC/data/datasets/generated_polyp_dataset",
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
            
            # Generate embeddings
            embeddings, labels = generate_quantum_embeddings(
                data_train=data_train,
                quantum_layer=quantum_layer,
                method_name=method.lower(),
                n_examples=1000  # Set your desired number of examples
            )
            
            # Save embeddings immediately after generation
            save_method_embeddings_to_json(
                method_name=method.lower(),
                embeddings=embeddings,
                labels=labels,
                output_dir=embeddings_dir
            )
            
            # Store in results dictionary
            results[method] = {
                "embeddings": embeddings,
                "labels": labels
            }
        
        # Save all embeddings in a single file
        if results:
            save_embeddings_to_json(results, output_dir=embeddings_dir)
    
    # If we still have no results, something went wrong
    if not results:
        print("Failed to load or generate any embeddings. Exiting.")
        return
    
    print(f"Processing visualizations for methods: {', '.join(results.keys())}")
    
    # Generate all visualizations with error handling
    try:
        visualize_embeddings(
            results, method="tsne", output_dir=figures_dir, n_plot_samples=1000
        )
        print("3D t-SNE visualization completed successfully")
    except Exception as e:
        print(f"Error in 3D t-SNE visualization: {str(e)}")
    
    try:
        visualize_embeddings(
            results, method="umap", output_dir=figures_dir, n_plot_samples=1000
        )
        print("3D UMAP visualization completed successfully")
    except Exception as e:
        print(f"Error in 3D UMAP visualization: {str(e)}")
    
    # New visualization methods with error handling
    try:
        visualize_2d_embeddings(
            results, output_dir=figures_dir, method="tsne", n_plot_samples=1000
        )
        print("2D t-SNE visualization completed successfully")
    except Exception as e:
        print(f"Error in 2D t-SNE visualization: {str(e)}")
    
    try:
        visualize_2d_embeddings(
            results, output_dir=figures_dir, method="umap", n_plot_samples=1000
        )
        print("2D UMAP visualization completed successfully")
    except Exception as e:
        print(f"Error in 2D UMAP visualization: {str(e)}")
    
    try:
        save_probability_distributions(results, output_dir=figures_dir)
        print("Probability distribution visualization completed successfully")
    except Exception as e:
        print(f"Error in probability distribution visualization: {str(e)}")
    
    # Add the new combined distributions visualization
    try:
        save_combined_distributions(results, output_dir=figures_dir)
        print("Combined distribution visualization completed successfully")
    except Exception as e:
        print(f"Error in combined distribution visualization: {str(e)}")
    
    try:
        visualize_class_separation(results, output_dir=figures_dir)
        print("Class separation visualization completed successfully")
    except Exception as e:
        print(f"Error in class separation visualization: {str(e)}")
    
    try:
        visualize_embedding_correlations(results, output_dir=figures_dir)
        print("Embedding correlation visualization completed successfully")
    except Exception as e:
        print(f"Error in embedding correlation visualization: {str(e)}")
    
    try:
        calculate_metrics(results, output_dir=figures_dir)
        print("Metrics calculation completed successfully")
    except Exception as e:
        print(f"Error in metrics calculation: {str(e)}")
    
    # Add interactive 3D visualizations with Plotly
    try:
        create_interactive_3d_plots(
            results, method="tsne", output_dir=figures_dir, n_plot_samples=1000
        )
        print("Interactive 3D t-SNE visualization completed successfully")
    except Exception as e:
        print(f"Error in interactive 3D t-SNE visualization: {str(e)}")
    
    try:
        create_interactive_3d_plots(
            results, method="umap", output_dir=figures_dir, n_plot_samples=1000
        )
        print("Interactive 3D UMAP visualization completed successfully")
        
    except Exception as e:
        print(f"Error in metrics calculation: {str(e)}")


if __name__ == "__main__":
    main()
