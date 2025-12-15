"""
K-means Segmentation Module
Image segmentation using K-means clustering for flower extraction
"""

import numpy as np
import cv2
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score


def segment_image_kmeans(img, n_clusters=5, random_state=42):
    """
    Segment an image using K-means clustering on RGB pixel values
    
    Mathematical approach:
        1. Reshape image to (n_pixels, 3) for RGB values
        2. Apply K-means to find K color clusters
        3. Replace each pixel with its cluster center color
    
    Args:
        img: Input image (numpy array, RGB)
        n_clusters: Number of clusters (default: 5)
        random_state: Random seed for reproducibility (default: 42)
    
    Returns:
        segmented_img: Segmented image with cluster colors
        labels: Cluster labels for each pixel (2D array)
        kmeans_model: Trained K-means model
    """
    # Save original shape
    original_shape = img.shape
    
    # Reshape to 2D array: (n_pixels, 3)
    pixel_values = img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(pixel_values)
    
    # Replace each pixel with its cluster center color
    centers = np.uint8(kmeans.cluster_centers_)
    segmented_image = centers[labels.flatten()]
    
    # Reshape back to original image shape
    segmented_image = segmented_image.reshape(original_shape)
    
    return segmented_image, labels.reshape(original_shape[:2]), kmeans


def extract_largest_segment(img, labels):
    """
    Extract the largest segment (typically the flower) from background
    
    Strategy:
        - Find the 2nd most frequent cluster (1st is usually background)
        - Create a binary mask for that cluster
        - Apply mask to extract the flower
    
    Args:
        img: Original image (RGB)
        labels: Cluster labels (2D array)
    
    Returns:
        result: Extracted flower image
        mask: Binary mask (255 for flower, 0 for background)
    """
    # Find cluster frequencies
    unique, counts = np.unique(labels, return_counts=True)
    
    # Sort by frequency (descending)
    sorted_indices = np.argsort(counts)[::-1]
    
    # Take 2nd most frequent (1st is usually background)
    main_label = unique[sorted_indices[1]] if len(unique) > 1 else unique[0]
    
    # Create binary mask
    mask = (labels == main_label).astype(np.uint8) * 255
    
    # Apply mask to extract flower
    result = cv2.bitwise_and(img, img, mask=mask)
    
    return result, mask


def find_optimal_clusters(img, max_k=10):
    """
    Find optimal number of clusters using Elbow Method and Silhouette Score
    
    Methods:
        1. Elbow Method: Look for "elbow" in inertia curve
        2. Silhouette Score: Measure cluster cohesion (closer to 1 is better)
    
    Args:
        img: Input image (RGB)
        max_k: Maximum number of clusters to test (default: 10)
    
    Returns:
        k_range: List of K values tested
        inertias: List of inertia values (within-cluster sum of squares)
        silhouette_scores: List of silhouette scores
    """
    # Prepare pixel data
    pixel_values = img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    print("🔍 Recherche du nombre optimal de clusters...")
    
    for k in k_range:
        # Train K-means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixel_values)
        
        # Calculate inertia
        inertias.append(kmeans.inertia_)
        
        # Calculate silhouette score (sample for speed)
        sample_size = min(5000, len(pixel_values))
        indices = np.random.choice(len(pixel_values), sample_size, replace=False)
        score = silhouette_score(pixel_values[indices], labels[indices])
        silhouette_scores.append(score)
        
        print(f"   K={k}: Silhouette={score:.4f}, Inertia={kmeans.inertia_:,.0f}")
    
    # Find best K
    best_k_idx = np.argmax(silhouette_scores)
    best_k = list(k_range)[best_k_idx]
    
    print(f"\n✅ Nombre optimal recommandé: K = {best_k}")
    print(f"   Silhouette Score: {silhouette_scores[best_k_idx]:.4f}")
    
    return list(k_range), inertias, silhouette_scores


def calculate_segmentation_metrics(img, labels, kmeans_model):
    """
    Calculate quality metrics for image segmentation
    
    Metrics:
        - Silhouette Score: Cluster cohesion (-1 to 1, higher is better)
        - Inertia: Within-cluster sum of squares (lower is better)
        - Davies-Bouldin Index: Cluster separation (closer to 0 is better)
    
    Args:
        img: Input image (RGB)
        labels: Cluster labels (2D array)
        kmeans_model: Trained K-means model
    
    Returns:
        Dictionary with metrics
    """
    # Prepare data
    pixel_values = img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    labels_flat = labels.flatten()
    
    # Sample for computational efficiency
    sample_size = min(10000, len(pixel_values))
    indices = np.random.choice(len(pixel_values), sample_size, replace=False)
    
    # Calculate metrics
    metrics = {
        'silhouette_score': silhouette_score(
            pixel_values[indices], 
            labels_flat[indices]
        ),
        'inertia': kmeans_model.inertia_,
        'davies_bouldin': davies_bouldin_score(
            pixel_values[indices], 
            labels_flat[indices]
        ),
        'n_clusters': len(np.unique(labels_flat)),
        'cluster_sizes': dict(zip(*np.unique(labels_flat, return_counts=True)))
    }
    
    return metrics


def print_segmentation_metrics(metrics, img_shape):
    """
    Print segmentation metrics in a formatted way
    
    Args:
        metrics: Dictionary of metrics from calculate_segmentation_metrics
        img_shape: Original image shape (for percentage calculations)
    """
    print("\n" + "="*70)
    print("📊 MÉTRIQUES DE SEGMENTATION")
    print("="*70)
    
    print(f"\n✅ Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"   → Score entre -1 et 1 (plus proche de 1 = meilleur)")
    
    print(f"\n📉 Inertia: {metrics['inertia']:,.2f}")
    print(f"   → Somme des distances au centroïde (plus bas = meilleur)")
    
    print(f"\n🎯 Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}")
    print(f"   → Plus proche de 0 = meilleure séparation des clusters")
    
    print(f"\n🔢 Nombre de clusters: {metrics['n_clusters']}")
    
    print(f"\n📊 Taille des segments (en pixels):")
    total_pixels = img_shape[0] * img_shape[1]
    for cluster_id, size in sorted(metrics['cluster_sizes'].items()):
        percentage = (size / total_pixels) * 100
        print(f"   Segment {cluster_id}: {size:,} pixels ({percentage:.2f}%)")
    
    print("="*70 + "\n")


def save_kmeans_model(kmeans_model, optimal_k, target_size, classes, 
                      metrics, filepath='flower_segmentation_model.pkl'):
    """
    Save K-means model and configuration
    
    Args:
        kmeans_model: Trained K-means model
        optimal_k: Optimal number of clusters
        target_size: Image preprocessing size
        classes: List of flower classes
        metrics: Segmentation metrics dictionary
        filepath: Output file path (default: 'flower_segmentation_model.pkl')
    """
    model_data = {
        'model': kmeans_model,
        'optimal_k': optimal_k,
        'target_size': target_size,
        'classes': classes,
        'metrics': metrics
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✅ Modèle sauvegardé dans '{filepath}'")


def load_kmeans_model(filepath='flower_segmentation_model.pkl'):
    """
    Load saved K-means model and configuration
    
    Args:
        filepath: Path to saved model file
    
    Returns:
        Dictionary with model and configuration
    """
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"✅ Modèle chargé depuis '{filepath}'")
    return model_data
