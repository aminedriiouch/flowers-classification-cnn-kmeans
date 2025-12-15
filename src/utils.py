"""
Utilities Module
Helper functions for visualization, evaluation, and model management
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import pandas as pd


CLASSES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']


def predict_flower(image_path, model, img_size=128, show_plot=True):
    """
    Predict flower class from an image using trained CNN model
    
    Args:
        image_path: Path to the image file
        model: Trained Keras model
        img_size: Image size for preprocessing (default: 128)
        show_plot: Whether to display the result (default: True)
    
    Returns:
        predicted_class: Name of predicted flower class
        confidence: Prediction confidence (0-100%)
        probabilities: Array of probabilities for all classes
    """
    # Load and preprocess image
    img = image.load_img(image_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class = CLASSES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    
    if show_plot:
        # Display image with prediction
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Prédiction: {predicted_class}\nConfiance: {confidence:.2f}%',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Print probabilities
        print("\n📊 Probabilités par classe:")
        for i, flower in enumerate(CLASSES):
            prob = predictions[0][i] * 100
            bar = '█' * int(prob / 2)  # Visual bar
            print(f"   {flower:12s}: {prob:6.2f}% {bar}")
    
    return predicted_class, confidence, predictions[0]


def plot_training_history(history, save_path=None):
    """
    Plot training history (accuracy and loss curves)
    
    Args:
        history: Keras training history object
        save_path: Path to save the figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', 
                linewidth=2, marker='o')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', 
                linewidth=2, marker='s')
    axes[0].set_title('📈 Évolution de l\'Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Train Loss', 
                linewidth=2, marker='o')
    axes[1].plot(history.history['val_loss'], label='Val Loss', 
                linewidth=2, marker='s')
    axes[1].set_title('📉 Évolution de la Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figure sauvegardée: {save_path}")
    
    plt.show()


def save_model(model, filepath='models/flower_classifier_cnn.h5'):
    """
    Save trained Keras model
    
    Args:
        model: Trained Keras model
        filepath: Path to save the model (default: 'models/flower_classifier_cnn.h5')
    """
    model.save(filepath)
    print(f"✅ Modèle sauvegardé: {filepath}")


def load_model(filepath='models/flower_classifier_cnn.h5'):
    """
    Load saved Keras model
    
    Args:
        filepath: Path to the saved model
    
    Returns:
        Loaded Keras model
    """
    from tensorflow import keras
    model = keras.models.load_model(filepath)
    print(f"✅ Modèle chargé depuis: {filepath}")
    return model


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix for model evaluation
    
    Args:
        y_true: True labels (integers)
        y_pred: Predicted labels (integers)
        save_path: Path to save the figure (optional)
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASSES, yticklabels=CLASSES,
                cbar_kws={'label': 'Count'})
    plt.title('📊 Matrice de Confusion', fontsize=16, fontweight='bold')
    plt.ylabel('Vraie Classe', fontsize=12)
    plt.xlabel('Classe Prédite', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Matrice de confusion sauvegardée: {save_path}")
    
    plt.show()


def visualize_segmentation_results(original, segmented, mask, extracted, 
                                   flower_name="", save_path=None):
    """
    Visualize K-means segmentation results
    
    Args:
        original: Original image
        segmented: Segmented image
        mask: Binary mask
        extracted: Extracted flower
        flower_name: Name of the flower (optional)
        save_path: Path to save the figure (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    title = f'🌸 Résultats de Segmentation'
    if flower_name:
        title += f' - {flower_name.capitalize()}'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Original
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Image Originale', fontweight='bold', fontsize=12)
    axes[0, 0].axis('off')
    
    # Segmented
    axes[0, 1].imshow(segmented)
    axes[0, 1].set_title('Segmentation K-means', fontweight='bold', fontsize=12)
    axes[0, 1].axis('off')
    
    # Mask
    axes[1, 0].imshow(mask, cmap='gray')
    axes[1, 0].set_title('Masque de la Fleur', fontweight='bold', fontsize=12)
    axes[1, 0].axis('off')
    
    # Extracted
    axes[1, 1].imshow(extracted)
    axes[1, 1].set_title('Fleur Extraite du Fond', fontweight='bold', fontsize=12)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figure sauvegardée: {save_path}")
    
    plt.show()


def plot_optimal_k_analysis(k_values, inertias, silhouette_scores, save_path=None):
    """
    Plot Elbow Method and Silhouette Score analysis
    
    Args:
        k_values: List of K values tested
        inertias: List of inertia values
        silhouette_scores: List of silhouette scores
        save_path: Path to save the figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Elbow Method
    axes[0].plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Nombre de clusters (K)', fontsize=12)
    axes[0].set_ylabel('Inertia', fontsize=12)
    axes[0].set_title('📊 Méthode du Coude (Elbow Method)', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Silhouette Score
    axes[1].plot(k_values, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Nombre de clusters (K)', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_title('🎯 Score de Silhouette', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Mark optimal K
    best_k_idx = np.argmax(silhouette_scores)
    axes[1].axvline(x=k_values[best_k_idx], color='green', 
                    linestyle='--', linewidth=2, label=f'Optimal K={k_values[best_k_idx]}')
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figure sauvegardée: {save_path}")
    
    plt.show()


def create_comparison_report(cnn_results, kmeans_results, save_path=None):
    """
    Create a comparison report between CNN and K-means approaches
    
    Args:
        cnn_results: Dictionary with CNN metrics
        kmeans_results: Dictionary with K-means metrics
        save_path: Path to save the report (optional)
    
    Returns:
        Pandas DataFrame with comparison
    """
    comparison = pd.DataFrame({
        'Métrique': ['Accuracy', 'Silhouette Score', 'Temps Inférence (ms)', 
                     'Paramètres', 'Type d\'Approche'],
        'CNN': [
            f"{cnn_results.get('accuracy', 0)*100:.2f}%",
            'N/A',
            cnn_results.get('inference_time', 'N/A'),
            f"~{cnn_results.get('parameters', 0)/1e6:.1f}M",
            'Supervisé'
        ],
        'K-means': [
            'N/A (Segmentation)',
            f"{kmeans_results.get('silhouette_score', 0):.4f}",
            kmeans_results.get('inference_time', 'N/A'),
            'K clusters',
            'Non-Supervisé'
        ]
    })
    
    print("\n" + "="*70)
    print("📊 COMPARAISON CNN vs K-means")
    print("="*70)
    print(comparison.to_string(index=False))
    print("="*70 + "\n")
    
    if save_path:
        comparison.to_csv(save_path, index=False)
        print(f"✅ Rapport sauvegardé: {save_path}")
    
    return comparison
    
    




import json

def save_metrics_json(metrics, filepath='results/metrics/evaluation_metrics.json'):

    Save evaluation metrics to JSON file
    
    Args:
        metrics: Dictionary with evaluation metrics
        filepath: Path to save JSON file
    
    Example:
        metrics = {
            'cnn': {
                'accuracy': 0.85,
                'val_accuracy': 0.82,
                'loss': 0.45,
                'val_loss': 0.52,
                'inference_time_ms': 50.2
            },
            'kmeans': {
                'silhouette_score': 0.47,
                'inertia': 125000,
                'davies_bouldin': 0.89,
                'inference_time_ms': 200.5
            },
            'comparison': {
                'cnn_better_for': 'classification',
                'kmeans_better_for': 'segmentation'
            }
        }

    import os
    
    # Create directory if needed
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save to JSON
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Métriques sauvegardées: {filepath}")


def load_metrics_json(filepath='results/metrics/evaluation_metrics.json'):
    """
    Load metrics from JSON file
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        Dictionary with metrics
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    
    print(f"✅ Métriques chargées depuis: {filepath}")
    return metrics


# Exemple d'utilisation dans evaluate.py


all_metrics = {
    'cnn': {
        'accuracy': cnn_results['accuracy'],
        'inference_time': cnn_results['inference_time'],
        'parameters': cnn_results['parameters']
    },
    'kmeans': {
        'silhouette_score': kmeans_results['silhouette_score'],
        'inference_time': kmeans_results['inference_time']
    },
    'timestamp': datetime.now().isoformat()
}

save_metrics_json(all_metrics)
