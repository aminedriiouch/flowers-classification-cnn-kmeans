#!/usr/bin/env python3
"""
Script d'évaluation des modèles
Evaluate CNN and K-means models on test data

Usage:
    python scripts/evaluate.py --cnn-model models/flower_classifier_cnn.h5
"""

import argparse
import os
import sys
import numpy as np
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import (
    predict_flower, 
    load_model,
    plot_confusion_matrix,
    create_comparison_report
)
from src.kmeans_segmentation import (
    load_kmeans_model,
    segment_image_kmeans,
    calculate_segmentation_metrics
)
from src.data_preprocessing import preprocess_image


CLASSES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate trained models'
    )
    
    parser.add_argument(
        '--cnn-model',
        type=str,
        default='models/flower_classifier_cnn.h5',
        help='Path to trained CNN model'
    )
    
    parser.add_argument(
        '--kmeans-model',
        type=str,
        default='models/flower_segmentation_model.pkl',
        help='Path to trained K-means model'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='data/flowers',
        help='Path to test dataset'
    )
    
    parser.add_argument(
        '--n-samples',
        type=int,
        default=10,
        help='Number of samples to evaluate per class'
    )
    
    parser.add_argument(
        '--save-confusion',
        type=str,
        default='results/figures/confusion_matrix.png',
        help='Path to save confusion matrix'
    )
    
    parser.add_argument(
        '--save-report',
        type=str,
        default='results/metrics/comparison_report.csv',
        help='Path to save comparison report'
    )
    
    return parser.parse_args()


def evaluate_cnn(model, data_path, n_samples=10, img_size=128):
    """
    Evaluate CNN model on test samples
    
    Args:
        model: Trained CNN model
        data_path: Path to dataset
        n_samples: Number of samples per class
        img_size: Image size
    
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n🔍 Évaluation du modèle CNN...")
    
    y_true = []
    y_pred = []
    inference_times = []
    
    correct = 0
    total = 0
    
    for class_idx, flower in enumerate(CLASSES):
        path = os.path.join(data_path, flower)
        if not os.path.exists(path):
            continue
        
        img_files = os.listdir(path)[:n_samples]
        
        for img_name in img_files:
            img_path = os.path.join(path, img_name)
            
            try:
                # Predict
                start_time = time.time()
                predicted_class, confidence, probs = predict_flower(
                    img_path, model, img_size=img_size, show_plot=False
                )
                inference_time = (time.time() - start_time) * 1000  # ms
                
                inference_times.append(inference_time)
                
                # Track predictions
                y_true.append(class_idx)
                y_pred.append(CLASSES.index(predicted_class))
                
                total += 1
                if predicted_class == flower:
                    correct += 1
                
                print(f"   {flower}/{img_name}: {predicted_class} ({confidence:.1f}%)")
                
            except Exception as e:
                print(f"   ❌ Erreur: {str(e)}")
                continue
    
    accuracy = correct / total if total > 0 else 0
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    
    print(f"\n✅ CNN Évaluation terminée:")
    print(f"   - Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    print(f"   - Temps d'inférence moyen: {avg_inference_time:.2f} ms")
    
    return {
        'accuracy': accuracy,
        'inference_time': f"{avg_inference_time:.2f} ms",
        'y_true': y_true,
        'y_pred': y_pred,
        'correct': correct,
        'total': total,
        'parameters': model.count_params()
    }


def evaluate_kmeans(model_path, data_path, n_samples=5):
    """
    Evaluate K-means segmentation
    
    Args:
        model_path: Path to K-means model
        data_path: Path to dataset
        n_samples: Number of samples per class
    
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n🔍 Évaluation du modèle K-means...")
    
    # Load model
    model_data = load_kmeans_model(model_path)
    
    silhouette_scores = []
    inference_times = []
    
    for flower in CLASSES:
        path = os.path.join(data_path, flower)
        if not os.path.exists(path):
            continue
        
        img_files = os.listdir(path)[:n_samples]
        
        for img_name in img_files:
            img_path = os.path.join(path, img_name)
            
            try:
                # Preprocess
                img = preprocess_image(img_path, target_size=model_data['target_size'])
                
                # Segment
                start_time = time.time()
                seg_img, labels, kmeans = segment_image_kmeans(
                    img, 
                    n_clusters=model_data['optimal_k']
                )
                inference_time = (time.time() - start_time) * 1000  # ms
                
                inference_times.append(inference_time)
                
                # Calculate metrics
                metrics = calculate_segmentation_metrics(img, labels, kmeans)
                silhouette_scores.append(metrics['silhouette_score'])
                
                print(f"   {flower}/{img_name}: Silhouette={metrics['silhouette_score']:.4f}")
                
            except Exception as e:
                print(f"   ❌ Erreur: {str(e)}")
                continue
    
    avg_silhouette = np.mean(silhouette_scores) if silhouette_scores else 0
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    
    print(f"\n✅ K-means Évaluation terminée:")
    print(f"   - Silhouette Score moyen: {avg_silhouette:.4f}")
    print(f"   - Temps d'inférence moyen: {avg_inference_time:.2f} ms")
    
    return {
        'silhouette_score': avg_silhouette,
        'inference_time': f"{avg_inference_time:.2f} ms",
        'n_samples': len(silhouette_scores)
    }


def main():
    """Main evaluation function"""
    args = parse_args()
    
    print("\n" + "="*70)
    print("🎯 ÉVALUATION DES MODÈLES")
    print("="*70)
    
    # Create output directories
    os.makedirs(os.path.dirname(args.save_confusion), exist_ok=True)
    os.makedirs(os.path.dirname(args.save_report), exist_ok=True)
    
    # Evaluate CNN
    print("\n" + "="*70)
    print("📊 ÉVALUATION CNN")
    print("="*70)
    
    if os.path.exists(args.cnn_model):
        cnn_model = load_model(args.cnn_model)
        cnn_results = evaluate_cnn(
            model=cnn_model,
            data_path=args.data,
            n_samples=args.n_samples
        )
        
        # Plot confusion matrix
        if cnn_results['y_true'] and cnn_results['y_pred']:
            print("\n📊 Génération de la matrice de confusion...")
            plot_confusion_matrix(
                y_true=cnn_results['y_true'],
                y_pred=cnn_results['y_pred'],
                save_path=args.save_confusion
            )
    else:
        print(f"⚠️  Modèle CNN non trouvé: {args.cnn_model}")
        cnn_results = None
    
    # Evaluate K-means
    print("\n" + "="*70)
    print("📊 ÉVALUATION K-MEANS")
    print("="*70)
    
    if os.path.exists(args.kmeans_model):
        kmeans_results = evaluate_kmeans(
            model_path=args.kmeans_model,
            data_path=args.data,
            n_samples=min(5, args.n_samples)
        )
    else:
        print(f"⚠️  Modèle K-means non trouvé: {args.kmeans_model}")
        kmeans_results = None
    
    # Create comparison report
    if cnn_results and kmeans_results:
        print("\n" + "="*70)
        print("📊 COMPARAISON CNN vs K-MEANS")
        print("="*70)
        
        comparison = create_comparison_report(
            cnn_results=cnn_results,
            kmeans_results=kmeans_results,
            save_path=args.save_report
        )
    
    # Final summary
    print("\n" + "="*70)
    print("✅ ÉVALUATION TERMINÉE")
    print("="*70)
    
    if cnn_results:
        print(f"\n🎯 CNN:")
        print(f"   - Accuracy: {cnn_results['accuracy']*100:.2f}%")
        print(f"   - Temps d'inférence: {cnn_results['inference_time']}")
    
    if kmeans_results:
        print(f"\n🎯 K-means:")
        print(f"   - Silhouette Score: {kmeans_results['silhouette_score']:.4f}")
        print(f"   - Temps d'inférence: {kmeans_results['inference_time']}")
    
    print(f"\n📁 Fichiers générés:")
    if cnn_results:
        print(f"   - Matrice de confusion: {args.save_confusion}")
    if cnn_results and kmeans_results:
        print(f"   - Rapport de comparaison: {args.save_report}")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
