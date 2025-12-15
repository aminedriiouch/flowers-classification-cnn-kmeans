#!/usr/bin/env python3
"""
Script de segmentation K-means
Apply K-means segmentation to flower images

Usage:
    python scripts/train_kmeans.py --data data/flowers --k 5 --samples 10
"""

import argparse
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_preprocessing import preprocess_image
from src.kmeans_segmentation import (
    segment_image_kmeans,
    extract_largest_segment,
    find_optimal_clusters,
    calculate_segmentation_metrics,
    print_segmentation_metrics,
    save_kmeans_model
)
from src.utils import (
    visualize_segmentation_results,
    plot_optimal_k_analysis
)
import pandas as pd
import numpy as np


CLASSES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='K-means segmentation for flower images'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='data/flowers',
        help='Path to dataset directory (default: data/flowers)'
    )
    
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='Number of clusters (default: 5)'
    )
    
    parser.add_argument(
        '--find-optimal-k',
        action='store_true',
        help='Find optimal K using elbow method and silhouette score'
    )
    
    parser.add_argument(
        '--max-k',
        type=int,
        default=10,
        help='Maximum K to test when finding optimal (default: 10)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=10,
        help='Number of images to process (default: 10)'
    )
    
    parser.add_argument(
        '--img-size',
        type=int,
        default=256,
        help='Image size for processing (default: 256)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='models/flower_segmentation_model.pkl',
        help='Output model path (default: models/flower_segmentation_model.pkl)'
    )
    
    parser.add_argument(
        '--save-results',
        type=str,
        default='results/figures/kmeans_results.png',
        help='Path to save segmentation results'
    )
    
    return parser.parse_args()


def process_multiple_images(data_path, n_samples=10, k=5, img_size=256):
    """
    Process multiple images with K-means segmentation
    
    Args:
        data_path: Path to dataset
        n_samples: Number of samples per class
        k: Number of clusters
        img_size: Target image size
    
    Returns:
        List of results dictionaries
    """
    results = []
    
    for flower in CLASSES:
        path = os.path.join(data_path, flower)
        if not os.path.exists(path):
            print(f"⚠️  Classe '{flower}' non trouvée, ignorée.")
            continue
        
        img_files = os.listdir(path)[:n_samples]
        
        for img_name in img_files:
            img_path = os.path.join(path, img_name)
            
            try:
                # Preprocessing
                img = preprocess_image(img_path, target_size=(img_size, img_size))
                
                # Segmentation
                seg_img, labels, model = segment_image_kmeans(img, n_clusters=k)
                
                # Extract flower
                extracted, mask = extract_largest_segment(img, labels)
                
                # Calculate metrics
                metrics = calculate_segmentation_metrics(img, labels, model)
                
                results.append({
                    'class': flower,
                    'file': img_name,
                    'original': img,
                    'segmented': seg_img,
                    'extracted': extracted,
                    'mask': mask,
                    'labels': labels,
                    'model': model,
                    'silhouette': metrics['silhouette_score'],
                    'inertia': metrics['inertia'],
                    'davies_bouldin': metrics['davies_bouldin']
                })
                
                print(f"✅ {flower}/{img_name} - Silhouette: {metrics['silhouette_score']:.4f}")
                
            except Exception as e:
                print(f"❌ Erreur avec {flower}/{img_name}: {str(e)}")
                continue
    
    return results


def main():
    """Main segmentation function"""
    args = parse_args()
    
    print("\n" + "="*70)
    print("🌸 SEGMENTATION K-MEANS - FLOWERS CLASSIFICATION")
    print("="*70)
    print(f"\n⚙️  Configuration:")
    print(f"   - Dataset: {args.data}")
    print(f"   - Nombre de clusters (K): {args.k}")
    print(f"   - Taille d'image: {args.img_size}x{args.img_size}")
    print(f"   - Échantillons par classe: {args.samples}")
    print(f"   - Output model: {args.output}")
    
    # Check dataset
    if not os.path.exists(args.data):
        print(f"\n❌ Erreur: Dataset non trouvé à '{args.data}'")
        sys.exit(1)
    
    # Create output directories
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.save_results), exist_ok=True)
    
    # Step 1: Find optimal K (optional)
    if args.find_optimal_k:
        print("\n" + "="*70)
        print("🔍 ÉTAPE 1: Recherche du K optimal")
        print("="*70)
        
        # Use a sample image
        sample_class = CLASSES[0]
        sample_path = os.path.join(args.data, sample_class)
        sample_img_name = os.listdir(sample_path)[0]
        sample_img_path = os.path.join(sample_path, sample_img_name)
        
        sample_img = preprocess_image(sample_img_path, target_size=(args.img_size, args.img_size))
        
        k_values, inertias, silhouette_scores = find_optimal_clusters(
            sample_img, 
            max_k=args.max_k
        )
        
        # Plot analysis
        plot_optimal_k_analysis(k_values, inertias, silhouette_scores)
        
        # Update K to optimal
        optimal_k = k_values[np.argmax(silhouette_scores)]
        print(f"\n💡 K optimal trouvé: {optimal_k}")
        print(f"   Voulez-vous utiliser K={optimal_k} ? (actuellement: K={args.k})")
        args.k = optimal_k
    
    # Step 2: Process multiple images
    print("\n" + "="*70)
    print("🚀 ÉTAPE 2: Segmentation des images")
    print("="*70)
    print(f"\nTraitement de {args.samples} images par classe avec K={args.k}...\n")
    
    results = process_multiple_images(
        data_path=args.data,
        n_samples=args.samples,
        k=args.k,
        img_size=args.img_size
    )
    
    if not results:
        print("\n❌ Aucune image n'a pu être traitée.")
        sys.exit(1)
    
    print(f"\n✅ {len(results)} images segmentées avec succès!")
    
    # Step 3: Display statistics
    print("\n" + "="*70)
    print("📊 ÉTAPE 3: Statistiques globales")
    print("="*70)
    
    df_results = pd.DataFrame([{
        'Classe': r['class'],
        'Silhouette Score': r['silhouette'],
        'Inertia': r['inertia'],
        'Davies-Bouldin': r['davies_bouldin']
    } for r in results])
    
    print("\n📊 Statistiques par classe:")
    print(df_results.groupby('Classe').mean().to_string())
    
    print("\n📊 Moyennes générales:")
    print(df_results.describe().loc[['mean', 'std']].to_string())
    
    # Step 4: Visualize some results
    print("\n" + "="*70)
    print("🎨 ÉTAPE 4: Visualisation des résultats")
    print("="*70)
    
    # Show first few results
    n_display = min(3, len(results))
    for i in range(n_display):
        r = results[i]
        visualize_segmentation_results(
            original=r['original'],
            segmented=r['segmented'],
            mask=r['mask'],
            extracted=r['extracted'],
            flower_name=r['class']
        )
    
    # Step 5: Save model
    print("\n" + "="*70)
    print("💾 ÉTAPE 5: Sauvegarde du modèle")
    print("="*70)
    
    # Use the last model as representative
    last_result = results[-1]
    
    save_kmeans_model(
        kmeans_model=last_result['model'],
        optimal_k=args.k,
        target_size=(args.img_size, args.img_size),
        classes=CLASSES,
        metrics={
            'mean_silhouette': df_results['Silhouette Score'].mean(),
            'mean_inertia': df_results['Inertia'].mean(),
            'mean_davies_bouldin': df_results['Davies-Bouldin'].mean()
        },
        filepath=args.output
    )
    
    # Final summary
    print("\n" + "="*70)
    print("✅ SEGMENTATION TERMINÉE AVEC SUCCÈS!")
    print("="*70)
    print(f"\n📁 Fichiers générés:")
    print(f"   - Modèle: {args.output}")
    
    print(f"\n🎯 Performances moyennes:")
    print(f"   - Silhouette Score: {df_results['Silhouette Score'].mean():.4f}")
    print(f"   - Inertia: {df_results['Inertia'].mean():,.0f}")
    print(f"   - Davies-Bouldin: {df_results['Davies-Bouldin'].mean():.4f}")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
