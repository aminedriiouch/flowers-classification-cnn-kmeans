#!/usr/bin/env python3
"""
Script d'entraînement du modèle CNN
Train a CNN model for flower classification

Usage:
    python scripts/train_cnn.py --data data/flowers --epochs 30 --batch-size 32
"""

import argparse
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_preprocessing import create_data_generators, visualize_samples
from src.cnn_model import (
    create_cnn_model, 
    compile_model, 
    train_model, 
    create_callbacks,
    get_model_summary
)
from src.utils import plot_training_history, save_model


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train CNN model for flower classification'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='data/flowers',
        help='Path to dataset directory (default: data/flowers)'
    )
    
    parser.add_argument(
        '--img-size',
        type=int,
        default=128,
        help='Image size for training (default: 128)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of training epochs (default: 30)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    
    parser.add_argument(
        '--validation-split',
        type=float,
        default=0.2,
        help='Validation split ratio (default: 0.2)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='models/flower_classifier_cnn.h5',
        help='Output model path (default: models/flower_classifier_cnn.h5)'
    )
    
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Skip data visualization'
    )
    
    parser.add_argument(
        '--save-history',
        type=str,
        default='results/figures/training_history.png',
        help='Path to save training history plot'
    )
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    print("\n" + "="*70)
    print("🌸 ENTRAÎNEMENT DU MODÈLE CNN - FLOWERS CLASSIFICATION")
    print("="*70)
    print(f"\n⚙️  Configuration:")
    print(f"   - Dataset: {args.data}")
    print(f"   - Image size: {args.img_size}x{args.img_size}")
    print(f"   - Batch size: {args.batch_size}")
    print(f"   - Epochs: {args.epochs}")
    print(f"   - Learning rate: {args.learning_rate}")
    print(f"   - Validation split: {args.validation_split}")
    print(f"   - Output model: {args.output}")
    
    # Check if dataset exists
    if not os.path.exists(args.data):
        print(f"\n❌ Erreur: Dataset non trouvé à '{args.data}'")
        print("   Assurez-vous que le dataset est dans le bon dossier.")
        sys.exit(1)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.save_history), exist_ok=True)
    
    # Step 1: Load data
    print("\n" + "="*70)
    print("📂 ÉTAPE 1: Chargement des données")
    print("="*70)
    
    train_gen, val_gen = create_data_generators(
        data_path=args.data,
        img_size=args.img_size,
        batch_size=args.batch_size,
        validation_split=args.validation_split
    )
    
    # Visualize samples
    if not args.no_visualize:
        print("\n📸 Visualisation d'échantillons...")
        visualize_samples(train_gen, n_samples=10)
    
    # Step 2: Create model
    print("\n" + "="*70)
    print("🏗️  ÉTAPE 2: Création du modèle CNN")
    print("="*70)
    
    model = create_cnn_model(img_size=args.img_size, num_classes=5)
    get_model_summary(model)
    
    # Step 3: Compile model
    print("\n" + "="*70)
    print("⚙️  ÉTAPE 3: Compilation du modèle")
    print("="*70)
    
    model = compile_model(model, learning_rate=args.learning_rate)
    
    # Step 4: Setup callbacks
    print("\n" + "="*70)
    print("🔧 ÉTAPE 4: Configuration des callbacks")
    print("="*70)
    
    callbacks = create_callbacks(patience_early_stop=5, patience_reduce_lr=3)
    
    # Step 5: Train model
    print("\n" + "="*70)
    print("🚀 ÉTAPE 5: Entraînement du modèle")
    print("="*70)
    
    history = train_model(
        model=model,
        train_generator=train_gen,
        validation_generator=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Step 6: Save model
    print("\n" + "="*70)
    print("💾 ÉTAPE 6: Sauvegarde du modèle")
    print("="*70)
    
    save_model(model, filepath=args.output)
    
    # Step 7: Visualize results
    print("\n" + "="*70)
    print("📊 ÉTAPE 7: Visualisation des résultats")
    print("="*70)
    
    plot_training_history(history, save_path=args.save_history)
    
    # Final summary
    print("\n" + "="*70)
    print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
    print("="*70)
    print(f"\n📁 Fichiers générés:")
    print(f"   - Modèle: {args.output}")
    print(f"   - Historique: {args.save_history}")
    
    final_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"\n🎯 Performances finales:")
    print(f"   - Training Accuracy: {final_acc*100:.2f}%")
    print(f"   - Validation Accuracy: {final_val_acc*100:.2f}%")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
