"""
Data Preprocessing Module
Handles data loading, augmentation, and visualization for the flowers dataset
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Configuration
CLASSES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
NUM_CLASSES = len(CLASSES)


def create_data_generators(data_path, img_size=128, batch_size=32, validation_split=0.2):
    """
    Create training and validation data generators with augmentation
    
    Args:
        data_path: Path to the dataset directory
        img_size: Target image size (default: 128)
        batch_size: Batch size for training (default: 32)
        validation_split: Proportion of data for validation (default: 0.2)
    
    Returns:
        train_generator, validation_generator
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        validation_split=validation_split
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        data_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    validation_generator = train_datagen.flow_from_directory(
        data_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    print(f"\n✅ Données chargées:")
    print(f"   - Training samples: {train_generator.samples}")
    print(f"   - Validation samples: {validation_generator.samples}")
    
    return train_generator, validation_generator


def preprocess_image(image_path, target_size=(256, 256)):
    """
    Preprocess a single image for K-means segmentation
    
    Args:
        image_path: Path to the image file
        target_size: Target dimensions (width, height)
    
    Returns:
        Preprocessed image as numpy array (RGB)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to load image: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    
    return img


def visualize_samples(train_generator, n_samples=10, figsize=(15, 6)):
    """
    Visualize sample images from the training generator
    
    Args:
        train_generator: Keras ImageDataGenerator
        n_samples: Number of samples to display (default: 10)
        figsize: Figure size (default: (15, 6))
    """
    plt.figure(figsize=figsize)
    images, labels = next(train_generator)
    
    rows = 2
    cols = n_samples // rows
    
    for i in range(min(n_samples, len(images))):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i])
        class_idx = np.argmax(labels[i])
        plt.title(CLASSES[class_idx], fontweight='bold')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def analyze_dataset_structure(data_path):
    """
    Analyze and print dataset structure and statistics
    
    Args:
        data_path: Path to the dataset directory
    
    Returns:
        Dictionary with class names and image counts
    """
    stats = {}
    
    print(f"\n📊 Structure du Dataset:")
    print("=" * 50)
    
    for flower in CLASSES:
        path = os.path.join(data_path, flower)
        if os.path.exists(path):
            num_images = len(os.listdir(path))
            stats[flower] = num_images
            print(f"  - {flower.capitalize():12s}: {num_images:4d} images")
        else:
            print(f"  - {flower.capitalize():12s}: ⚠️  NOT FOUND")
            stats[flower] = 0
    
    print("=" * 50)
    print(f"Total: {sum(stats.values())} images\n")
    
    return stats


def visualize_class_samples(data_path, n_per_class=2, figsize=(20, 8)):
    """
    Display sample images from each class
    
    Args:
        data_path: Path to the dataset directory
        n_per_class: Number of samples per class (default: 2)
        figsize: Figure size (default: (20, 8))
    """
    fig, axes = plt.subplots(n_per_class, len(CLASSES), figsize=figsize)
    fig.suptitle('🌸 Exemples d\'Images par Classe', fontsize=16, fontweight='bold')
    
    for idx, flower in enumerate(CLASSES):
        path = os.path.join(data_path, flower)
        if not os.path.exists(path):
            continue
            
        images = os.listdir(path)[:n_per_class]
        
        for i, img_name in enumerate(images):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i, idx].imshow(img)
                axes[i, idx].set_title(flower.capitalize(), fontweight='bold')
                axes[i, idx].axis('off')
    
    plt.tight_layout()
    plt.show()
