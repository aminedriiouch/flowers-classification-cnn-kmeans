"""
CNN Model Module
Custom Convolutional Neural Network for flower image classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def create_cnn_model(img_size=128, num_classes=5):
    """
    Create a custom CNN architecture for flower classification
    
    Architecture:
        - 4 Convolutional blocks (32, 64, 128, 128 filters)
        - MaxPooling after each conv block
        - Dropout layers (0.5, 0.3) for regularization
        - Dense layer (256 neurons)
        - Softmax output for multi-class classification
    
    Args:
        img_size: Input image size (default: 128)
        num_classes: Number of output classes (default: 5)
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # Block 1: 32 filters
        layers.Conv2D(32, (3, 3), activation='relu', 
                     input_shape=(img_size, img_size, 3)),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2: 64 filters
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3: 128 filters
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 4: 128 filters
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dropout(0.5),  # Strong dropout to prevent overfitting
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile the CNN model with optimizer and loss function
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for Adam optimizer (default: 0.001)
    
    Returns:
        Compiled model
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Modèle compilé avec succès!")
    return model


def create_callbacks(patience_early_stop=5, patience_reduce_lr=3):
    """
    Create training callbacks for early stopping and learning rate reduction
    
    Args:
        patience_early_stop: Epochs to wait before early stopping (default: 5)
        patience_reduce_lr: Epochs to wait before reducing LR (default: 3)
    
    Returns:
        List of callbacks
    """
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=patience_early_stop,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=patience_reduce_lr,
        min_lr=1e-7,
        verbose=1
    )
    
    callbacks = [early_stop, reduce_lr]
    print("✅ Callbacks configurés!")
    
    return callbacks


def train_model(model, train_generator, validation_generator, 
                epochs=30, callbacks=None, verbose=1):
    """
    Train the CNN model
    
    Args:
        model: Compiled Keras model
        train_generator: Training data generator
        validation_generator: Validation data generator
        epochs: Number of training epochs (default: 30)
        callbacks: List of Keras callbacks (default: None)
        verbose: Verbosity mode (default: 1)
    
    Returns:
        Training history object
    """
    print("\n🚀 Début de l'entraînement...\n")
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=verbose
    )
    
    print("\n✅ Entraînement terminé!")
    
    # Print final results
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    print(f"\n📊 Résultats finaux:")
    print(f"   - Train Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
    print(f"   - Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
    
    return history


def get_model_summary(model):
    """
    Print detailed model summary
    
    Args:
        model: Keras model
    """
    print("\n" + "="*70)
    print("📋 ARCHITECTURE DU MODÈLE CNN")
    print("="*70)
    model.summary()
    print("="*70)
    
    # Calculate total parameters
    total_params = model.count_params()
    print(f"\n💡 Paramètres totaux: {total_params:,}")
    print(f"   (~{total_params/1e6:.2f}M parameters)\n")
