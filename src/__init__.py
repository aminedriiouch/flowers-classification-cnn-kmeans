"""
Flowers Classification Project
Comparison of CNN and K-means for flower image classification
INSEA - Régression en Grande Dimension


__version__ = 1.0.0
__authors__ = [Mohamed Amine Driouch, Mouad Belkamel, Khalid El Faghloumi]

"""

from .cnn_model import create_cnn_model, compile_model, train_model
from .kmeans_segmentation import (
    segment_image_kmeans,
    extract_largest_segment,
    find_optimal_clusters,
    calculate_segmentation_metrics
)
from .data_preprocessing import (
    create_data_generators,
    preprocess_image,
    visualize_samples
)
from .utils import (
    predict_flower,
    plot_training_history,
    save_model,
    load_model
)

__all__ = [
    'create_cnn_model',
    'compile_model',
    'train_model',
    'segment_image_kmeans',
    'extract_largest_segment',
    'find_optimal_clusters',
    'calculate_segmentation_metrics',
    'create_data_generators',
    'preprocess_image',
    'visualize_samples',
    'predict_flower',
    'plot_training_history',
    'save_model',
    'load_model'
]