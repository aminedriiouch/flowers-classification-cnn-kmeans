#  Flowers Classification: CNN vs K-means

Comparaison de deux approches pour la classification d'images de fleurs : Deep Learning (CNN) vs Machine Learning Non-Supervisé (K-means).

##  Projet

**Cours:** Régression en Grande Dimension  
**Institution:** Institut National de Statistique et d'Économie Appliquée (INSEA)  
**Encadrant:** Mr. Janati

**Étudiants:**
- Mohamed Amine Driouch
- Mouad Belkamel
- Khalid El Faghloumi

##  Objectifs

- Implémenter un CNN custom pour la classification de 5 classes de fleurs
- Appliquer K-means pour la segmentation et l'extraction de features
- Comparer les deux approches de manière rigoureuse
- Proposer une approche hybride

##  Dataset

- **Source:** Flowers Recognition Dataset
- **Classes:** Daisy, Dandelion, Rose, Sunflower, Tulip
- **Taille:** ~4000 images (800 par classe)
- **Lien:** [Kaggle Flowers Dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)

##  Architecture

### CNN
- 4 blocs de convolution (32→64→128→128 filtres)
- MaxPooling, Dropout (0.5, 0.3)
- Dense (256) + Softmax (5 classes)
- **Paramètres:** ~1M

### K-means
- Segmentation par couleur (RGB)
- K optimal déterminé par Silhouette Score
- Extraction de la fleur du fond

##  Résultats

| Métrique | CNN | K-means |
|----------|-----|---------|
| Accuracy | **85%** | N/A (segmentation) |
| Silhouette Score | N/A | **0.47** |
| Temps inférence | **50 ms** | 200 ms |

## Installation
```bash
# Cloner le repository
git clone https://github.com/aminedriiouch/flowers-classification-cnn-kmeans.git
cd flowers-classification-cnn-kmeans

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OU
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

### Entraîner le CNN
```bash
python scripts/train_cnn.py --epochs 30 --batch-size 32
```

### Segmentation K-means
```bash
python scripts/train_kmeans.py --k 5 --input data/flowers
```

### Évaluation
```bash
python scripts/evaluate.py --model models/flower_classifier_cnn.h5
```

## Structure du Projet
```
├── notebooks/          # Jupyter Notebooks
├── src/                # Code source Python
├── models/             # Modèles sauvegardés
├── results/            # Résultats et figures
├── docs/               # Documentation et rapport
└── scripts/            # Scripts d'exécution
```

## Notebooks

1. **01_CNN_Classification.ipynb** - Implémentation complète du CNN
2. **02_Kmeans_Segmentation.ipynb** - Segmentation K-means

## Méthodologie

Voir le [rapport complet](docs/rapport_final.pdf) pour les détails mathématiques et l'analyse comparative.

## Rapport

- [Rapport Final PDF](docs/rapport_final.pdf)

## Contribution

Chaque membre a contribué de manière égale :
- **Mohamed Amine Driouch:** CNN implementation, training
- **Mouad Belkamel:** K-means segmentation, evaluation
- **Khalid El Faghloumi:** Data preprocessing, documentation

