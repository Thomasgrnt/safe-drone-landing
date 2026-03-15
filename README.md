# Safe Drone Landing avec YOLO

## Présentation du projet

L’objectif de ce projet est de développer un système capable de déterminer si une zone est adaptée à l’atterrissage d’un drone à partir d’une image.

Pour cela, nous utilisons un modèle de détection d’objets basé sur l’algorithme YOLO (You Only Look Once) afin de détecter automatiquement une plateforme d’atterrissage appelée landing pad.

Le système analyse une image capturée par une caméra (webcam dans notre cas) et réalise les étapes suivantes :

1. Détection du landing pad dans l’image  
2. Calcul du score de confiance de la détection  
3. Calcul de la taille de la zone détectée dans l’image  
4. Prise d’une décision finale :

SAFE → l’atterrissage est considéré comme sûr  
NOT SAFE → les conditions ne sont pas jugées suffisantes pour un atterrissage sécurisé

Cette approche permet de transformer un problème complexe de perception en un problème de détection d’objets suivi d’une règle de décision simple.

# Organisation du projet

Le projet est organisé de la manière suivante :

safe-drone-landing/

dataset/
    images/
        train/
        val/
        test/

    labels/
        train/
        val/
        test/

metadata/

best.pt : modèle YOLO entraîné

train.py : script permettant d’entraîner le modèle

evaluate_test.py : script permettant d’évaluer les performances du modèle

live_demo.py : script permettant d’effectuer une détection en temps réel avec la webcam

live_inference.py : script permettant de tester le modèle sur une image

choose_thresholds.py : script utilisé pour déterminer les seuils de décision

thresholds.json : fichier contenant les paramètres utilisés pour la décision SAFE / NOT SAFE

live_demo_log.csv : fichier contenant les résultats enregistrés lors de la démonstration

requirements.txt : liste des bibliothèques nécessaires pour exécuter le projet

README.md : documentation du projet

# Dataset

Le dataset utilisé pour entraîner le modèle suit le format Ultralytics YOLO.

Les images sont réparties en trois ensembles :

| dossier | rôle |
|------|------|
| train | images utilisées pour l’entraînement |
| val | images utilisées pour la validation |
| test | images utilisées pour l’évaluation finale |

Chaque image possède un fichier `.txt` associé contenant les annotations de la bounding box.

Format des annotations : classe x_center y_center largeur hauteur


Les coordonnées sont normalisées entre 0 et 1.

Le dataset contient environ 300 images du landing pad générées dans différentes conditions :

- angles de vue différents  
- distances différentes  
- variations d’éclairage  
- arrière-plans différents  

Cela permet de rendre le modèle plus robuste.

# Entraînement du modèle

L’entraînement du modèle a été réalisé à l’aide de Google Colab afin d’utiliser un GPU.

Le modèle YOLO est entraîné à partir des images annotées du dataset.

Exemple de code utilisé pour l’entraînement :

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="landingpad.yaml",
    epochs=50,
    imgsz=640
)
```

À la fin de l’entraînement, le modèle génère un fichier appelé : best.pt
Ce fichier contient les poids du modèle entraîné et est utilisé pour effectuer les prédictions.

# Principe de décision SAFE / NOT SAFE

Lorsque le modèle détecte un landing pad, deux paramètres sont utilisés pour prendre une décision.

## 1. Score de confiance

Le modèle fournit un score indiquant la probabilité que l’objet détecté soit réellement un landing pad : max_conf

## 2. Taille de la zone détectée

La surface de la bounding box est comparée à la surface totale de l’image : area_ratio = surface_box / surface_image


# Règle de décision

La scène est considérée comme SAFE si : max_conf ≥ tau_conf ET area_ratio ≥ tau_area

Sinon la décision est : NOT SAFE

# Installation et reproduction du projet

Pour reproduire ce projet, il faut suivre les étapes suivantes.

## 1. Installer les dépendances

Dans un terminal, installer les bibliothèques nécessaires : pip install -r requirements.txt


## 2. Se placer dans le dossier du projet

cd safe-drone-landing

## 3. Lancer la démonstration en temps réel

python live_demo.py

La webcam s’ouvre et le modèle analyse les images en temps réel.


# Commandes pendant la démonstration

| touche | action |
|------|------|
| q | quitter le programme |
| s | enregistrer le résultat dans un fichier CSV |
| r | changer l’identifiant du scénario |

Les résultats sont enregistrés dans :

live_demo_log.csv

# Résultats

Le système affiche à l’écran :

- la bounding box autour du landing pad  
- le score de confiance  
- la taille de la zone détectée  
- la décision SAFE ou NOT SAFE  

# Limites du projet

Certaines limites ont été observées :

- taille relativement réduite du dataset (~300 images)  
- qualité de la webcam inférieure aux images du dataset  

Ces éléments peuvent diminuer la précision du modèle.


# Conclusion

Ce projet permet de mettre en place une pipeline complète de deep learning :

- création du dataset  
- annotation des images  
- entraînement d’un modèle YOLO  
- évaluation du modèle  
- démonstration en temps réel avec une webcam  

Le système est capable de détecter automatiquement une plateforme d’atterrissage et de déterminer si l’atterrissage est sûr ou non.
