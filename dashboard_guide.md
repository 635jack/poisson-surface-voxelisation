# Guide de Référence : Interface & Paramètres

Ce document récapitule le rôle de chaque contrôle (sliders, menus) et la signification de chaque résultat affiché dans l'application **Mesh Comparison Viewer**.

---

## 🎛️ Paramètres de Configuration (Barre Latérale)

### Configuration Générale
*   **Object** (Menu déroulant) : Choisit le modèle 3D source dans le dataset GLB (`sphere`, `box`, `cylinder`, etc.).
*   **Grasp Strategy** (Menu déroulant) : Sélectionne la stratégie de saisie (`front_back`, `left_right`...) qui définit quels sont les points de contact et normales injectés dans l'algorithme.

### 🎨 Options d'Affichage (Display)
*   **Opacities** (Plusieurs sliders) : Règlent la transparence (de 0.05 à 1.0) des différents calques (Maillage GT, Maillage Poisson, Voxels GT, Voxels Poisson). Utile pour regarder à l'intérieur d'un volume superposé.
*   **Show Contact Points** (Case à cocher) : Affiche ou masque les sphères colorées (doigts) et les vecteurs blancs (normales).
*   **SDF Threshold Offset** (Slider : -0.5 à 0.5) : Biais appliqué à l'isovaleur finale avant la voxélisation.
    *   *Impact* : Permet d'ajuster l'épaisseur du volume voxélisé reconstruit pour qu'il matche mieux visuellement le volume réel.

### 🔧 Poisson Parameters
*   **Grid Resolution** (Slider : 16 à 96) : Définit le découpage interne en petits cubes ($N \times N \times N$) de l'espace pour la Transformée de Fourier Rapide. 
    *   *Impact* : Plus haut = formes plus fidèles et arêtes plus nettes, mais calcul plus gourmand en RAM/CPU.
*   **Bounding Box Padding (mm)** (Slider : 5 à 50) : Distance en millimètres rajoutée au-delà des points de contact les plus externes pour créer la boîte de calcul.
    *   *Impact* : Empêche les coupures brutales du maillage reconstruit sur les bords de la zone de calcul.
*   **Sigma Factor** (Slider : 2.0 à 8.0) : Contrôle l'étalement de l'influence de chaque normale sur le voisinage (largeur du noyau gaussien).
    *   *Impact* : Un score faible rend la surface très anguleuse autour du point. Un score élevé crée un objet plus lisse et plus "gonflé".

---

## 📊 Résultats et Métriques



### 🧊 Voxel Metrics (Onglet 2)
Les calculs sont faits sur la grille commune de $128^3$ (2.1 millions de voxels) dans un espace de $20 \times 20 \times 20$ cm.

*   **IoU (Intersection over Union)** : Le taux de chevauchement. `Voxels_Communs / Total_Voxels_Occupés_Par_L'un_Ou_L'autre`. $1.0$ est le score parfait.
*   **Dice Score** : Similaire à l'IoU mais donnant un peu plus de poids à la surface partagée. Équivalent au F1-Score.
*   **Precision** : Sur 100 voxels prédits par la reconstruction Poisson, combien sont *réellement* à l'intérieur du vrai objet ? (Évalue le risque de sur-prédiction/gonflement).
*   **Recall** : Sur 100 voxels du vrai objet, combien ont *bien été trouvés* par la reconstruction ? (Évalue les zones manquantes de l'objet).

### 📊 Voxel Statistics
*   **GT Voxels** : Nombre total de petits cubes constituant le volume réel.
*   **Pred Voxels** : Nombre total de petits cubes constituant le volume reconstruit.
*   **Intersection** : Nombre de cubes qui se superposent parfaitement.
*   **Union** : Nombre de cubes uniques couverts par au moins l'une des deux représentations.

### 📈 Field Statistics (Onglet 3)
*   **Min/Max/Mean/Std Dev** : Statistiques de base sur le champ scalaire brut sortant de la FFT de Poisson. Donne une indication sur le contraste des densités dans l'espace 3D.
