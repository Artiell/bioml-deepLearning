# 🔢 Projet Deep Learning – Classification MNIST (MLP, Deep NN, CNN)

**Université Claude Bernard Lyon 1 — 2025**  
**Master 2 Intelligence Artificielle**  
**UE : Introduction au Deep Learning**

**Auteurs :**  
- **Youssef Abida**  
- **Nathan Corroller**

Ce projet a été réalisé dans un **cadre universitaire** afin de mettre en œuvre différents modèles profonds en PyTorch : Perceptron, réseau peu profond (Shallow), réseau profond (Deep Network) et réseau de neurones convolutionnel (CNN).  
Il repose sur le dataset **MNIST**, composé d’images 28×28 de chiffres manuscrits.

---

## 🎯 1. Objectifs du projet

Le but du projet est de :

- Implémenter et comprendre les architectures MLP, Deep MLP et CNN.  
- Se familiariser avec **PyTorch** : tenseurs, gradients automatiques, DataLoader, modules, optimizers.  
- Expérimenter différents **couples optimiseur / fonction de perte**.  
- Étudier l’influence des **hyperparamètres**.  
- Comparer les résultats obtenus entre architectures.  

Ce travail reprend les consignes du document pédagogique fourni ― notamment les parties :

1. Perceptron  
2. Shallow network  
3. Deep network  
4. CNN  

---

## 🗂️ 2. Le dataset MNIST

- Images : **28×28**, aplaties en vecteurs de dimension **784**  
- Labels : encodés en **one-hot**  
- Structure des données : (train_images, train_labels), (test_images, test_labels)


Les labels représentent les chiffres 0 à 9.

---

## 🧩 3. Partie 1 – Perceptron

Cette première partie introduit les tenseurs et la mise à jour des poids.

### 📌 Description des tenseurs manipulés

| Nom | Taille | Description |
|------|---------|-------------|
| data_train | (63000, 784) | Images d’entraînement |
| label_train | (63000, 10) | Labels one-hot |
| data_test | (7000, 784) | Images de test |
| label_test | (7000, 10) | Labels test |
| w | (784, 10) | Poids du perceptron |
| b | (1, 10) | Biais |
| x | (batch_size, 784) | Batch d’images |
| y | (batch_size, 10) | Prédictions |
| t | (batch_size, 10) | Labels cibles |
| grad | (batch_size, 10) | Gradient d’erreur |

### 📘 Règle d’apprentissage

w ← w + η * Xᵀ * (t – y)
b ← b + η * sum(t – y)


Cette section permet de comprendre en profondeur la propagation avant et arrière dans PyTorch.

---

## 🌱 4. Partie 2 – Shallow Network (1 couche cachée)

### 🧪 Méthodologie

- Une classe `ShallowNetwork` a été développée.  
- Un ensemble **validation (10 %)** a été créé pour éviter l’overfitting.  
- L’entraînement suit les étapes classiques :  
  mélange des données, découpage en batchs, propagation, perte, backprop, mise à jour.  

### 🔍 GridSearch 1 — SGD + MSELoss  

Paramètres testés :

- η ∈ {0.08, 0.3}  
- batch_size ∈ {10, 30, 64}  
- hidden_size ∈ {512, 768, 1024}  
- epochs ∈ {20, 30}  

### 🔍 GridSearch 2 — Adam + CrossEntropyLoss  

Paramètres testés :

- η ∈ {0.001, 0.0008}  
- batch_size ∈ {32, 64}  
- hidden_size ∈ {512, 768}  
- epochs ∈ {20, 25}  

### 📊 Résultats

- Les deux approches donnent de **très bonnes performances**.  
- SGD + MSE fonctionne bien car les labels sont en one-hot.
- Adam + CrossEntropy converge plus vite mais demande un réglage plus précis.

**Accuracy obtenue : ~98 % sur le test.**

---

## 🧱 5. Partie 3 – Deep Network (MLP profond)

### 🎛️ Expérimentation

4 configurations ont été testées :

| Session | Optimizer | Loss | Objectif |
|---------|-----------|------|-----------|
| GS1 | SGD | MSE | Baseline cohérente avec le shallow |
| GS2 | SGD | CrossEntropy | Tester l’effet du softmax implicite |
| GS3 | Adam | MSE | Tester compatibilité Adam + MSE |
| GS4 | Adam | CrossEntropy | Combinaison la plus courante |

### 🔢 Hyperparamètres explorés

- η ∈ {0.0008, 0.001, 0.01}  
- batch_size ∈ {32, 64, 128}  
- architectures :  
  - [512, 256, 128]  
  - [1024, 768, 512, 256, 128]  
- epochs ∈ {20, 30}  

### 📊 Résultats remarquables

- **SGD + MSE :** stable mais plus lent  
- **SGD + CE :** bonne convergence, learning rate sensible  
- **Adam + MSE :** surprisingly efficient  
- **Adam + CE :** meilleures performances globales  

**Accuracy max : ~96.8 %.**

---

## 🧠 6. Partie 4 – Convolutional Neural Network (CNN)

Le CNN est le modèle le plus performant, exploitant la structure spatiale des images.

### 🏗️ Architecture finale

1. Conv2d(1 → 32, kernel=3, stride=1, padding=1) + ReLU  
2. Conv2d(32 → 64, kernel=3, stride=1, padding=1) + ReLU  
3. MaxPool2d(2×2)  
4. Dropout(0.25)  
5. Fully connected 64×14×14 → 128 + ReLU  
6. Dropout(0.25)  
7. Fully connected 128 → 10  

### ⚙️ Hyperparamètres finaux

| Paramètre | Valeur |
|-----------|--------|
| Learning rate | 0.001 |
| Batch size | 64 |
| Epochs | 9 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |

### 📈 Performances

| Jeu | Loss | Accuracy |
|-----|------|-----------|
| Entraînement | ≈ 0.0156 | 99.41 % |
| Validation | ≈ 0.0364 | 99.10 % |
| Test | ≈ 0.0395 | **99.29 %** |

Le CNN obtient les **meilleurs scores** du projet.

---

## 🥇 7. Comparaison globale des modèles

| Modèle | Acc. Train | Acc. Test | Commentaire |
|--------|------------|------------|--------------|
| Perceptron | 90 % | 85 % | Linéaire, très simple |
| Shallow (SGD/MSE) | 99.5 % | 98.6 % | Très performant |
| Shallow (Adam/CE) | 99.1 % | 98.5 % | Convergence plus rapide |
| Deep (SGD/MSE) | 93.9 % | 93.8 % | Sous-apprentissage |
| Deep (Adam/CE) | 99 % | 98.5 % | Meilleure stabilité |
| CNN | 99.2 % | **98.8 %** | 🏆 Meilleur modèle |

---

## 🔮 8. Perspectives d’amélioration

Deux pistes envisagées mais non implémentées :

### 1. Early Stopping  
- Détection automatique de stagnation de la validation.  
- Évite le surapprentissage.

### 2. Optimisation bayésienne  
- Recherche automatique des hyperparamètres.  
- Plus efficace qu’une gridsearch exhaustive.

---

## 🏁 Conclusion

Ce projet universitaire nous a permis de :

- comprendre les mécanismes internes des réseaux de neurones,  
- maîtriser les outils PyTorch,  
- analyser l’influence des hyperparamètres,  
- comparer différentes architectures (MLP vs CNN),  
- constater l’efficacité des CNN pour les images.

Le travail réalisé montre une progression logique :  
**Perceptron → Shallow → Deep → CNN**,  
avec une montée en complexité et en performance.

---


