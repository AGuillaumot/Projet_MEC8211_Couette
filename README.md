# Projet MEC8211 – Vérification et Validation d'une simulation d’un Écoulement de Couette

Ce projet implémente une simulation numérique d'écoulement bidimensionnel basé sur l'algorithme **SIMPLE**, dans le cadre du cours de Vérification et Validation (**MEC8211**, Polytechnique Montréal).

L'algorithme est appliqué au cas de Couette en régime permanent et incompressible, avec visualisation des champs de vitesse et de pression.

Les fichiers mesh, meshConnectivity, meshGenerator ont été réalisés par El Hadji Abdou Aziz NDIAYE dans le cadre du cours Aérodynamique numérique (**MEC6616**, Polytechnique Montréal).

---

## 🗂️ Structure du projet

```
Projet_MEC8211_Couette-main/
├── bin/                # Scripts exécutables
│   └── run_simulation.py
├── simulations/        # Fichiers de test (Couette)
│   └── couette_test.py
├── src/                # Modules Python contenant le cœur du code
│   ├── simulation_core.py # Implémente l'algorithme SIMPLE
│   ├── monPlotter.py
│   ├── monGradient.py
│   └── ...
│
├── doc/                     # Documentation, rapport, explication du projet
│
├── data/                    # Données brutes 
│
├── results/                 # Résultats : fichiers, images, logs, figures
├── requirements.txt    # Dépendances Python
└── README.md
```

---

## 🚀 Utilisation

Pour exécuter une simulation :

```bash
python bin/run_simulation.py
```

Le fichier `run_simulation.py` appelle le scénario défini dans `simulations/couette_test.py`.

Tu peux modifier les **paramètres physiques, maillage ou conditions aux limites** dans ce fichier.

---
