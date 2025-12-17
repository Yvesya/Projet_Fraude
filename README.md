#Détection de Fraudes Bancaires (Pipeline MLOps)

**Contexte**
Les fraudes bancaires coûtent des milliards chaque année. L’objectif de ce projet est de détecter automatiquement les transactions frauduleuses dans un dataset réel de transactions bancaires synthétiques, tout en proposant une solution industrialisée via un pipeline MLOps.

**Dataset**
 Le dataset utilisé provient de Kaggle : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
 Format CSV, 284807 transactions
 30 features (V1-V28, Time, Amount)
 Classe cible Class :
 0 → Transaction normale
 1 → Transaction frauduleuse (~0.17% des transactions)

**Objectifs**
 1-Analyser les transactions et détecter les patterns suspects
 2-Construire un modèle de classification robuste (Random Forest, XGBoost)
 3-Gérer le déséquilibre extrême des classes (SMOTE)
 4-Déployer une API pour prédire les fraudes en temps réel
 5-Intégrer un système de logs pour le suivi des prédictions

 **Méthodologie**
  1.Analyse exploratoire (EDA)
   Visualisation de la distribution des classes
   Étude des correlations et des patterns des features

  2. Prétraitement
   Séparation des features (X) et de la cible (y)
   Split train/test stratifié (70% / 30%)
   Gestion des variables catégorielles avec pd.get_dummies()
   Rééchantillonnage de la classe minoritaire via SMOTE

  3. Modélisation
    *Random Forest*
    Pipeline avec StandardScaler et RandomForestClassifier
    Poids équilibrés pour la classe minoritaire
    *XGBoost* (modèle final)
    Pipeline avec StandardScaler et XGBClassifier
    Hyperparamètres : n_estimators=100, eval_metric='logloss'

  4. Évaluation
    Métriques principales :
    Recall sur la classe fraude (minimiser les fausses négatives)
    AUC-ROC pour la performance globale

Comparaison des modèles :
| Modèle                | Recall (Fraude) | AUC-ROC Score |
| --------------------- | --------------- | ------------- |
| Random Forest         | 0.7905          | 0.9494        |
| XGBoost (sélectionné) | **0.7973**      | **0.9779**    |


**Déploiement et MLOps**
Architecture
 1-Le modèle XGBoost entraîné est sérialisé en xgb_fraud_detector.joblib
 2-La liste des features est sauvegardée dans features.pkl
 3-Une API FastAPI est créée (api/main.py)
   .Schéma d’entrée validé avec Pydantic
   .Prédiction en temps réel sur les 30 features
 4-Logging MLOps :
   .Chaque requête, la prédiction et le temps de réponse sont enregistrés dans api_predictions.log
   .Préparation pour le suivi de la dérive des données (concept drift)
   
Lancer l’API
Prérequis : Docker Desktop installé et lancé
# Construire l'image Docker
docker build -t fraude-api .

# Lancer le conteneur
docker run -d --name fraude-container -p 8000:8000 fraude-api

# Tester l'API via Swagger
http://localhost:8000/docs


Structure du projet
Projet_Fraude/
├── api/
│   └── main.py                 # Code FastAPI + Logging
├── model_artefacts/
│   ├── xgb_fraud_detector.joblib
│   └── features.pkl
├── notebooks/
│   └── Detection_de_Fraude.ipynb      # Notebook complet (EDA, Modélisation)
├── Dockerfile
└── requirements.txt


**Conclusion**
XGBoost est le modèle retenu pour sa performance sur les fraudes (Recall et AUC-ROC élevés)
Le projet est industrialisation-ready grâce à Docker et FastAPI
Les logs assurent une traçabilité des prédictions, base pour un futur suivi de dérive
