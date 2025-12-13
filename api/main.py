from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import logging
import time
import pickle

# --- Configuration du Logging (MLOps) ---
# Enregistre les logs dans un fichier "api_predictions.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_predictions.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('FraudDetectorAPI')

# --- Définition des Chemins ---
MODEL_PATH = 'model_artefacts/xgb_fraud_detector.joblib'
FEATURES_PATH = 'model_artefacts/features.pkl'

# --- Chargement des Artéfacts ---
try:
    # 1. Charger le Modèle Pipeline
    model = joblib.load(MODEL_PATH)
    
    # 2. Charger la liste des features
    with open(FEATURES_PATH, 'rb') as f:
        FEATURE_COLUMNS = pickle.load(f)
    
    logger.info("Modèle et liste des features chargés avec succès.")
    logger.info(f"Le modèle attend les {len(FEATURE_COLUMNS)} features suivantes : {FEATURE_COLUMNS}")
    
except Exception as e:
    logger.error(f"Erreur critique lors du chargement des artéfacts : {e}")
    model = None
    FEATURE_COLUMNS = []


# =================================================================
# ⚠️ ACTION REQUISE : ADAPTER LE MODÈLE PYDANTIC CI-DESSOUS
# =================================================================

# La classe Pydantic doit contenir TOUS les éléments de FEATURE_COLUMNS
# avec le bon type (int, float). Elle servira de contrat d'entrée.
# Modifiez cette classe en fonction de la sortie du logger ci-dessus !
class TransactionInput(BaseModel):
    # Exemple basé sur un dataset de fraude générique. 
    # REMPLACEZ TOUS LES CHAMPS CI-DESSOUS par les noms exacts et l'ordre de votre liste FEATURE_COLUMNS
    
    # 1. Champs numériques originaux
    step: int
    amount: float
    oldbalanceOrg: float
    newbalanceOrg: float
    oldbalanceDest: float
    newbalanceDest: float
    
    # 2. Champs 'One-Hot Encoded' (si vous avez encodé le 'type' de transaction)
    # Exemple :
    type_CASH_OUT: int = 0
    type_TRANSFER: int = 0
    # Ajoutez toutes les colonnes issues de votre encodage ici...


app = FastAPI(title="Fraud Detection API")

@app.get("/", tags=["Health"])
def home():
    return {"message": "Fraud Detection API est opérationnelle."}

@app.post("/predict_fraud", tags=["Prediction"])
def predict_fraud(transaction: TransactionInput):
    """
    Accepte une transaction et retourne la probabilité de fraude et la prédiction (True/False).
    """
    if model is None:
        return {"error": "Le modèle n'est pas chargé. Vérifiez les logs du serveur."}

    start_time = time.time()
    
    # 1. Préparation des données pour le modèle
    input_data = transaction.model_dump()
    
    # Créer un DataFrame avec les colonnes dans l'ordre EXACT attendu
    df_input = pd.DataFrame([input_data])
    
    # S'assurer que les colonnes du DF sont dans l'ordre de FEATURE_COLUMNS
    try:
        df_input = df_input[FEATURE_COLUMNS]
    except KeyError as e:
        logger.error(f"Erreur de colonne : La donnée d'entrée ne contient pas la feature {e}")
        return {"error": f"Donnée d'entrée manquante : {e}"}

    # 2. Prédiction
    probability = model.predict_proba(df_input)[:, 1][0]
    prediction = model.predict(df_input)[0]

    end_time = time.time()
    response_time = end_time - start_time
    
    # 3. Logging de la Requête (MLOps)
    logger.info(
        f"PREDICTION | Pred: {bool(prediction)} | Prob: {probability:.4f} | Time: {response_time:.4f}s | Data: {input_data}"
    )

    # 4. Retourner le résultat
    return {
        "is_fraud": bool(prediction),
        "fraud_probability": float(probability),
        "prediction_time_ms": response_time * 1000
    }