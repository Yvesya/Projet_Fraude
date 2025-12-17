from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import logging
import time
import pickle

#  Configuration du Logging (MLOps) 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_predictions.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('FraudDetectorAPI')

#  Définition des Chemins 
MODEL_PATH = 'model_artefacts/xgb_fraud_detector.joblib' # le modele charger a ete entraine dans colab et sauvegarde avec joblib
FEATURES_PATH = 'model_artefacts/features.pkl'   #liens vers le ficher colab( https://colab.research.google.com/drive/1mcGX9wPXm6ed9dGJTbrnpsard2ShkOBe?usp=sharing ) 
try:
    # 1. Chargement du Modèle Pipeline
    model = joblib.load(MODEL_PATH)
    
    # 2. Chargement de la liste des features
    with open(FEATURES_PATH, 'rb') as f:
        FEATURE_COLUMNS = pickle.load(f)
    
    logger.info("Modèle et liste des features chargés avec succès.")
    logger.info(f"Le modèle attend les {len(FEATURE_COLUMNS)} features suivantes : {FEATURE_COLUMNS}")
    
except Exception as e:
    logger.error(f"Erreur critique lors du chargement des artéfacts : {e}")
    model = None
    FEATURE_COLUMNS = []

#  Définition du Modèle de Données pour l'API 
class TransactionInput(BaseModel):
    Time: int
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    
#  Creation de l'Application FastAPI

app = FastAPI(title="Detection de Fraude API")

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
    df_input = pd.DataFrame([input_data])
    
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

    # 4. Retourne le résultat
    return {
        "is_fraud": bool(prediction),
        "fraud_probability": float(probability),
        "prediction_time_ms": response_time * 1000
    }