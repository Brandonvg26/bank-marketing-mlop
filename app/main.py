# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import numpy as np
from typing import List


# ── Configuration ───────────────────────────────────────────────────
MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'
MODEL_NAME          = 'BankMarketingClassifier'
MODEL_VERSION       = '1'


# ── Load model at startup ────────────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model_uri = f'models:/{MODEL_NAME}/{MODEL_VERSION}'
model = mlflow.sklearn.load_model(model_uri)
print(f'Model loaded: {model_uri}')


# ── Feature list (must match training order) ─────────────────────────
# Load column order from saved training data
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURE_COLS = pd.read_csv(r'C:\Users\brand\OneDrive\Imágenes\Escritorio\Git Portfolio\bank-marketing-mlop\data\train\X_train.csv', nrows=0).columns.tolist()


# ── Schema ──────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    features: List[float]   # One row as flat list in FEATURE_COLS order


class PredictionResponse(BaseModel):
    predicted_class: int
    probability_subscribe: float


# ── App ─────────────────────────────────────────────────────────────
app = FastAPI(
    title='Bank Marketing Classifier API',
    description='Serves a GradientBoostingClassifier registered in MLflow.',
    version='1.0.0'
)


@app.get('/health')
def health():
    return {'status': 'ok', 'model': MODEL_NAME, 'version': MODEL_VERSION}


@app.post('/predict', response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if len(request.features) != len(FEATURE_COLS):
        raise HTTPException(
            status_code=422,
            detail=f'Expected {len(FEATURE_COLS)} features, got {len(request.features)}'
        )
    X = pd.DataFrame([request.features], columns=FEATURE_COLS)
    pred  = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])
    return PredictionResponse(
        predicted_class=pred,
        probability_subscribe=round(proba, 4)
    )