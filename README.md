# Bank Marketing — Model Serving & Monitoring

**Stack:** scikit-learn 1.5 · MLflow 2.16 · FastAPI 0.111 · Evidently AI 0.4

## What this project demonstrates
End-to-end MLE workflow: train → register → serve → monitor.
Focus is on operationalizing a model, not just training one.

## Architecture
1. Data preparation — UCI Bank Marketing, feature engineering, leakage removal
2. Model training — GradientBoostingClassifier, logged to MLflow Model Registry
3. Model serving — REST API via FastAPI, model loaded from MLflow at startup
4. Drift monitoring — Evidently AI report on simulated production traffic

## Key decisions
- `duration` dropped: post-call feature unavailable at prediction time (leakage)
- MLflow 2.x chosen over 3.x: matches version deployed at most organizations
- GradientBoostingClassifier: better probability calibration than RandomForest
  (validated with calibration curve, see notebook 02)
- Evidently 0.4.x: stable report API with more community examples than 0.5.x

## How to run
```bash
python -m venv .venv && .venv\Scripts\Activate.ps1
pip install -r requirements.txt
mlflow ui                          # Terminal 1 → http://localhost:5000
uvicorn app.main:app --port 8000   # Terminal 2 → http://localhost:8000
# Run notebooks 01 → 02 → 03 → 04 in Jupyter
```

## Dataset
UCI Bank Marketing (bank-additional-full.csv)
Binary classification: predict term deposit subscription.
