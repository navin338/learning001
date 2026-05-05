import os
import sys
import traceback
import pandas as pd
import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# =========================
# Config (Environment Driven)
# =========================
# Default to host.docker.internal if not set. 
# This works on Docker Desktop (Windows/Mac) to reach your host machine.
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:30007")
MODEL_NAME = "iris-classifier"

class IrisRequest(BaseModel):
    features: list[float]

# =============
# Model Loading
# =============
mlflow.set_tracking_uri(MLFLOW_URI)
print(f"Connecting to MLflow at: {MLFLOW_URI}...")

try:
    # Try loading the latest version
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/latest")
    if model is None:
        raise Exception("Model load returned None")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {str(e)}")
    traceback.print_exc()
    model = None

# ===========
# FastAPI App
# ===========
app = FastAPI(title="Iris API - Kind Edition")

@app.get("/")
def home():
    return {"status": "online", "model_loaded": model is not None, "mlflow_uri": MLFLOW_URI}

@app.post("/predict")
def predict(request: IrisRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check Pod logs.")

    try:
        if len(request.features) != 4:
            raise HTTPException(status_code=400, detail="4 features required")

        df = pd.DataFrame([request.features], columns=[
            "sepal length (cm)", "sepal width (cm)", 
            "petal length (cm)", "petal width (cm)"
        ])

        prediction = model.predict(df)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))