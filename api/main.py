"""
api/main.py — Group A1 — Taxi Trip Duration Predictor

FastAPI prediction API. Loads the Production model from MLflow on startup.
All four required endpoints are stubbed below.
"""

import os
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import threading

# ── Config ────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
MODEL_REGISTRY_NAME = os.environ.get("MODEL_REGISTRY_NAME", "group-a1-model")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(title="Group A1 — Taxi Trip Duration Predictor")

# ── Model state ───────────────────────────────────────────────────
model_lock = threading.Lock()
model_state = {
    "model":      None,
    "version":    None,
    "stage":      None,
    "trained_at": None,
    "metrics":    {},
}


def load_production_model():
    """Load the current Production model from MLflow registry."""
    client = MlflowClient()
    versions = client.get_latest_versions(MODEL_REGISTRY_NAME, stages=["Production"])
    if not versions:
        raise RuntimeError(f"No Production model found in registry: {MODEL_REGISTRY_NAME}")
    v = versions[0]
    model = mlflow.sklearn.load_model(f"models:/{MODEL_REGISTRY_NAME}/Production")
    run = client.get_run(v.run_id)
    with model_lock:
        model_state["model"]      = model
        model_state["version"]    = v.version
        model_state["stage"]      = "Production"
        model_state["trained_at"] = run.info.start_time
        model_state["metrics"]    = run.data.metrics
    print(f"Loaded model version {v.version} from Production")


@app.on_event("startup")
def startup():
    load_production_model()


# ── Request / Response models ─────────────────────────────────────
class PredictRequest(BaseModel):
    # TODO: define input fields for Project A
    pass

class PredictResponse(BaseModel):
    # TODO: define output fields for Project A
    model_version: str


# ── Endpoints ─────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Main prediction endpoint."""
    from src.features import clean, engineer
    # TODO: convert request to DataFrame, call features, call model.predict()
    raise HTTPException(status_code=501, detail="Not implemented")


@app.get("/health")
def health():
    """Health check — must respond in under 100ms."""
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    """Return current Production model metadata."""
    with model_lock:
        return {
            "model_name":   MODEL_REGISTRY_NAME,
            "version":      model_state["version"],
            "stage":        model_state["stage"],
            "trained_at":   model_state["trained_at"],
            "metrics":      model_state["metrics"],
        }


@app.post("/reload-model")
def reload_model():
    """Hot-swap to the latest Production model without container restart."""
    def _reload():
        try:
            load_production_model()
        except Exception as e:
            print(f"Reload failed: {e}")
    threading.Thread(target=_reload, daemon=True).start()
    return {"status": "reloading"}
