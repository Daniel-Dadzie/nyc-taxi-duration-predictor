"""
api/main.py — Group A1 — Taxi Trip Duration Predictor

FastAPI prediction API. Loads the Production model from MLflow on startup.
All four required endpoints are stubbed below.
"""

import os
import logging
import threading
import uuid
from datetime import datetime

import numpy as np
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, field_validator

# ── Config ────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
MODEL_REGISTRY_NAME = os.environ.get("MODEL_REGISTRY_NAME", "group-a1-model")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(title="Group A1 — Taxi Trip Duration Predictor")
logger = logging.getLogger(__name__)

# ── Model state ───────────────────────────────────────────────────
model_lock = threading.Lock()
model_state = {
    "model": None,
    "version": None,
    "stage": None,
    "trained_at": None,
    "metrics": {},
}


def load_production_model():
    """Load the current Production model from MLflow registry."""
    client = MlflowClient()
    versions = client.get_latest_versions(MODEL_REGISTRY_NAME, stages=["Production"])
    if not versions:
        raise RuntimeError(
            f"No Production model found in registry: {MODEL_REGISTRY_NAME}"
        )
    v = versions[0]
    model = mlflow.sklearn.load_model(f"models:/{MODEL_REGISTRY_NAME}/Production")
    run = client.get_run(v.run_id)
    with model_lock:
        model_state["model"] = model
        model_state["version"] = v.version
        model_state["stage"] = "Production"
        model_state["trained_at"] = run.info.start_time
        model_state["metrics"] = run.data.metrics
    print(f"Loaded model version {v.version} from Production")


@app.on_event("startup")
def startup():
    load_production_model()


# ── Request / Response models ─────────────────────────────────────
class PredictRequest(BaseModel):
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    passenger_count: int
    vendor_id: int
    pickup_datetime: datetime

    @field_validator("pickup_longitude", "dropoff_longitude")
    @classmethod
    def validate_longitude(cls, v):
        if not (-74.25 <= v <= -73.70):
            raise ValueError("Longitude must be within NYC bounds (-74.25 to -73.70)")
        return v

    @field_validator("pickup_latitude", "dropoff_latitude")
    @classmethod
    def validate_latitude(cls, v):
        if not (40.49 <= v <= 40.92):
            raise ValueError("Latitude must be within NYC bounds (40.49 to 40.92)")
        return v

    @field_validator("passenger_count")
    @classmethod
    def validate_passengers(cls, v):
        if not (1 <= v <= 6):
            raise ValueError("Passenger count must be between 1 and 6")
        return v

    @field_validator("vendor_id")
    @classmethod
    def validate_vendor_id(cls, v):
        if v not in (1, 2):
            raise ValueError("vendor_id must be either 1 or 2")
        return v

    @field_validator("pickup_datetime", mode="before")
    @classmethod
    def validate_pickup_datetime(cls, v):
        """
        Parse and validate pickup_datetime in strict "YYYY-MM-DD HH:MM:SS" format.
        Ensures invalid timestamps return 422 validation error, not 500.
        """
        if isinstance(v, datetime):
            return v
        if not isinstance(v, str):
            raise ValueError(
                'pickup_datetime must be a string in "YYYY-MM-DD HH:MM:SS" format'
            )
        try:
            return datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
        except ValueError as exc:
            raise ValueError(
                'pickup_datetime must be in "YYYY-MM-DD HH:MM:SS" format (received: {!r})'.format(
                    v
                )
            ) from exc


class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    trip_duration_seconds: float
    trip_duration_minutes: float
    model_version: str


# ── Endpoints ─────────────────────────────────────────────────────


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        422: {"description": "Invalid input"},
        500: {"description": "Internal prediction error"},
        503: {"description": "Model not loaded"},
    },
)
def predict(payload: PredictRequest, http_request: Request):
    """Main prediction endpoint."""
    import pandas as pd
    from src.features import clean, engineer, get_feature_columns

    with model_lock:
        model = model_state["model"]
        version = model_state["version"]

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Build DataFrame from request
        df = pd.DataFrame(
            [
                {
                    "pickup_longitude": payload.pickup_longitude,
                    "pickup_latitude": payload.pickup_latitude,
                    "dropoff_longitude": payload.dropoff_longitude,
                    "dropoff_latitude": payload.dropoff_latitude,
                    "passenger_count": payload.passenger_count,
                    "pickup_datetime": payload.pickup_datetime,
                    "vendor_id": payload.vendor_id,
                }
            ]
        )

        # Apply same preprocessing as training
        df = clean(df)

        if df.empty:
            raise HTTPException(
                status_code=422, detail="Input coordinates are outside NYC bounds"
            )

        df = engineer(df)
        feature_cols = get_feature_columns()
        log_prediction = float(model.predict(df[feature_cols])[0])
        prediction = float(np.expm1(log_prediction))

        # Prevent unrealistic predictions
        prediction = max(60.0, min(prediction, 10800.0))

        return PredictResponse(
            trip_duration_seconds=round(prediction, 2),
            trip_duration_minutes=round(prediction / 60, 2),
            model_version=str(version),
        )

    except HTTPException:
        raise
    except Exception:
        correlation_id = (
            http_request.headers.get("x-correlation-id")
            or http_request.headers.get("x-request-id")
            or str(uuid.uuid4())
        )
        logger.exception("Prediction error | correlation_id=%s", correlation_id)
        raise HTTPException(
            status_code=500, detail="Prediction failed. Please try again later."
        )


@app.get("/health")
def health():
    """Health check — must respond in under 100ms."""
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    with model_lock:
        metrics = model_state["metrics"]
        return {
            "model_name": MODEL_REGISTRY_NAME,
            "version": model_state["version"],
            "stage": model_state["stage"],
            "trained_at": model_state["trained_at"],
            "primary_metric": "RMSE",
            "primary_metric_value": metrics.get("primary_metric", None),
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
