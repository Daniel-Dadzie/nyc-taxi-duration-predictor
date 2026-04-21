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

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, field_validator
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# ── Config ────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
MODEL_REGISTRY_NAME = os.environ.get("MODEL_REGISTRY_NAME", "group-a1-model")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(
    title="Group A1 — Taxi Trip Duration Predictor",
    description="""
    Predicts NYC taxi trip duration given pickup and dropoff details.
    
    ## Features
    - Predict by GPS coordinates
    - Predict by street address (auto-geocoding)
    - Only supports NYC metropolitan area
    - Returns duration in seconds and minutes
    
    ## Coverage
    - Manhattan, Brooklyn, Queens, Bronx, Staten Island
    - JFK, LaGuardia and Newark airports
    """,
    version="1.0.0",
)
logger = logging.getLogger(__name__)

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
    versions = client.get_latest_versions(
        MODEL_REGISTRY_NAME, stages=["Production"]
    )
    if not versions:
        raise RuntimeError(
            f"No Production model found in registry: {MODEL_REGISTRY_NAME}"
        )
    v = versions[0]
    model = mlflow.sklearn.load_model(
        f"models:/{MODEL_REGISTRY_NAME}/Production"
    )
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


# ── Helper functions ──────────────────────────────────────────────

def is_in_nyc(latitude: float, longitude: float) -> bool:
    """Check if coordinates are within NYC bounding box."""
    return (
        -74.25 <= longitude <= -73.70 and
        40.49  <= latitude  <= 40.92
    )


def parse_datetime(v) -> str:
    """
    Parse and validate datetime string.
    Accepts both:
    - "2016-06-12 00:43:35" (space separator)
    - "2016-06-12T00:43:35" (T separator)
    If None, returns current datetime.
    """
    if v is None:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not isinstance(v, str):
        raise ValueError(
            'pickup_datetime must be a string in "YYYY-MM-DD HH:MM:SS" format'
        )
    # Accept both space and T separator
    v = v.replace("T", " ")
    try:
        datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
        return v
    except ValueError as exc:
        raise ValueError(
            'pickup_datetime must match format "YYYY-MM-DD HH:MM:SS" '
            'e.g. "2016-06-12 08:30:00"'
        ) from exc


def geocode_address(address: str):
    """
    Convert street address to GPS coordinates using Nominatim.

    Args:
        address: Street address string

    Returns:
        Location object with latitude and longitude

    Raises:
        HTTPException if service times out or is unavailable
    """
    try:
        geolocator = Nominatim(user_agent="group-a1-taxi-predictor")
        location   = geolocator.geocode(address, timeout=10)
        return location
    except GeocoderTimedOut:
        raise HTTPException(
            status_code=503,
            detail="Geocoding service timed out. Please try again."
        )
    except GeocoderServiceError:
        raise HTTPException(
            status_code=503,
            detail="Geocoding service unavailable. Please try again later."
        )


# ── Request / Response models ─────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Prediction request using GPS coordinates.
    pickup_datetime is optional — defaults to current time if not provided.
    """
    pickup_longitude:  float
    pickup_latitude:   float
    dropoff_longitude: float
    dropoff_latitude:  float
    pickup_datetime:   str = None  # optional — defaults to now

    @field_validator("pickup_longitude", "dropoff_longitude")
    @classmethod
    def validate_longitude(cls, v):
        if not (-74.25 <= v <= -73.70):
            raise ValueError(
                "Longitude must be within NYC bounds (-74.25 to -73.70)"
            )
        return v

    @field_validator("pickup_latitude", "dropoff_latitude")
    @classmethod
    def validate_latitude(cls, v):
        if not (40.49 <= v <= 40.92):
            raise ValueError(
                "Latitude must be within NYC bounds (40.49 to 40.92)"
            )
        return v

    @field_validator("pickup_datetime", mode="before")
    @classmethod
    def validate_pickup_datetime(cls, v):
        return parse_datetime(v)


class PredictRequestByAddress(BaseModel):
    """
    Prediction request using street addresses.
    Addresses are automatically converted to GPS coordinates.
    pickup_datetime is optional — defaults to current time if not provided.
    """
    pickup_address:  str
    dropoff_address: str
    pickup_datetime: str = None  # optional — defaults to now

    @field_validator("pickup_datetime", mode="before")
    @classmethod
    def validate_pickup_datetime(cls, v):
        return parse_datetime(v)


class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    trip_duration_seconds: float
    trip_duration_minutes: float
    model_version:         str


# ── Endpoints ─────────────────────────────────────────────────────

@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict trip duration by coordinates",
    responses={
        422: {"description": "Invalid input — coordinates out of bounds"},
        500: {"description": "Internal prediction error"},
        503: {"description": "Model not loaded"},
    },
)
def predict(payload: PredictRequest, http_request: Request):
    """
    Predict NYC taxi trip duration using GPS coordinates.

    - **pickup_longitude**: must be between -74.25 and -73.70
    - **pickup_latitude**: must be between 40.49 and 40.92
    - **dropoff_longitude**: must be between -74.25 and -73.70
    - **dropoff_latitude**: must be between 40.49 and 40.92
    - **pickup_datetime**: optional — format "YYYY-MM-DD HH:MM:SS". 
      Defaults to current time if not provided.
    """
    import pandas as pd
    from src.features import clean, engineer, get_feature_columns

    with model_lock:
        model   = model_state["model"]
        version = model_state["version"]

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        df = pd.DataFrame([{
            "pickup_longitude":  payload.pickup_longitude,
            "pickup_latitude":   payload.pickup_latitude,
            "dropoff_longitude": payload.dropoff_longitude,
            "dropoff_latitude":  payload.dropoff_latitude,
            "pickup_datetime":   pd.to_datetime(payload.pickup_datetime),
        }])

        df = clean(df)

        if df.empty:
            raise HTTPException(
                status_code=422,
                detail="Input coordinates are outside NYC bounds"
            )

        df = engineer(df)
        feature_cols   = get_feature_columns()
        log_prediction = float(model.predict(df[feature_cols])[0])
        prediction     = float(np.expm1(log_prediction))
        prediction     = max(60.0, min(prediction, 10800.0))

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
        logger.exception(
            "Prediction error | correlation_id=%s", correlation_id
        )
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please try again later."
        )


@app.post(
    "/predict-by-address",
    response_model=PredictResponse,
    summary="Predict trip duration by street address",
    responses={
        422: {"description": "Invalid input or address outside NYC"},
        500: {"description": "Internal prediction error"},
        503: {"description": "Model not loaded or geocoding unavailable"},
    },
)
def predict_by_address(
    request: PredictRequestByAddress,
    http_request: Request
):
    """
    Predict NYC taxi trip duration using street addresses.
    Automatically converts addresses to GPS coordinates.

    - **pickup_address**: NYC pickup address e.g. "Times Square, New York"
    - **dropoff_address**: NYC dropoff address e.g. "JFK Airport, New York"
    - **pickup_datetime**: optional — format "YYYY-MM-DD HH:MM:SS".
      Defaults to current time if not provided.

    Both addresses must be within New York City.
    """
    # Geocode pickup
    pickup = geocode_address(request.pickup_address)
    if not pickup:
        raise HTTPException(
            status_code=422,
            detail=f"Could not find pickup address: '{request.pickup_address}'. "
                   f"Please enter a valid NYC address."
        )

    # Geocode dropoff
    dropoff = geocode_address(request.dropoff_address)
    if not dropoff:
        raise HTTPException(
            status_code=422,
            detail=f"Could not find dropoff address: '{request.dropoff_address}'. "
                   f"Please enter a valid NYC address."
        )

    # Validate pickup is in NYC
    if not is_in_nyc(pickup.latitude, pickup.longitude):
        raise HTTPException(
            status_code=422,
            detail=f"Pickup address '{request.pickup_address}' is outside NYC. "
                   f"This predictor only supports NYC taxi trips. "
                   f"Please enter an address in New York City."
        )

    # Validate dropoff is in NYC
    if not is_in_nyc(dropoff.latitude, dropoff.longitude):
        raise HTTPException(
            status_code=422,
            detail=f"Dropoff address '{request.dropoff_address}' is outside NYC. "
                   f"This predictor only supports NYC taxi trips. "
                   f"Please enter an address in New York City."
        )

    # Build PredictRequest from coordinates
    predict_request = PredictRequest(
        pickup_longitude=pickup.longitude,
        pickup_latitude=pickup.latitude,
        dropoff_longitude=dropoff.longitude,
        dropoff_latitude=dropoff.latitude,
        pickup_datetime=request.pickup_datetime
    )

    # Reuse existing predict logic
    return predict(predict_request, http_request)


@app.get(
    "/health",
    summary="Health check"
)
def health():
    """
    Health check endpoint.
    Must respond in under 100ms.
    Used by Cloud Run to verify the container is running.
    """
    return {"status": "ok"}


@app.get(
    "/model-info",
    summary="Current model metadata"
)
def model_info():
    """
    Returns metadata about the current Production model.
    Includes model version, training time and performance metrics.
    """
    with model_lock:
        metrics = model_state["metrics"]
        return {
            "model_name":           MODEL_REGISTRY_NAME,
            "version":              model_state["version"],
            "stage":                model_state["stage"],
            "trained_at":           model_state["trained_at"],
            "primary_metric":       "RMSE",
            "primary_metric_value": metrics.get("primary_metric", None),
        }


@app.get(
    "/coverage",
    summary="Geographic coverage area"
)
def coverage():
    """
    Returns the geographic coverage area of this predictor.
    Use this to check if your trip is within the supported area
    before making a prediction.
    """
    return {
        "supported_city": "New York City, USA",
        "boroughs": [
            "Manhattan",
            "Brooklyn",
            "Queens",
            "Bronx",
            "Staten Island"
        ],
        "nearby_areas_supported": [
            "Newark, New Jersey (airport trips only)"
        ],
        "airports_supported": [
            {
                "name":    "John F. Kennedy International Airport (JFK)",
                "borough": "Queens, NYC"
            },
            {
                "name":    "LaGuardia Airport (LGA)",
                "borough": "Queens, NYC"
            },
            {
                "name":    "Newark Liberty International Airport (EWR)",
                "borough": "Newark, New Jersey",
                "note":    "Outside NYC but supported for airport trips"
            }
        ],
        "bounding_box": {
            "longitude": {"min": -74.25, "max": -73.70},
            "latitude":  {"min": 40.49,  "max": 40.92}
        },
        "data_trained_on": "NYC Yellow Taxi trips — January to June 2016",
        "note": "Only trips within the NYC metropolitan area are supported."
    }


@app.post(
    "/reload-model",
    summary="Hot-swap Production model"
)
def reload_model():
    """
    Hot-swaps to the latest Production model without container restart.
    Called automatically by Cloud Scheduler after retraining.
    """
    def _reload():
        try:
            load_production_model()
        except Exception as e:
            print(f"Reload failed: {e}")

    threading.Thread(target=_reload, daemon=True).start()
    return {"status": "reloading"}