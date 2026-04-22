"""
api/main.py — Group A1 — Taxi Trip Duration Predictor

FastAPI prediction API. Loads the Production model from MLflow on startup.

Validation coverage:
  - GPS coordinates: bounds-checked against NYC bounding box
  - Street addresses: geocoded and bounds-checked; misspelling suggestions provided
  - pickup_datetime: optional, accepts both space and T separator
  - All endpoints: typed error responses with actionable messages
"""

import os
import logging
import threading
import uuid
import time
from datetime import datetime

from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

import numpy as np
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, field_validator
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# ── Config ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

if not MLFLOW_TRACKING_URI:
    # Fallback for testing environments (CI/CD)
    MLFLOW_TRACKING_URI = "http://localhost:5000"
    
MODEL_REGISTRY_NAME = os.environ.get("MODEL_REGISTRY_NAME", "group-a1-model")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# NYC strict bounding box (longitude, latitude)
NYC_LON_MIN, NYC_LON_MAX = -74.25, -73.70
NYC_LAT_MIN, NYC_LAT_MAX = 40.49, 40.92

# Viewbox for Nominatim: list of (lon, lat) corners  [top-left, bottom-right]
NYC_VIEWBOX = [
    (NYC_LON_MIN, NYC_LAT_MAX),  # top-left
    (NYC_LON_MAX, NYC_LAT_MIN),  # bottom-right
]

# Newark Airport bounding box (special supported case)
EWR_LON_MIN, EWR_LON_MAX = -74.19, -74.15
EWR_LAT_MIN, EWR_LAT_MAX = 40.67, 40.71

# Known NYC neighbourhoods / landmarks for fuzzy-match hints
NYC_KNOWN_PLACES = [
    "Manhattan",
    "Brooklyn",
    "Queens",
    "Bronx",
    "Staten Island",
    "Harlem",
    "Midtown",
    "Downtown",
    "Uptown",
    "Financial District",
    "Chelsea",
    "Greenwich Village",
    "SoHo",
    "Tribeca",
    "Chinatown",
    "Lower East Side",
    "Upper East Side",
    "Upper West Side",
    "Flushing",
    "Astoria",
    "Long Island City",
    "Jamaica",
    "JFK Airport",
    "LaGuardia Airport",
    "Newark Airport",
    "Times Square",
    "Central Park",
    "Empire State Building",
    "Grand Central",
    "Penn Station",
    "Wall Street",
    "Williamsburg",
    "Bushwick",
    "Crown Heights",
    "Flatbush",
    "Bay Ridge",
    "Coney Island",
    "Bensonhurst",
]

# Shared geocoder instance
geolocator = Nominatim(user_agent="group-a1-taxi-predictor")


# ── Lifespan ──────────────────────────────────────────────────────────────────
# FastAPI now recommends lifespan handlers instead of @app.on_event("startup").
# We load the Production model once when the application starts serving.
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_production_model()
    yield
    # No shutdown cleanup is required yet, but this block is kept so future
    # resource cleanup can be added here without changing the app structure.


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    lifespan=lifespan,
    title="Group A1 — Taxi Trip Duration Predictor",
    description="""
Predicts NYC taxi trip duration given pickup and dropoff details.

## Endpoints
| Method | Path | Description |
|--------|------|-------------|
| POST | `/predict` | Predict by GPS coordinates |
| POST | `/predict-by-address` | Predict by street address (auto-geocoded) |
| GET  | `/health` | Liveness check (< 100 ms) |
| GET  | `/model-info` | Current model metadata |
| GET  | `/coverage` | Geographic coverage info |
| POST | `/reload-model` | Hot-swap Production model |

## Coverage
- Manhattan, Brooklyn, Queens, Bronx, Staten Island
- JFK, LaGuardia, and Newark airports
- Bounding box: lon −74.25 → −73.70 | lat 40.49 → 40.92
""",
    version="1.0.0",
)

logger = logging.getLogger(__name__)

# ── Model state ───────────────────────────────────────────────────────────────
model_lock = threading.Lock()
model_state: dict = {
    "model": None,
    "version": None,
    "stage": None,
    "trained_at": None,
    "metrics": {},
}


# ── Global exception handlers ─────────────────────────────────────────────────


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Override FastAPI's default 422 handler so every field error returns a
    clean, human-readable message instead of the raw Pydantic error tree.
    """
    errors = []
    for error in exc.errors():
        field = " → ".join(str(loc) for loc in error["loc"] if loc != "body")
        errors.append({"field": field, "message": error["msg"]})
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation failed",
            "details": errors,
            "hint": "Check the field values and try again.",
        },
    )


# ── Model loader ──────────────────────────────────────────────────────────────


def load_production_model() -> None:
    """Load the current Production model from the MLflow registry."""
    client = MlflowClient()
    versions = client.get_latest_versions(MODEL_REGISTRY_NAME, stages=["Production"])
    if not versions:
        raise RuntimeError(
            f"No Production model found in registry '{MODEL_REGISTRY_NAME}'. "
            "Promote a model run to Production in the MLflow UI first."
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
    logger.info("Loaded model version %s from Production.", v.version)


# ── Helpers ───────────────────────────────────────────────────────────────────


def is_in_nyc(latitude: float, longitude: float) -> bool:
    """Return True if the coordinate falls inside the NYC bounding box."""
    return (
        NYC_LON_MIN <= longitude <= NYC_LON_MAX
        and NYC_LAT_MIN <= latitude <= NYC_LAT_MAX
    )


def is_newark_airport(latitude: float, longitude: float) -> bool:
    """Return True if the coordinate falls inside the Newark Airport box."""
    return (
        EWR_LON_MIN <= longitude <= EWR_LON_MAX
        and EWR_LAT_MIN <= latitude <= EWR_LAT_MAX
    )


def is_supported_location(latitude: float, longitude: float) -> bool:
    """
    Return True if the coordinate falls inside the supported area.

    Supported area includes:
      - NYC bounding box
      - Newark Airport special-case box
    """
    return is_in_nyc(latitude, longitude) or is_newark_airport(latitude, longitude)


def _closest_nyc_place(address: str) -> str | None:
    """
    Very lightweight fuzzy hint: return the NYC place name whose lower-case
    form shares the most leading characters with the query.  Used only to
    suggest a correction in error messages — not for geocoding.
    """
    lower = address.lower()
    best_match, best_score = None, 0
    for place in NYC_KNOWN_PLACES:
        place_lower = place.lower()
        # Count common prefix length as a simple similarity score
        score = sum(1 for a, b in zip(lower, place_lower) if a == b)
        # Also give credit for any shared words
        query_words = set(lower.split())
        place_words = set(place_lower.split())
        score += len(query_words & place_words) * 3
        if score > best_score:
            best_score, best_match = score, place
    # Only surface a hint when the score is meaningful
    return best_match if best_score >= 3 else None


def parse_datetime(v) -> str:
    """
    Parse and validate pickup_datetime.

    Accepts:
      - None  → defaults to current datetime
      - "YYYY-MM-DD HH:MM:SS"  (space separator)
      - "YYYY-MM-DDTHH:MM:SS"  (ISO 8601 T separator)

    Returns the value normalised to "YYYY-MM-DD HH:MM:SS".
    Raises ValueError with an actionable message on bad input.
    """
    if v is None:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not isinstance(v, str):
        raise ValueError(
            "pickup_datetime must be a string. " 'Example: "2016-06-12 08:30:00"'
        )

    # Normalise T separator → space
    normalised = v.strip().replace("T", " ")

    try:
        datetime.strptime(normalised, "%Y-%m-%d %H:%M:%S")
        return normalised
    except ValueError:
        raise ValueError(
            f'Invalid pickup_datetime: "{v}". '
            'Expected format: "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DDTHH:MM:SS". '
            'Example: "2016-06-12 08:30:00"'
        )


def geocode_address(address: str):
    """
    Convert a street address to GPS coordinates using Nominatim.

    Strategy:
      1. Strict search — bounded to the NYC bounding box.
      2. Relaxed search — append ", New York City, NY, USA" and validate
         the returned coordinates fall inside NYC.
      3. Relaxed search — allow Newark Airport as a supported special case.

    Returns a geopy Location or None if nothing was found inside the supported area.
    Raises HTTPException 503 if the geocoding service is unavailable.
    """
    try:
        # ── Attempt 1: strict bounding box ────────────────────────────────
        location = geolocator.geocode(
            address,
            timeout=10,
            viewbox=NYC_VIEWBOX,
            bounded=True,
            country_codes="us",
        )
        if location:
            logger.info(
                "Geocoded strict '%s' → (%f, %f) | %s",
                address,
                location.latitude,
                location.longitude,
                location.address,
            )
            if is_supported_location(location.latitude, location.longitude):
                return location

        # ── Attempt 2: relaxed — add NYC context, then bounds-check ───────
        location = geolocator.geocode(
            f"{address}, New York City, NY, USA",
            timeout=10,
            country_codes="us",
        )
        if location:
            logger.info(
                "Geocoded relaxed '%s' → (%f, %f) | %s",
                address,
                location.latitude,
                location.longitude,
                location.address,
            )
            if is_supported_location(location.latitude, location.longitude):
                return location

        # ── Attempt 3: relaxed — add Newark context for airport trips ─────
        location = geolocator.geocode(
            f"{address}, Newark, NJ, USA",
            timeout=10,
            country_codes="us",
        )
        if location:
            logger.info(
                "Geocoded Newark '%s' → (%f, %f) | %s",
                address,
                location.latitude,
                location.longitude,
                location.address,
            )
            if is_supported_location(location.latitude, location.longitude):
                return location

        # Nothing resolved to a point inside supported area
        return None

    except GeocoderTimedOut:
        raise HTTPException(
            status_code=503,
            detail=(
                "The geocoding service timed out. Please try again in a moment. "
                "If the problem persists, use the /predict endpoint with GPS coordinates instead."
            ),
        )
    except GeocoderServiceError:
        raise HTTPException(
            status_code=503,
            detail=(
                "The geocoding service is temporarily unavailable. "
                "Please try again later or use the /predict endpoint with GPS coordinates."
            ),
        )


def _address_error(label: str, address: str) -> HTTPException:
    """
    Build a rich 422 error for an address that could not be geocoded to NYC.
    Includes a fuzzy-match suggestion when possible.
    """
    hint = _closest_nyc_place(address)
    suggestion = f' Did you mean "{hint}, New York"?' if hint else ""
    return HTTPException(
        status_code=422,
        detail=(
            f"{label} address '{address}' could not be found within the supported area.{suggestion} "
            "Please enter a valid NYC address such as "
            '"Times Square, New York" or "JFK Airport, Queens, NY". '
            "For Newark Airport trips, try a specific address such as "
            '"Newark Liberty International Airport, Newark, NJ".'
        ),
    )


# ── Request / Response schemas ────────────────────────────────────────────────


class PredictRequest(BaseModel):
    """
    Predict by GPS coordinates.

    All four coordinate fields are validated against the supported area:
      - NYC longitudes: −74.25 to −73.70
      - NYC latitudes:   40.49 to  40.92
      - Newark Airport is also allowed as a special case

    `pickup_datetime` is optional; omit it to default to the current time.
    """

    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    pickup_datetime: str | None = None

    # ── Field validators ──────────────────────────────────────────────────

    @field_validator("pickup_longitude")
    @classmethod
    def validate_pickup_longitude(cls, v: float) -> float:
        if not (NYC_LON_MIN <= v <= NYC_LON_MAX or EWR_LON_MIN <= v <= EWR_LON_MAX):
            raise ValueError(
                f"pickup_longitude {v} is outside the supported area. "
                f"Supported NYC bounds are {NYC_LON_MIN} to {NYC_LON_MAX}. "
                "Newark Airport is also supported as a special case."
            )
        return v

    @field_validator("dropoff_longitude")
    @classmethod
    def validate_dropoff_longitude(cls, v: float) -> float:
        if not (NYC_LON_MIN <= v <= NYC_LON_MAX or EWR_LON_MIN <= v <= EWR_LON_MAX):
            raise ValueError(
                f"dropoff_longitude {v} is outside the supported area. "
                f"Supported NYC bounds are {NYC_LON_MIN} to {NYC_LON_MAX}. "
                "Newark Airport is also supported as a special case."
            )
        return v

    @field_validator("pickup_latitude")
    @classmethod
    def validate_pickup_latitude(cls, v: float) -> float:
        if not (NYC_LAT_MIN <= v <= NYC_LAT_MAX or EWR_LAT_MIN <= v <= EWR_LAT_MAX):
            raise ValueError(
                f"pickup_latitude {v} is outside the supported area. "
                f"Supported NYC bounds are {NYC_LAT_MIN} to {NYC_LAT_MAX}. "
                "Newark Airport is also supported as a special case."
            )
        return v

    @field_validator("dropoff_latitude")
    @classmethod
    def validate_dropoff_latitude(cls, v: float) -> float:
        if not (NYC_LAT_MIN <= v <= NYC_LAT_MAX or EWR_LAT_MIN <= v <= EWR_LAT_MAX):
            raise ValueError(
                f"dropoff_latitude {v} is outside the supported area. "
                f"Supported NYC bounds are {NYC_LAT_MIN} to {NYC_LAT_MAX}. "
                "Newark Airport is also supported as a special case."
            )
        return v

    @field_validator("pickup_datetime", mode="before")
    @classmethod
    def validate_pickup_datetime(cls, v):
        return parse_datetime(v)

    # ── Cross-field validator ─────────────────────────────────────────────

    @field_validator("pickup_datetime")
    @classmethod
    def validate_supported_coordinates(cls, v, info):
        data = info.data

        pickup_longitude = data.get("pickup_longitude")
        pickup_latitude = data.get("pickup_latitude")
        dropoff_longitude = data.get("dropoff_longitude")
        dropoff_latitude = data.get("dropoff_latitude")

        if pickup_longitude is not None and pickup_latitude is not None:
            if not is_supported_location(pickup_latitude, pickup_longitude):
                raise ValueError(
                    f"Pickup coordinates ({pickup_latitude}, {pickup_longitude}) are outside the supported area. "
                    f"Supported NYC bounds are lon {NYC_LON_MIN} to {NYC_LON_MAX}, "
                    f"lat {NYC_LAT_MIN} to {NYC_LAT_MAX}. "
                    "Newark Airport is also supported as a special case."
                )

        if dropoff_longitude is not None and dropoff_latitude is not None:
            if not is_supported_location(dropoff_latitude, dropoff_longitude):
                raise ValueError(
                    f"Dropoff coordinates ({dropoff_latitude}, {dropoff_longitude}) are outside the supported area. "
                    f"Supported NYC bounds are lon {NYC_LON_MIN} to {NYC_LON_MAX}, "
                    f"lat {NYC_LAT_MIN} to {NYC_LAT_MAX}. "
                    "Newark Airport is also supported as a special case."
                )

        return v


class PredictRequestByAddress(BaseModel):
    """
    Predict by street address.

    Addresses are geocoded via Nominatim and must resolve to a point inside NYC
    or Newark Airport.
    `pickup_datetime` is optional; omit it to default to the current time.
    """

    pickup_address: str
    dropoff_address: str
    pickup_datetime: str | None = None

    @field_validator("pickup_address", "dropoff_address")
    @classmethod
    def validate_address_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError(
                "Address must not be empty. "
                'Example: "Times Square, New York" or "JFK Airport, Queens, NY".'
            )
        if len(stripped) < 3:
            raise ValueError(
                f'Address "{v}" is too short to identify a location. '
                'Please provide a more specific address such as "Brooklyn Bridge, New York".'
            )
        return stripped

    @field_validator("pickup_datetime", mode="before")
    @classmethod
    def validate_pickup_datetime(cls, v):
        return parse_datetime(v)


class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    trip_duration_seconds: float
    trip_duration_minutes: float
    model_version: str
    latency_ms: float


# ── Internal prediction helper ────────────────────────────────────────────────


def _run_prediction(
    payload: PredictRequest,
    http_request: Request,
) -> PredictResponse:
    """
    Core prediction logic shared by /predict and /predict-by-address.
    Raises HTTPException on any failure.
    """
    import pandas as pd
    from src.features import clean, engineer, get_feature_columns

    with model_lock:
        model = model_state["model"]
        version = model_state["version"]

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "The prediction model is not loaded. "
                "The service may still be starting up — please retry in a few seconds."
            ),
        )

    try:
        df = pd.DataFrame(
            [
                {
                    "pickup_longitude": payload.pickup_longitude,
                    "pickup_latitude": payload.pickup_latitude,
                    "dropoff_longitude": payload.dropoff_longitude,
                    "dropoff_latitude": payload.dropoff_latitude,
                    "pickup_datetime": pd.to_datetime(payload.pickup_datetime),
                }
            ]
        )

        df = clean(df)

        if df.empty:
            raise HTTPException(
                status_code=422,
                detail=(
                    "The supplied coordinates were filtered out during data cleaning. "
                    "Ensure both pickup and dropoff points are within the supported area "
                    f"(NYC: lon {NYC_LON_MIN}→{NYC_LON_MAX}, lat {NYC_LAT_MIN}→{NYC_LAT_MAX}; "
                    "Newark Airport is also allowed as a special case)."
                ),
            )

        df = engineer(df)
        feature_cols = get_feature_columns()

        # ── Latency tracking ─────────────────────────────────────────────
        # Measure model inference time so performance can be monitored in
        # local testing, CI runs, and production deployments.
        start = time.time()
        log_prediction = float(model.predict(df[feature_cols])[0])
        latency = time.time() - start
        logger.info("Prediction latency: %.2f ms", latency * 1000)

        prediction = float(np.expm1(log_prediction))
        # Clamp to realistic taxi trip range: 1 min → 3 hours
        prediction = max(60.0, min(prediction, 10_800.0))

        return PredictResponse(
            trip_duration_seconds=round(prediction, 2),
            trip_duration_minutes=round(prediction / 60, 2),
            model_version=str(version),
            latency_ms=round(latency * 1000, 2),
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
            status_code=500,
            detail=(
                "An unexpected error occurred while generating the prediction. "
                f"Reference ID: {correlation_id}. Please try again or contact support."
            ),
        )


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict trip duration by GPS coordinates",
    responses={
        200: {
            "description": "Successful prediction",
            "content": {
                "application/json": {
                    "example": {
                        "trip_duration_seconds": 847.5,
                        "trip_duration_minutes": 14.12,
                        "model_version": "3",
                        "latency_ms": 3.42,
                    }
                }
            },
        },
        422: {
            "description": (
                "Validation error — coordinates out of supported bounds, "
                "wrong data types, or invalid datetime format"
            ),
            "content": {
                "application/json": {
                    "example": {
                        "error": "Validation failed",
                        "details": [
                            {
                                "field": "pickup_datetime",
                                "message": (
                                    "Pickup coordinates (0.0, 0.0) are outside the supported area. "
                                    "NYC longitudes are negative values around -74. "
                                    "Example: -73.9857 (Times Square)."
                                ),
                            }
                        ],
                        "hint": "Check the field values and try again.",
                    }
                }
            },
        },
        500: {
            "description": "Unexpected prediction error — see Reference ID in detail",
        },
        503: {
            "description": "Model not yet loaded — retry in a few seconds",
        },
    },
)
def predict(payload: PredictRequest, http_request: Request) -> PredictResponse:
    """
    Predict NYC taxi trip duration using GPS coordinates.

    ### Required fields
    | Field | Type | Valid range | Example |
    |-------|------|-------------|---------|
    | `pickup_longitude` | float | −74.25 → −73.70 | −73.9857 |
    | `pickup_latitude`  | float | 40.49 → 40.92   | 40.7580  |
    | `dropoff_longitude`| float | −74.25 → −73.70 | −73.7781 |
    | `dropoff_latitude` | float | 40.49 → 40.92   | 40.6413  |

    ### Optional fields
    | Field | Type | Format | Default |
    |-------|------|--------|---------|
    | `pickup_datetime` | string | `YYYY-MM-DD HH:MM:SS` | current time |

    ### Notes
    - Coordinates must be within the supported area.
    - Predicted duration is clamped to a realistic range (60 s – 10 800 s).
    - Newark Airport is allowed as a special supported case.
    """
    logger.info("Incoming request to /predict")
    return _run_prediction(payload, http_request)


@app.post(
    "/predict-by-address",
    response_model=PredictResponse,
    summary="Predict trip duration by street address",
    responses={
        200: {
            "description": "Successful prediction",
            "content": {
                "application/json": {
                    "example": {
                        "trip_duration_seconds": 1523.0,
                        "trip_duration_minutes": 25.38,
                        "model_version": "3",
                        "latency_ms": 4.42,
                    }
                }
            },
        },
        422: {
            "description": (
                "Address not found in supported area, address is outside supported bounds, "
                "address string too short, or invalid datetime format"
            ),
            "content": {
                "application/json": {
                    "examples": {
                        "address_not_found": {
                            "summary": "Address not found in supported area",
                            "value": {
                                "detail": (
                                    "Pickup address 'Flushng' could not be found within the supported area. "
                                    "Did you mean 'Flushing, New York'? "
                                    "Please enter a valid NYC address such as "
                                    '"Times Square, New York" or "JFK Airport, Queens, NY". '
                                    "For Newark Airport trips, try a specific address such as "
                                    '"Newark Liberty International Airport, Newark, NJ".'
                                )
                            },
                        },
                        "address_outside_nyc": {
                            "summary": "Address resolved but outside supported area",
                            "value": {
                                "detail": (
                                    "Dropoff address 'Los Angeles, CA' could not be found "
                                    "within the supported area. Please enter a valid NYC or Newark Airport address."
                                )
                            },
                        },
                        "address_too_short": {
                            "summary": "Address string too short",
                            "value": {
                                "error": "Validation failed",
                                "details": [
                                    {
                                        "field": "pickup_address",
                                        "message": (
                                            'Address "NY" is too short to identify a location. '
                                            "Please provide a more specific address."
                                        ),
                                    }
                                ],
                                "hint": "Check the field values and try again.",
                            },
                        },
                    }
                }
            },
        },
        500: {
            "description": "Unexpected prediction error — see Reference ID in detail",
        },
        503: {
            "description": "Model not loaded or geocoding service unavailable",
            "content": {
                "application/json": {
                    "example": {
                        "detail": (
                            "The geocoding service timed out. Please try again or "
                            "use the /predict endpoint with GPS coordinates instead."
                        )
                    }
                }
            },
        },
    },
)
def predict_by_address(
    request: PredictRequestByAddress,
    http_request: Request,
) -> PredictResponse:
    """
    Predict NYC taxi trip duration using street addresses.

    Addresses are automatically geocoded to GPS coordinates via Nominatim.
    Both addresses **must** resolve to a point inside the supported area.

    ### Required fields
    | Field | Type | Example |
    |-------|------|---------|
    | `pickup_address`  | string | `"Times Square, New York"` |
    | `dropoff_address` | string | `"JFK Airport, Queens, NY"` |

    ### Optional fields
    | Field | Type | Format | Default |
    |-------|------|--------|---------|
    | `pickup_datetime` | string | `YYYY-MM-DD HH:MM:SS` | current time |

    ### Common mistakes
    - **Misspelled neighbourhoods** — e.g. `"Flushng"` → try `"Flushing, Queens"`
    - **Non-supported city** — e.g. `"Los Angeles"` → not supported
    - **Ambiguous short name** — e.g. `"Brooklyn"` works; `"B"` does not
    - **Missing city context** — prefer `"Harlem, New York"` over just `"Harlem"`
    """
    # ── Geocode pickup ────────────────────────────────────────────────────
    pickup = geocode_address(request.pickup_address)
    if not pickup:
        raise _address_error("Pickup", request.pickup_address)

    # ── Geocode dropoff ───────────────────────────────────────────────────
    dropoff = geocode_address(request.dropoff_address)
    if not dropoff:
        raise _address_error("Dropoff", request.dropoff_address)

    # ── Safety-net: explicit bounds check on resolved coordinates ─────────
    # (geocode_address already validates, but guard against edge cases)
    if not is_supported_location(pickup.latitude, pickup.longitude):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Pickup address '{request.pickup_address}' resolved to coordinates "
                f"({pickup.latitude:.4f}, {pickup.longitude:.4f}) which are outside the supported area. "
                "Please enter an address within New York City or Newark Airport."
            ),
        )

    if not is_supported_location(dropoff.latitude, dropoff.longitude):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Dropoff address '{request.dropoff_address}' resolved to coordinates "
                f"({dropoff.latitude:.4f}, {dropoff.longitude:.4f}) which are outside the supported area. "
                "Please enter an address within New York City or Newark Airport."
            ),
        )

    # ── Build coordinate request and predict ──────────────────────────────
    predict_request = PredictRequest(
        pickup_longitude=pickup.longitude,
        pickup_latitude=pickup.latitude,
        dropoff_longitude=dropoff.longitude,
        dropoff_latitude=dropoff.latitude,
        pickup_datetime=request.pickup_datetime,
    )
    return _run_prediction(predict_request, http_request)


@app.get(
    "/health",
    summary="Liveness check",
    responses={
        200: {
            "description": "Service is running",
            "content": {"application/json": {"example": {"status": "ok"}}},
        }
    },
)
def health():
    """
    Liveness check endpoint.

    - Must respond in under **100 ms**.
    - Used by Cloud Run to verify the container is alive.
    - Does **not** verify that the model is loaded — use `/model-info` for that.
    """
    return {"status": "ok"}


@app.get(
    "/model-info",
    summary="Current model metadata",
    responses={
        200: {
            "description": "Model metadata",
            "content": {
                "application/json": {
                    "example": {
                        "model_name": "group-a1-model",
                        "version": "3",
                        "stage": "Production",
                        "trained_at": 1686520800000,
                        "primary_metric": "RMSE",
                        "primary_metric_value": 282.4,
                        "model_loaded": True,
                    }
                }
            },
        },
        503: {
            "description": "Model not yet loaded",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Model is not loaded. The service may still be starting up."
                    }
                }
            },
        },
    },
)
def model_info():
    """
    Returns metadata about the currently loaded Production model.

    Includes model version, training timestamp and evaluation metrics.
    If the model has not yet been loaded (e.g. still starting), returns 503.
    """
    with model_lock:
        if model_state["model"] is None:
            raise HTTPException(
                status_code=503,
                detail="Model is not loaded. The service may still be starting up.",
            )
        metrics = model_state["metrics"]
        return {
            "model_name": MODEL_REGISTRY_NAME,
            "version": model_state["version"],
            "stage": model_state["stage"],
            "trained_at": model_state["trained_at"],
            "primary_metric": "RMSE",
            "primary_metric_value": metrics.get("primary_metric"),
            "model_loaded": True,
        }


@app.get(
    "/coverage",
    summary="Geographic coverage area",
    responses={
        200: {
            "description": "Coverage details",
        }
    },
)
def coverage():
    """
    Returns the geographic coverage area supported by this predictor.

    Use this endpoint to verify that your trip origin and destination are within
    the supported region before calling `/predict` or `/predict-by-address`.
    """
    return {
        "supported_city": "New York City, USA",
        "boroughs": [
            "Manhattan",
            "Brooklyn",
            "Queens",
            "Bronx",
            "Staten Island",
        ],
        "nearby_areas_supported": [
            "Newark, New Jersey (airport trips only)",
        ],
        "airports_supported": [
            {
                "name": "John F. Kennedy International Airport (JFK)",
                "borough": "Queens, NYC",
                "coordinates": {"latitude": 40.6413, "longitude": -73.7781},
            },
            {
                "name": "LaGuardia Airport (LGA)",
                "borough": "Queens, NYC",
                "coordinates": {"latitude": 40.7769, "longitude": -73.8740},
            },
            {
                "name": "Newark Liberty International Airport (EWR)",
                "borough": "Newark, New Jersey",
                "note": "Supported as a special-case airport area",
                "coordinates": {"latitude": 40.6895, "longitude": -74.1745},
            },
        ],
        "bounding_box": {
            "longitude": {"min": NYC_LON_MIN, "max": NYC_LON_MAX},
            "latitude": {"min": NYC_LAT_MIN, "max": NYC_LAT_MAX},
        },
        "data_trained_on": "NYC Yellow Taxi trips — January to June 2016",
        "note": (
            "Trips inside the NYC bounding box are supported. Newark Airport (EWR) "
            "is also supported as a special-case airport location."
        ),
    }


@app.post(
    "/reload-model",
    summary="Hot-swap the Production model",
    responses={
        200: {
            "description": "Reload triggered",
            "content": {"application/json": {"example": {"status": "reloading"}}},
        }
    },
)
def reload_model():
    """
    Hot-swaps to the latest Production model without restarting the container.

    The reload happens asynchronously in a background thread.
    Called automatically by Cloud Scheduler after a retraining run.
    The current model continues serving requests during the swap.
    """

    def _reload() -> None:
        try:
            load_production_model()
            logger.info("Model hot-swap completed successfully.")
        except Exception as exc:
            logger.error("Model hot-swap failed: %s", exc)

    threading.Thread(target=_reload, daemon=True).start()
    return {"status": "reloading"}
