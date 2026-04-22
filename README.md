# Group A1 — Taxi Trip Duration Predictor

**Organisation:** cloud-ml-internship-2026  
**Project:** Project A — Taxi Trip Duration Predictor  
**Competing against:** Group A2 (same project, head-to-head)  
**Primary metric:** RMSE  
**Dataset:** NYC Taxi (Kaggle) — January to June 2016  

---

## Project Overview

This project predicts NYC taxi trip duration in seconds given pickup and dropoff coordinates and time of day. The prediction engine is powered by **XGBoost** trained on 1.4M+ taxi trips with 40 engineered features including distance metrics, time-based features, airport flags, interaction features, and advanced temporal/route complexity features.

**Key deliverables:**
- ✅ ML pipeline: data cleaning, feature engineering, model training, MLflow logging
- ✅ Prediction API: 6 endpoints with coordinate and address-based predictions
- ✅ Cloud infrastructure: ready for deployment on Google Cloud Run
- ✅ Model monitoring: real-time model info and geographic coverage endpoints
- ✅ Hot-swap capability: reload Production model without container restart

---

## Project Structure

```
group-a1/
├── train.py                    # Training pipeline (single command entry point)
├── src/
│   ├── features.py            # Feature engineering shared by train.py and API
│   └── __init__.py
├── api/
│   ├── main.py                # FastAPI prediction service (6 endpoints)
│   └── __init__.py
├── infra/                      # Cloud infrastructure configuration
│   ├── cloud_run/             # Cloud Run service definitions
│   ├── scheduler/             # Cloud Scheduler retraining job config
│   ├── monitoring/            # Dashboard and alerting
│   └── setup.sh               # Infrastructure setup script
├── notebooks/
│   └── taxi_duration.ipynb    # EDA and model exploration
├── mlruns/                     # MLflow local artifact store
├── requirements.txt            # Python dependencies (XGBoost, FastAPI, pandas, etc.)
├── Dockerfile                  # Container image for Cloud Run deployment
├── model_card.md              # Model documentation (features, training data, decisions)
├── known_issues.md            # Data quality issues and analysis notes
└── README.md                  # This file
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- `pip`
- Docker (optional — for container-based runs)
- An MLflow tracking server URL (provided by the Cloud team)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

Create a `.env` file in the project root or export them in your shell before running any commands:

```bash
# Required
export MLFLOW_TRACKING_URI="<MLflow server URL from Cloud team>"
export DATA_PATH="<path or GCS URL to training CSV>"

# Optional — defaults are shown
export MLFLOW_EXPERIMENT_NAME="project-a-group-a1"
export MODEL_REGISTRY_NAME="group-a1-model"
```

### 3. Train the model

```bash
python train.py
```

This will:
1. Load raw data from `DATA_PATH`
2. Clean the data (remove outliers, invalid coordinates, impossible passenger counts)
3. Engineer 35 features from coordinates, time, and location data
4. Train XGBoost on log-transformed trip duration
5. Log the run to MLflow (metrics, params, model)
6. Automatically promote to Production if it beats the current baseline RMSE

**Output:** Model registered in MLflow registry as `group-a1-model`

### 4. Run the prediction API locally

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8080
```

The API will be available at `http://localhost:8080`. Check the health endpoint:

```bash
curl http://localhost:8080/health
# {"status":"ok"}
```

---

## API Endpoints

The prediction API (FastAPI) exposes **6 endpoints**:

### 1. **POST /predict** — Predict by GPS coordinates

**Request:**
```json
{
  "pickup_longitude": -73.97,
  "pickup_latitude": 40.77,
  "dropoff_longitude": -73.95,
  "dropoff_latitude": 40.75,
  "pickup_datetime": "2016-06-12 08:30:00"  // Optional — defaults to now
}
```

**Response:**
```json
{
  "trip_duration_seconds": 623.45,
  "trip_duration_minutes": 10.39,
  "model_version": "3"
}
```

### 2. **POST /predict-by-address** — Predict by street address

Automatically geocodes addresses to GPS coordinates using Nominatim.

**Request:**
```json
{
  "pickup_address": "Times Square, New York",
  "dropoff_address": "JFK Airport, New York",
  "pickup_datetime": "2016-06-12 08:30:00"  // Optional
}
```

**Response:** Same as `/predict`

### 3. **GET /health** — Health check

Used by Cloud Run to verify container is running. Must respond in <100ms.

**Response:**
```json
{"status": "ok"}
```

### 4. **GET /model-info** — Current model metadata

Returns version, training time, and performance metrics of the Production model.

**Response:**
```json
{
  "model_name": "group-a1-model",
  "version": "3",
  "stage": "Production",
  "trained_at": 1718889234000,
  "primary_metric": "RMSE",
  "primary_metric_value": 456.78
}
```

### 5. **GET /coverage** — Geographic coverage

Returns supported areas, boroughs, airports, and bounding box.

**Response:**
```json
{
  "supported_city": "New York City, USA",
  "boroughs": ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"],
  "airports_supported": [
    {"name": "JFK", "borough": "Queens"},
    {"name": "LaGuardia (LGA)", "borough": "Queens"},
    {"name": "Newark (EWR)", "borough": "Newark, NJ"}
  ],
  "bounding_box": {
    "longitude": {"min": -74.25, "max": -73.70},
    "latitude": {"min": 40.49, "max": 40.92}
  }
}
```

### 6. **POST /reload-model** — Hot-swap Production model

Reloads the latest Production model without container restart. Called automatically by Cloud Scheduler after retraining.

**Response:**
```json
{"status": "reloading"}
```

---

## Feature Engineering

All **40 engineered features** are defined in `src/features.py` and shared between `train.py` and `api/main.py`:

**Distance metrics (6):**
- `distance_km` — Haversine distance (geodesic)
- `distance_km_sq` — Squared distance
- `manhattan_km` — Manhattan distance (grid-based)
- `bearing` — Direction of travel (degrees)
- `lat_diff`, `lon_diff` — Coordinate deltas

**Coordinate features (4):**
- `pickup_latitude`, `pickup_longitude`, `dropoff_latitude`, `dropoff_longitude`

**Time features (13):**
- `hour`, `day_of_week`, `month` — Raw time components
- `hour_sin`, `hour_cos` — Cyclical hour encoding
- `is_weekend`, `is_rush_hour`, `is_night` — Binary flags
- `is_early_morning`, `is_morning_rush`, `is_midday`, `is_evening_rush`, `is_late_night` — Hour buckets

**Airport & location flags (7):**
- `is_jfk_pickup`, `is_jfk_dropoff`
- `is_lga_pickup`, `is_lga_dropoff`
- `is_newark_pickup`, `is_newark_dropoff`
- `is_manhattan_pickup`

**Interaction features (3):**
- `distance_hour`, `distance_weekday`, `distance_rush`

**Directional features (2):**
- `going_north`, `going_east`

**Advanced temporal & route features (5) — Added v2:**
- `is_manhattan` — Both pickup OR dropoff in Manhattan (consolidates spatial signals for high-demand area)
- `route_complexity` — Manhattan distance / Euclidean distance ratio (captures route winding; higher = more grid-like)
- `is_shift_start` — Driver shift change indicator (hour == 6, captures demand spike at shift changes)
- `is_holiday` — US holiday flag for Jan-Jun 2016 (MLK Jr. Day, Presidents Day, Memorial Day)
- `minutes_into_day` — Finer temporal granularity (0-1440 instead of 0-23 hours, better captures intra-hour patterns)

See `model_card.md` for complete feature documentation with coordinates and analysis.

---

## Data Pipeline

### Cleaning (`src/features.py:clean()`)
- Filters passenger count to [1, 6] range (removes 0, 7-9 as impossible)
- Filters trip duration to [60, 10800] seconds (1 min to 3 hours)
- Removes out-of-bounds coordinates (outside NYC bounding box: [-74.25, -73.70] lon, [40.49, 40.92] lat)
- Converts datetime columns to proper datetime objects
- Returns cleaned DataFrame with same schema

### Training (`train.py`)
- Loads data from `DATA_PATH` (local file or GCS)
- Cleans data: filters outliers, invalid records, out-of-bounds coordinates
- Engineers 40 features from coordinates, time, location, route complexity, and temporal data
- Splits into train/val/test (70%/15%/15%, random_state=42)
- Trains XGBoost (2000 estimators, learning_rate=0.02, early stopping at 50 rounds)
- Evaluates on validation set → calculates RMSE and RMSLE
- **Evaluation gate**: New RMSE compared against current production model
  - If no production model exists → promotes as first model
  - If new RMSE < production RMSE → promotes to Production (using MLflow aliases)
  - Otherwise → keeps current production model (no demotion)
- Logs all metrics, hyperparameters, and model to MLflow

### Inference (`api/main.py`)
- Loads Production model from MLflow registry on startup
- Accepts predictions via `/predict` or `/predict-by-address`
- Applies same cleaning + feature engineering
- Returns trip duration in seconds and minutes
- Hot-swaps to new Production model on request

---

## Model Details

**Algorithm:** XGBoost Regressor (Gradient Boosted Trees)

**Hyperparameters:**
- `n_estimators`: 2000 trees with early stopping at 50 rounds
- `learning_rate`: 0.02 (conservative to prevent overfitting)
- `max_depth`: 9 (tree depth constraint)
- `subsample`: 0.8, `colsample_bytree`: 0.7 (reduce variance)
- `min_child_weight`: 5, `gamma`: 0.1 (regularization)
- `reg_alpha`: 0.1, `reg_lambda`: 1.0 (L1 & L2 regularization)
- `tree_method`: "hist" (memory-efficient histogram-based training)

**Target:** trip_duration in seconds, trained on `log1p(trip_duration)` to normalize skewed distribution

**Training data:** 1,446,766 samples (cleaned from 1,458,644 raw rows, NYC Yellow Taxi, Jan–Jun 2016)

**Features:** 40 engineered features across 8 categories (distance, coordinates, time, airports, interactions, directional, route complexity, temporal)

**Primary metric:** RMSE (Root Mean Squared Error in seconds)

**Validation performance:** RMSE 279.02s, RMSLE 0.3021 (improved from 282.47s via v2 feature expansion)

See `model_card.md` for complete model documentation including all features, training data details, and performance metrics.

---

## Known Issues & Decisions

See `known_issues.md` for:
- Data quality issues (outliers, invalid coordinates, passenger count imbalance)
- Feature engineering decisions and rationale
- Open questions for future improvement

---

## Cloud Deployment

The `infra/` folder contains all cloud infrastructure configuration:
- **cloud_run/**: Cloud Run service definitions for the prediction API
- **scheduler/**: Cloud Scheduler job to retrain the model weekly
- **monitoring/**: Dashboard and alerting configuration
- **setup.sh**: One-command infrastructure setup

See `infra/README.md` for Cloud team setup instructions.

---

## MLflow Integration

- **Tracking URI**: Points to Cloud Run MLflow server (set via `MLFLOW_TRACKING_URI`)
- **Experiment:** `project-a-group-a1`
- **Model Registry:** `group-a1-model` with stages: Staging, Production
- **Artifact Backend:** Google Cloud Storage bucket
- **Auto-promotion:** train.py promotes to Production if RMSE improves

To view runs locally:
```bash
mlflow ui
```
Then open `http://localhost:5000` and navigate to the experiment.

---

## Testing & Validation

### Manual API testing

```bash
# Start the API
uvicorn api.main:app --reload

# Test /predict endpoint
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_longitude": -73.97,
    "pickup_latitude": 40.77,
    "dropoff_longitude": -73.95,
    "dropoff_latitude": 40.75,
    "pickup_datetime": "2016-06-12 08:30:00"
  }'

# Test /health endpoint
curl http://localhost:8080/health

# View interactive API docs
# Open http://localhost:8080/docs
```

### Docker validation

```bash
docker build -t group-a1-api .
docker run -p 8080:8080 \
  -e MLFLOW_TRACKING_URI="http://mlflow-server:5000" \
  -e MODEL_REGISTRY_NAME="group-a1-model" \
  group-a1-api

# Test health endpoint
curl http://localhost:8080/health
```

---

## Dependencies

All dependencies pinned in `requirements.txt`:
- **fastapi==0.111.0** — Web framework
- **uvicorn==0.30.1** — ASGI server
- **mlflow==2.13.0** — Model tracking & registry
- **xgboost==2.0.3** — Gradient boosting
- **pandas==2.2.2** — Data manipulation
- **numpy==1.26.4** — Numerical computing
- **scikit-learn==1.5.0** — Feature scaling, model utilities
- **geopy==2.4.1** — Address geocoding
- **pydantic==2.7.1** — Request validation
- **python-dotenv==1.0.1** — Environment variables

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Model not loading on API startup | Check `MLFLOW_TRACKING_URI` is correct and Production model exists in registry |
| Prediction times out | Check geocoding service (Nominatim) availability; may be rate-limited |
| Out-of-bounds coordinates error | Coordinates must be within NYC: longitude [-74.25, -73.70], latitude [40.49, 40.92] |
| Docker container exits immediately | Check `MLFLOW_TRACKING_URI` is accessible from inside container |
| Training fails with "No data" | Verify `DATA_PATH` is correct and readable; check data format matches expectations |

---

## Next Steps

1. **Cloud team:** Deploy infrastructure using `infra/setup.sh`
2. **ML team:** Train initial model with `python train.py` after data ingestion is ready
3. **Cloud team:** Deploy API container to Cloud Run
4. **ML team:** Set up retraining scheduler to run weekly
5. **Both teams:** Configure monitoring and alerts

---

## Resources

- [Model Card](model_card.md) — Model documentation and decisions
- [Known Issues](known_issues.md) — Data quality and EDA notes
- [Infrastructure](infra/README.md) — Cloud deployment guide
- [FastAPI Docs](http://localhost:8080/docs) — Interactive API documentation (when running locally)


## Quick Start

### 1. Setup environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Set environment variables
```bash
# Windows
set MLFLOW_TRACKING_URI=https://mlflow-server-662636572351.us-central1.run.app
set MLFLOW_EXPERIMENT_NAME=project-a-group-a1
set MODEL_REGISTRY_NAME=group-a1-model
set DATA_PATH=gs://group-a1-mlflow-artifacts/data/train.csv

# Mac/Linux
export MLFLOW_TRACKING_URI=https://mlflow-server-662636572351.us-central1.run.app
export MLFLOW_EXPERIMENT_NAME=project-a-group-a1
export MODEL_REGISTRY_NAME=group-a1-model
export DATA_PATH=gs://group-a1-mlflow-artifacts/data/train.csv
```

> **MLflow server** is hosted on Google Cloud Run — no local server needed.  
> Artifacts are stored in GCS: `gs://group-a1-mlflow-artifacts/mlflow-artifacts`

### 3. Train the model
```bash
python train.py
```

### 4. Start the API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 5. Test the API
Open your browser and go to:
http://127.0.0.1:8000/docs


---

## API Contract

> **Status: DRAFT — must be finalised and signed by both ML and Cloud leads by end of Week 1**
---

### POST /predict
Predicts NYC taxi trip duration given pickup and dropoff details.

**Input:**
```json
{
  "pickup_longitude": -73.982155,
  "pickup_latitude": 40.767937,
  "dropoff_longitude": -73.964630,
  "dropoff_latitude": 40.765602,
  "passenger_count": 1,
  "vendor_id": 1,
  "pickup_datetime": "2016-06-12 00:43:35"
}
```

**Output:**
```json
{
  "trip_duration_seconds": 671.49,
  "trip_duration_minutes": 11.19,
  "model_version": "6"
}


```

**Validation rules:**
- `pickup_longitude` and `dropoff_longitude`: must be between -74.25 and -73.70
- `pickup_latitude` and `dropoff_latitude`: must be between 40.49 and 40.92
 
- `pickup_datetime`: format must be "YYYY-MM-DD HH:MM:SS"

**Error responses:**
- `422` — Invalid input (coordinates out of bounds, invalid passenger count, invalid vendor_id, or invalid pickup_datetime format)
- `500` — Internal prediction error (generic response)
- `503` — Model not loaded

---

### GET /health
Health check endpoint. Must respond in under 100ms.

**Output:**
```json
{ "status": "ok" }
```

---

### GET /model-info
Returns current Production model metadata.

**Output:**
```json
{
  "model_name": "group-a1-model",
  "version": "6",
  "stage": "Production",
  "trained_at": 1775438057855,
  "primary_metric": "RMSE",
  "primary_metric_value": 282.47
}
```

---

### POST /reload-model
Hot-swaps to the latest Production model without container restart.

**Output:**
```json
{ "status": "reloading" }

### GET /coverage
Returns geographic coverage area of the predictor.

**Output:**
```json
{
  "supported_city": "New York City, USA",
  "boroughs": ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"],
  "nearby_areas_supported": ["Newark, New Jersey (airport trips only)"],
  "airports_supported": [...],
  "bounding_box": {
    "longitude": {"min": -74.25, "max": -73.70},
    "latitude":  {"min": 40.49,  "max": 40.92}
  }
}
```

### POST /predict-by-address
Predicts trip duration using street addresses instead of coordinates.
Automatically converts addresses to GPS coordinates.

**Input:**
```json
{
  "pickup_address":  "Times Square, New York",
  "dropoff_address": "JFK Airport, New York",
  "pickup_datetime": "2016-06-12 00:43:35"
}
```

**Output:**
```json
{
  "trip_duration_seconds": 2145.67,
  "trip_duration_minutes": 35.76,
  "model_version": "6"
}
```

**Error responses:**

| Code | Reason |
|------|--------|
| 422 | Address not found or outside NYC |
| 503 | Geocoding service unavailable |
| 500 | Internal prediction error |
```

---

## Model Performance

| Version | RMSE (seconds) | RMSLE | Notes |
|---------|---------------|-------|-------|
| v1 | 297.82 | 0.344 | Baseline XGBoost |
| v3 | 288.81 | 0.332 | Better hyperparameters |
| v4 | 284.33 | 0.326 | More features added |
| v6 | 282.47 | 0.305 | Best — log target + 2000 estimators |

**Current Production model:** version 6 — RMSE 282.47 seconds (~4.7 minutes average error)

---

## Environment Variables

| Variable | Value | Set by |
|----------|-------|--------|
| `MLFLOW_TRACKING_URI` | `https://mlflow-server-662636572351.us-central1.run.app` | Cloud team |
| `MLFLOW_EXPERIMENT_NAME` | `project-a-group-a1` | Both |
| `MODEL_REGISTRY_NAME` | `group-a1-model` | Both |
| `DATA_PATH` | `gs://group-a1-mlflow-artifacts/data/train.csv` | Cloud team |

---

## MLflow Naming Convention

- Experiment: `project-a-group-a1`
- Registry: `group-a1-model`
- Visibility: **private until Week 5 checkpoint**

---

## Week 1 Checklist

- [x] All group members have repo access
- [x] API contract fields filled in and agreed
- [x] MLflow experiment name agreed: `project-a-group-a1`
- [x] MLflow server URL received from Cloud team
- [x] EDA notebook started
- [x] known_issues.md created
- [x] Model trained and promoted to Production
- [x] API deployed and tested locally
- [x] API contract implemented and validated
