# Group A1 — Taxi Trip Duration Predictor

**Organisation:** cloud-ml-internship-2026  
**Project:** Project A — Taxi Trip Duration Predictor  
**Competing against:** Group A2 (same project, head-to-head)  
**Primary metric:** RMSE  
**Dataset:** NYC Taxi (Kaggle)  

---

## Group Members

| Name | Track | Role |
|------|-------|------|
| TBC  | ML    | Project Manager |
| TBC  | ML    | Technical Deputy |
| TBC  | ML    | ML Member |
| TBC  | Cloud | Cloud Member |
| TBC  | Cloud | Cloud Member |

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

Copy the block below into a `.env` file or export them in your shell before running any commands:

```bash
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

This will load the data, engineer features, train the model, log the run to MLflow, and promote the model to Production if it beats the current baseline.

### 4. Run the prediction API locally

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8080
```

The API will be available at `http://localhost:8080`. Check the health endpoint:

```bash
curl http://localhost:8080/health
# {"status":"ok"}
```

### 5. Run via Docker

```bash
# Build the image
docker build -t group-a1-api .

# Run the container (pass required env vars)
docker run -p 8080:8080 \
  -e MLFLOW_TRACKING_URI="<MLflow server URL>" \
  -e MODEL_REGISTRY_NAME="group-a1-model" \
  group-a1-api
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

Copy the block below into a `.env` file or export them in your shell before running any commands:

```bash
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

This will load the data, engineer features, train the model, log the run to MLflow, and promote the model to Production if it beats the current baseline.

### 4. Run the prediction API locally

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8080
```

The API will be available at `http://localhost:8080`. Check the health endpoint:

```bash
curl http://localhost:8080/health
# {"status":"ok"}
```

### 5. Run via Docker

```bash
# Build the image
docker build -t group-a1-api .

# Run the container (pass required env vars)
docker run -p 8080:8080 \
  -e MLFLOW_TRACKING_URI="<MLflow server URL>" \
  -e MODEL_REGISTRY_NAME="group-a1-model" \
  group-a1-api
```

---

## Competition Rules

- Both Group A1 and Group A2 build the same system independently
- MLflow experiments are **separate** — you cannot see rival runs until after the Week 5 checkpoint
- After Week 5 the ML Technical Lead opens visibility — both teams can compare metrics
- **Head-to-head crown:** the group with the better **RMSE** on the test set wins the project crown
- **Overall winner:** highest combined rubric score + crown points across the programme

---

## Repo Structure

```
group-a1/
├── notebooks/          # EDA and exploration — never import from here in production code
├── src/
│   └── features.py     # ALL feature engineering logic — imported by train.py and the API
├── api/
│   └── main.py         # FastAPI prediction API
├── train.py            # Standalone training script — must run with a single command
├── Dockerfile          # Builds the prediction API container
├── requirements.txt    # All dependencies pinned
├── model_card.md       # Model documentation
├── known_issues.md     # Data quality issues and surprises from EDA
└── README.md           # This file
```

---

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
- `passenger_count`: must be between 1 and 6
- `vendor_id`: must be either 1 or 2
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
