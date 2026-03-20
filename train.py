"""
train.py — Group A1 — Taxi Trip Duration Predictor

Standalone training script. Must run with a single command:
    python train.py

All config comes from environment variables — no hardcoded paths or values.
"""

import os
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

# ── Config from environment ───────────────────────────────────────
MLFLOW_TRACKING_URI   = os.environ["MLFLOW_TRACKING_URI"]
MLFLOW_EXPERIMENT     = os.environ.get("MLFLOW_EXPERIMENT_NAME", "project-a-group-a1")
MODEL_REGISTRY_NAME   = os.environ.get("MODEL_REGISTRY_NAME", "group-a1-model")
DATA_PATH             = os.environ["DATA_PATH"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

# ── Imports ───────────────────────────────────────────────────────
import pandas as pd
import numpy as np
from src.features import clean, engineer, get_feature_columns

# TODO: import your model class
# from sklearn.ensemble import RandomForestRegressor


def load_data(path: str) -> pd.DataFrame:
    """Load raw data from path or cloud storage URL."""
    # TODO: implement
    raise NotImplementedError


def split(df: pd.DataFrame):
    """Return train, val, test splits with stratify where appropriate."""
    # TODO: implement — use 70/15/15
    raise NotImplementedError


def train_model(X_train, y_train):
    """Train and return the model."""
    # TODO: implement
    raise NotImplementedError


def evaluate(model, X, y) -> dict:
    """Return a dict of metrics for the given split."""
    # TODO: implement — primary metric: RMSE
    raise NotImplementedError


def get_production_metric(client: MlflowClient, registry_name: str) -> float | None:
    """Return the primary metric of the current Production model, or None."""
    try:
        versions = client.get_latest_versions(registry_name, stages=["Production"])
        if not versions:
            return None
        run = client.get_run(versions[0].run_id)
        return run.data.metrics.get("primary_metric")
    except Exception:
        return None


def main():
    print(f"Training group-a1 | experiment: {MLFLOW_EXPERIMENT}")

    # 1. Load data
    df = load_data(DATA_PATH)

    # 2. Clean and engineer features
    df = clean(df)
    df = engineer(df)

    # 3. Split
    train_df, val_df, test_df = split(df)
    feature_cols = get_feature_columns()
    # TODO: set target column
    TARGET = "target"
    X_train, y_train = train_df[feature_cols], train_df[TARGET]
    X_val,   y_val   = val_df[feature_cols],   val_df[TARGET]

    # 4. Train
    model = train_model(X_train, y_train)

    # 5. Evaluate
    val_metrics = evaluate(model, X_val, y_val)
    print(f"Validation metrics: {val_metrics}")

    # 6. MLflow run
    with mlflow.start_run() as run:
        # Log params
        mlflow.log_params({
            "model_type": type(model).__name__,
            # TODO: add hyperparameters
        })

        # Log metrics
        for k, v in val_metrics.items():
            mlflow.log_metric(k, v)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model",
                                 registered_model_name=MODEL_REGISTRY_NAME)

        run_id = run.info.run_id
        print(f"MLflow run ID: {run_id}")

    # 7. Evaluation gate
    client = MlflowClient()
    current_best = get_production_metric(client, MODEL_REGISTRY_NAME)
    new_metric   = val_metrics.get("primary_metric")

    # TODO: set promote_if_better=True for metrics where lower is better (e.g. RMSE, FPR)
    # and False where higher is better (e.g. F1)
    lower_is_better = False  # change to True for RMSE / FPR

    if current_best is None:
        promote = True
        reason  = "No Production model exists — registering as first Production model"
    elif lower_is_better:
        promote = new_metric < current_best
        reason  = f"New {new_metric:.4f} vs current {current_best:.4f} (lower is better)"
    else:
        promote = new_metric > current_best
        reason  = f"New {new_metric:.4f} vs current {current_best:.4f} (higher is better)"

    print(f"Evaluation gate: {reason}")

    if promote:
        latest = client.get_latest_versions(MODEL_REGISTRY_NAME, stages=["None"])
        if latest:
            client.transition_model_version_stage(
                name=MODEL_REGISTRY_NAME,
                version=latest[-1].version,
                stage="Production"
            )
        mlflow.log_param("promoted", True)
        print("✓ Model promoted to Production")
    else:
        mlflow.log_param("promoted", False)
        print("✗ Model NOT promoted — current Production model is better")


if __name__ == "__main__":
    main()
