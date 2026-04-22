"""
train.py — Group A1 — Taxi Trip Duration Predictor

Standalone training script. Must run with a single command:
    python train.py

All config comes from environment variables — no hardcoded paths or values.
"""

import os
import mlflow
import mlflow.xgboost
from mlflow import MlflowClient
from dotenv import load_dotenv

load_dotenv()

# ── Config from environment ───────────────────────────────────────
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
MLFLOW_EXPERIMENT   = os.environ.get("MLFLOW_EXPERIMENT_NAME", "project-a-group-a1")
MODEL_REGISTRY_NAME = os.environ.get("MODEL_REGISTRY_NAME", "group-a1-model")
DATA_PATH           = os.environ["DATA_PATH"]

PRODUCTION_ALIAS = "production"  # single source of truth for the alias name

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

# ── Imports ───────────────────────────────────────────────────────
import pandas as pd
import numpy as np
from src.features import clean, engineer, get_feature_columns

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def load_data(path: str) -> pd.DataFrame:
    """Load raw data from path or cloud storage URL."""
    return pd.read_csv(path)


def split(df: pd.DataFrame):
    """Return train, val, test splits."""
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)
    val_df, test_df   = train_test_split(temp_df, test_size=0.50, random_state=42)
    return train_df, val_df, test_df


def train_model(X_train, y_train, X_val, y_val):
    # Cast to float32 to halve memory usage (float64 is unnecessary for XGBoost)
    X_train = X_train.astype("float32")
    X_val   = X_val.astype("float32")

    model = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=9,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",  # memory-efficient histogram-based training
        max_bin=256,         # caps memory without impacting accuracy
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
    return model


def evaluate(model, X, y) -> dict:
    """Return a dict of metrics for the given split."""
    log_preds = model.predict(X)
    preds     = np.expm1(log_preds)
    y_actual  = np.expm1(y)
    rmse  = np.sqrt(mean_squared_error(y_actual, preds))
    rmsle = np.sqrt(mean_squared_error(y, np.log1p(np.clip(preds, 0, None))))
    return {"primary_metric": rmse, "rmse": rmse, "rmsle": rmsle}


def get_production_metric(client: MlflowClient, registry_name: str) -> float | None:
    """
    Return the primary_metric (RMSE) of the current production model, or None.
    Uses aliases instead of the deprecated stages API.
    """
    try:
        version = client.get_model_version_by_alias(registry_name, PRODUCTION_ALIAS)
        run     = client.get_run(version.run_id)
        return run.data.metrics.get("primary_metric")
    except Exception:
        # No alias set yet — first run
        return None


def get_latest_version(client: MlflowClient, registry_name: str) -> str:
    """Return the version number (str) of the most recently registered model version."""
    versions = client.search_model_versions(f"name='{registry_name}'")
    if not versions:
        raise RuntimeError(f"No versions found for model '{registry_name}'")
    return str(max(int(v.version) for v in versions))


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

    TARGET = "trip_duration"
    X_train, y_train = train_df[feature_cols], np.log1p(train_df[TARGET])
    X_val,   y_val   = val_df[feature_cols],   np.log1p(val_df[TARGET])

    # 4. Train
    model = train_model(X_train, y_train, X_val, y_val)

    # 5. Evaluate
    val_metrics = evaluate(model, X_val, y_val)
    print(f"Validation metrics: {val_metrics}")

    # 6. MLflow run
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "model_type":        type(model).__name__,
                "n_estimators":      model.n_estimators,
                "learning_rate":     model.learning_rate,
                "max_depth":         model.max_depth,
                "subsample":         model.subsample,
                "colsample_bytree":  model.colsample_bytree,
                "min_child_weight":  model.min_child_weight,
                "gamma":             model.gamma,
                "reg_alpha":         model.reg_alpha,
                "reg_lambda":        model.reg_lambda,
            }
        )

        for k, v in val_metrics.items():
            mlflow.log_metric(k, v)

        # Use mlflow.xgboost instead of mlflow.sklearn for native XGBoost flavour
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_REGISTRY_NAME,
        )

        run_id = run.info.run_id
        print(f"MLflow run ID: {run_id}")

    # 7. Evaluation gate — only promote if new RMSE strictly beats current production
    client       = MlflowClient()
    current_best = get_production_metric(client, MODEL_REGISTRY_NAME)
    new_metric   = val_metrics["primary_metric"]  # RMSE — lower is better

    if current_best is None:
        promote = True
        reason  = "No production model exists yet — promoting as first production model"
    else:
        promote = new_metric < current_best        # strict improvement required
        reason  = (
            f"New RMSE {new_metric:.4f} vs production RMSE {current_best:.4f} "
            f"({'better' if promote else 'not better'} — lower is better)"
        )

    print(f"Evaluation gate: {reason}")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("promoted", promote)

    if promote:
        latest_version = get_latest_version(client, MODEL_REGISTRY_NAME)
        client.set_registered_model_alias(
            name=MODEL_REGISTRY_NAME,
            alias=PRODUCTION_ALIAS,
            version=latest_version,
        )
        print(f"✓ Model v{latest_version} promoted to production (alias: '{PRODUCTION_ALIAS}')")
    else:
        print("✗ Model NOT promoted — current production model has better RMSE")


if __name__ == "__main__":
    main()