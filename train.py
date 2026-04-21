"""
train.py — Group A1 — Taxi Trip Duration Predictor

Standalone training script. Must run with a single command:
    python train.py

All config comes from environment variables — no hardcoded paths or values.
"""

import os
import json
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature

# ── Config from environment ───────────────────────────────────────
MLFLOW_TRACKING_URI  = os.environ["MLFLOW_TRACKING_URI"]
MLFLOW_EXPERIMENT    = os.environ.get("MLFLOW_EXPERIMENT_NAME", "project-a-group-a1")
MODEL_REGISTRY_NAME  = os.environ.get("MODEL_REGISTRY_NAME", "group-a1-model")
DATA_PATH            = os.environ["DATA_PATH"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

# ── Imports ───────────────────────────────────────────────────────
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from src.features import clean, engineer, get_feature_columns


def load_data(path: str) -> pd.DataFrame:
    """
    Load raw NYC taxi data from local CSV or GCS path.

    Args:
        path: Local file path or GCS URI (gs://...)

    Returns:
        Raw DataFrame with 11 columns
    """
    return pd.read_csv(path)


def split(df: pd.DataFrame):
    """
    Split data into train, validation and test sets.
    Uses 70/15/15 split with fixed random seed for reproducibility.

    Args:
        df: Cleaned and engineered DataFrame

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42
    )
    return train_df, val_df, test_df


def train_model(X_train, y_train, X_val, y_val):
    """
    Train XGBoost model on log-transformed trip duration.

    Uses log1p transform on target to handle skewed distribution.
    Early stopping monitors validation RMSE to prevent overfitting.

    Args:
        X_train: Training features
        y_train: Log-transformed training target
        X_val:   Validation features
        y_val:   Log-transformed validation target

    Returns:
        Trained XGBRegressor model
    """
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
        n_jobs=-1
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    return model


def evaluate(model, X, y) -> dict:
    """
    Evaluate model performance on a given split.

    Converts log predictions back to seconds using expm1().
    Primary metric is RMSE in seconds.

    Args:
        model: Trained XGBRegressor
        X:     Feature matrix
        y:     Log-transformed true target values

    Returns:
        Dict with keys: primary_metric, rmse, rmsle
    """
    log_preds = model.predict(X)
    preds     = np.expm1(log_preds)
    y_actual  = np.expm1(y)
    rmse  = np.sqrt(mean_squared_error(y_actual, preds))
    rmsle = np.sqrt(mean_squared_error(
        y, np.log1p(np.clip(preds, 0, None))
    ))
    return {
        "primary_metric": rmse,
        "rmse":           rmse,
        "rmsle":          rmsle,
    }


def get_git_commit() -> str:
    """Get current git commit hash for tracking."""
    try:
        return os.popen("git rev-parse HEAD").read().strip()
    except Exception:
        return "unknown"


def get_production_metric(
    client: MlflowClient,
    registry_name: str
) -> float | None:
    """
    Return the primary metric of the current Production model, or None.

    Args:
        client:        MLflow client
        registry_name: Name of model in registry

    Returns:
        Primary metric value or None if no Production model exists
    """
    try:
        versions = client.get_latest_versions(
            registry_name, stages=["Production"]
        )
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
    TARGET       = "trip_duration"

    X_train = train_df[feature_cols]
    y_train = np.log1p(train_df[TARGET])
    X_val   = val_df[feature_cols]
    y_val   = np.log1p(val_df[TARGET])
    X_test  = test_df[feature_cols]
    y_test  = np.log1p(test_df[TARGET])

    # 4. Train
    model = train_model(X_train, y_train, X_val, y_val)

    # 5. Evaluate all splits
    train_metrics = evaluate(model, X_train, y_train)
    val_metrics   = evaluate(model, X_val,   y_val)
    test_metrics  = evaluate(model, X_test,  y_test)

    print(f"Train metrics:      {train_metrics}")
    print(f"Validation metrics: {val_metrics}")
    print(f"Test metrics:       {test_metrics}")

    # 6. Evaluation gate
    client       = MlflowClient()
    current_best = get_production_metric(client, MODEL_REGISTRY_NAME)
    new_metric   = val_metrics.get("primary_metric")
    lower_is_better = True

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

    # 7. MLflow run
    with mlflow.start_run() as run:

        # ── Tags ──────────────────────────────────────────────────
        mlflow.set_tags({
            "team":         "group-a1",
            "project":      "taxi-trip-duration",
            "dataset":      "nyc-taxi-2016",
            "git_commit":   get_git_commit(),
            "promoted":     str(promote),
        })

        # ── Description ───────────────────────────────────────────
        mlflow.set_tag(
            "mlflow.note.content",
            "XGBoost with log-transformed target. "
            "Features: haversine distance, airport flags, "
            "cyclical hour encoding, interaction features. "
            "Removed vendor_id and passenger_count (weak features)."
        )

        # ── Hyperparameters ───────────────────────────────────────
        mlflow.log_params({
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
            "early_stopping":    model.early_stopping_rounds,
            "log_transform":     True,
            "random_state":      42,
        })

        # ── Data info ─────────────────────────────────────────────
        mlflow.log_params({
            "train_size":    len(X_train),
            "val_size":      len(X_val),
            "test_size":     len(X_test),
            "n_features":    len(feature_cols),
            "data_path":     DATA_PATH,
            "target":        TARGET,
        })

        # ── Metrics ───────────────────────────────────────────────
        mlflow.log_metrics({
            "train_rmse":       train_metrics["rmse"],
            "train_rmsle":      train_metrics["rmsle"],
            "val_rmse":         val_metrics["rmse"],
            "val_rmsle":        val_metrics["rmsle"],
            "test_rmse":        test_metrics["rmse"],
            "test_rmsle":       test_metrics["rmsle"],
            "primary_metric":   val_metrics["primary_metric"],
        })

        # ── Feature list artifact ─────────────────────────────────
        with open("features_used.json", "w") as f:
            json.dump(feature_cols, f, indent=2)
        mlflow.log_artifact("features_used.json")

        # ── Model with signature ──────────────────────────────────
        signature = infer_signature(
            X_val,
            model.predict(X_val)
        )
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_REGISTRY_NAME,
            signature=signature,
            input_example=X_val.head(3)
        )

        run_id = run.info.run_id
        print(f"MLflow run ID: {run_id}")

    # 8. Promote if better
    if promote:
        latest = client.get_latest_versions(
            MODEL_REGISTRY_NAME, stages=["None"]
        )
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