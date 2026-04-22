import pandas as pd
import numpy as np
from types import SimpleNamespace

from api.main import PredictRequest, _run_prediction, load_production_model

# ── Load model ────────────────────────────────────────────────────────────────
# When running outside FastAPI, lifespan startup does not run automatically.
# Load the Production model manually before calling _run_prediction.
load_production_model()

# ── Load unseen test data ────────────────────────────────────────────────────
df = pd.read_csv("test_split.csv")

# Use a small sample first so validation runs quickly
sample_df = df.sample(200, random_state=42)

actuals = []
predictions = []
errors = []

# Minimal request-like object for _run_prediction
dummy_request = SimpleNamespace(headers={})

for i, row in sample_df.iterrows():
    try:
        payload = PredictRequest(
            pickup_longitude=row["pickup_longitude"],
            pickup_latitude=row["pickup_latitude"],
            dropoff_longitude=row["dropoff_longitude"],
            dropoff_latitude=row["dropoff_latitude"],
            pickup_datetime=str(row["pickup_datetime"]),
        )

        result = _run_prediction(payload, dummy_request)

        predicted = result.trip_duration_seconds
        actual = float(row["trip_duration"])
        error = abs(predicted - actual)

        predictions.append(predicted)
        actuals.append(actual)
        errors.append(error)

        print(
            f"Row {i} | Predicted: {predicted:.2f} sec | "
            f"Actual: {actual:.2f} sec | Error: {error:.2f} sec"
        )

    except Exception as e:
        print(f"Skipping row {i}: {e}")

# ── Summary ──────────────────────────────────────────────────────────────────
if errors:
    mae_seconds = np.mean(errors)
    mae_minutes = mae_seconds / 60

    print("\nValidation summary")
    print(f"Rows evaluated: {len(errors)}")
    print(f"Average absolute error: {mae_seconds:.2f} seconds")
    print(f"Average absolute error: {mae_minutes:.2f} minutes")
else:
    print("No rows were successfully evaluated.")