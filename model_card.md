# Model Card — Group A1 — Taxi Trip Duration Predictor

## Model Description
XGBoost Regressor trained to predict NYC taxi trip duration in seconds.
XGBoost was chosen for its strong performance on tabular data, built-in 
handling of mixed feature types, and fast training with parallel processing.
The model is trained on log1p(trip_duration) to handle the skewed target 
distribution, and predictions are converted back using expm1().

## Training Data
- **Source**: NYC Taxi Trip Duration dataset (Kaggle competition)
- **Size**: 1,458,644 raw rows → 1,446,766 after cleaning
- **Date range**: January 2016 — June 2016
- **Known quality issues**: See known_issues.md
- **Preprocessing applied**:
  - Removed trips < 60 seconds and > 10,800 seconds
  - Removed invalid passenger counts (0, 7, 8, 9)
  - Removed coordinates outside NYC bounding box
  - Dropped id, dropoff_datetime, store_and_fwd_flag columns

## Features
All feature logic lives in `src/features.py`.

**Distance features:**
- `distance_km` — Haversine distance between pickup and dropoff
- `distance_km_sq` — Squared distance (captures non-linear relationship)
- `manhattan_km` — True Manhattan distance in km
- `bearing` — Direction of travel in degrees
- `lat_diff`, `lon_diff` — Coordinate differences

**Coordinate features:**
- `pickup_latitude`, `pickup_longitude`
- `dropoff_latitude`, `dropoff_longitude`

**Time features:**
- `hour`, `day_of_week`, `month`
- `hour_sin`, `hour_cos` — Cyclical hour encoding
- `is_weekend`, `is_rush_hour`, `is_night`
- `is_early_morning`, `is_morning_rush`, `is_midday`
- `is_evening_rush`, `is_late_night`

**Interaction features:**
- `distance_hour` — distance × hour
- `distance_weekday` — distance × day of week
- `distance_rush` — distance × rush hour flag

**Location flags:**
- `is_jfk_pickup`, `is_jfk_dropoff`
- `is_lga_pickup`, `is_lga_dropoff`
- `is_newark_pickup`, `is_newark_dropoff`
- `is_manhattan_pickup`

**Direction flags:**
- `going_north`, `going_east`

**Other:**
- `passenger_count` — Number of passengers in the trip (1-6)
- `vendor_id` — **REQUIRED: must be 1 (Yellow Cab) or 2 (Green Cab/Boro Taxi)**
  - Captures vendor-specific differences in service area, pricing, and operational patterns
  - Different vendors have significantly different trip duration distributions
  - **Providing incorrect vendor_id can skew predictions by 10-15%**

## Performance

| Split      | RMSE   | RMSLE | Notes                        |
|------------|--------|-------|------------------------------|
| Validation | 282.47 | 0.305 | 70/15/15 split, random_state=42 |
| Test       |        |       | Not yet evaluated            |

## MLflow Run Reference
Run ID: `165820e3bff64296890e53887c1bacbc`
Model version: 6
Registry name: group-a1-model
Stage: Production

## Known Limitations
- **Vendor-specific predictions**: Model performance varies significantly by vendor.
  - Yellow Cab (vendor_id=1): Primarily Manhattan-based, shorter average trips
  - Green Cab (vendor_id=2): Boro Taxi, covers outer boroughs, longer average trips
  - **Providing wrong vendor_id will produce inaccurate predictions; always validate vendor_id in requests**
- Model is trained on 2016 data only — may not generalize to current 
  traffic patterns, new roads, or post-COVID behavior changes.
- Coordinates must be within NYC bounding box — trips starting or 
  ending outside NYC are not supported.
- Does not account for real-time traffic, weather, or special events.
- Airport trips (JFK, LGA, EWR) may have higher prediction error due 
  to variable wait times not captured in coordinates alone.
- Passenger count has very low variance (71% are solo trips) — 
  weak signal but retained for completeness.

## Retraining Policy
- **Trigger**: Manual retraining when new data is available or RMSE 
  degrades significantly on production traffic.
- **Promotion metric**: RMSE (lower is better).
- **Promotion rule**: New model is promoted to Production only if its 
  validation RMSE is strictly lower than the current Production model.
- **On ties**: Current Production model is retained (no promotion).
- **Hot-swap**: API supports `/reload-model` endpoint to load new 
  Production model without container restart.

## API Usage
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_longitude": -73.982155,
    "pickup_latitude": 40.767937,
    "dropoff_longitude": -73.964630,
    "dropoff_latitude": 40.765602,
    "passenger_count": 1,
    "vendor_id": 1,
    "pickup_datetime": "2016-06-12 00:43:35"
  }'
```

**Note**: `pickup_datetime` must be in exact format `"YYYY-MM-DD HH:MM:SS"` (strict validation). 
Invalid formats (e.g., ISO8601, "2016/06/12") will return HTTP 422.

Expected response:
```json
{
  "trip_duration_seconds": 671.49,
  "trip_duration_minutes": 11.19,
  "model_version": "6"
}
```

## API Request Validation & Error Handling

All 7 input fields are **validated before prediction** to catch errors early and return meaningful HTTP 422 responses:

| Field | Validation Rule | Error if Invalid |
|-------|-----------------|------------------|
| `pickup_longitude` | -74.25 to -73.70 | `"Longitude must be within NYC bounds (-74.25 to -73.70)"` |
| `pickup_latitude` | 40.49 to 40.92 | `"Latitude must be within NYC bounds (40.49 to 40.92)"` |
| `dropoff_longitude` | -74.25 to -73.70 | `"Longitude must be within NYC bounds (-74.25 to -73.70)"` |
| `dropoff_latitude` | 40.49 to 40.92 | `"Latitude must be within NYC bounds (40.49 to 40.92)"` |
| `passenger_count` | 1 to 6 | `"Passenger count must be between 1 and 6"` |
| `vendor_id` | 1 or 2 | `"vendor_id must be either 1 or 2"` |
| `pickup_datetime` | Exact format `YYYY-MM-DD HH:MM:SS` | `"pickup_datetime must be in \"YYYY-MM-DD HH:MM:SS\" format (received: ...)"` |

**HTTP Status Codes:**
- **200**: Prediction successful
- **422**: Validation error (invalid input — see error message for details)
- **500**: Internal server error (prediction processing failed)
- **503**: Model not loaded from MLflow registry

**Error Response Format:**
Invalid requests return HTTP 422 with Pydantic validation details. Each error includes:
- `loc`: The field path where error occurred (e.g., `["body", "vendor_id"]`)
- `msg`: Human-readable error message with specific reason
- `input`: The invalid value received

Example: Invalid vendor_id returns:
```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "vendor_id"],
      "msg": "Value error, vendor_id must be either 1 or 2",
      "input": 3
    }
  ]
}
```

## Competition Note
Group A1 achieved RMSE 282.47 seconds on the validation set using:
- XGBoost with 2000 estimators and log-transformed target
- 30+ engineered features including airport flags, cyclical time 
  encoding, and distance interaction features
- Automated model promotion via MLflow evaluation gate