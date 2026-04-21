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

**Note:** Passenger count and vendor ID were analyzed but removed. Passenger count showed 
negligible correlation (0.0139) with trip duration. See `known_issues.md` for analysis.

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
- **Temporal shift**: Model trained on 2016 data only — may not generalize to current 
  traffic patterns, new infrastructure, or behavior changes (e.g., post-COVID).
- **Geographic constraint**: Coordinates must be within NYC bounding box 
  (longitude: [-74.25, -73.70], latitude: [40.49, 40.92]). Trips outside NYC 
  are not supported.
- **Missing real-world factors**: Does not account for real-time traffic, weather, 
  special events, or construction.
- **Airport variability**: Airport trips (JFK, LGA, EWR) may have higher prediction 
  error due to variable wait times not captured by coordinates alone.
- **Data limitations**: 2016 dataset may contain systematic biases that persist 
  (imbalanced passenger counts, pre-filtered GPS errors).

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

### Predict by GPS Coordinates
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_longitude": -73.982155,
    "pickup_latitude": 40.767937,
    "dropoff_longitude": -73.964630,
    "dropoff_latitude": 40.765602,
    "pickup_datetime": "2016-06-12 00:43:35"
  }'
```

Response:
```json
{
  "trip_duration_seconds": 671.49,
  "trip_duration_minutes": 11.19,
  "model_version": "6"
}
```

### Predict by Street Address
```bash
curl -X POST http://localhost:8080/predict-by-address \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_address": "Times Square, New York",
    "dropoff_address": "JFK Airport, New York",
    "pickup_datetime": "2016-06-12 08:30:00"
  }'
```

Response: Same as above (addresses auto-geocoded to coordinates)

## Competition Note
Group A1 achieved RMSE 282.47 seconds on the validation set using:
- XGBoost with 2000 estimators and log-transformed target
- 30+ engineered features including airport flags, cyclical time 
  encoding, and distance interaction features
- Automated model promotion via MLflow evaluation gate