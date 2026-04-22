# Model Card — Group A1 — Taxi Trip Duration Predictor

## Model Description
**Algorithm:** XGBoost Regressor (Gradient Boosted Trees)

**Hyperparameters:**
- `n_estimators`: 2000 trees (early stopping at 50 rounds without improvement)
- `learning_rate`: 0.02 (conservative, prevents overfitting)
- `max_depth`: 9 (tree depth constraint)
- `subsample`: 0.8 (use 80% of samples per tree)
- `colsample_bytree`: 0.7 (use 70% of features per tree)
- `min_child_weight`: 5 (minimum sum of instance weight in child)
- `gamma`: 0.1 (minimum loss reduction for split)
- `reg_alpha`: 0.1 (L1 regularization)
- `reg_lambda`: 1.0 (L2 regularization)
- `tree_method`: "hist" (memory-efficient histogram-based training)
- `max_bin`: 256 (histogram bins for splitting)

**Target transformation:**
- Input: Raw trip duration in seconds (highly skewed, range 60-10,800 seconds)
- Training: `log1p(trip_duration)` to normalize distribution
- Inference: Convert predictions back with `expm1()` then clip to [60, 10,800]

**Model was chosen for:**
- Strong performance on tabular data with mixed feature types
- Built-in handling of non-linear relationships
- Fast training with parallel processing (n_jobs=-1)
- Robustness to outliers
- Memory efficiency (float32 casting, histogram binning)

## Training Data
- **Source**: NYC Taxi Trip Duration dataset (Kaggle competition)
- **Size**: 1,458,644 raw rows → 1,446,766 after cleaning (99.2% retained)
- **Date range**: January 2016 — June 2016 (6 months)
- **Known quality issues**: See `known_issues.md`

**Data split (from cleaned data):**
- Training: 70% → ~1,012,736 samples
- Validation: 15% → ~216,915 samples (used for early stopping)
- Test: 15% → ~216,915 samples

**Preprocessing applied:**
- Removed trips < 60 seconds or > 10,800 seconds (outliers)
- Removed invalid passenger counts (< 1 or > 6)
- Removed coordinates outside NYC bounding box (GPS errors)
- Dropped columns: `id`, `store_and_fwd_flag` (irrelevant), `dropoff_datetime` (prevents data leakage)
- Converted datetime columns to proper datetime objects
- Cast all features to float32 (halves memory usage without accuracy loss)

## Features (35 total)
All feature logic lives in `src/features.py` and is shared between training and inference.

**Distance metrics (6):**
- `distance_km` — Haversine distance between pickup and dropoff (geodesic)
- `distance_km_sq` — Squared distance (captures non-linear relationship)
- `manhattan_km` — True Manhattan distance in km (grid-based)
- `bearing` — Direction of travel in degrees (0-360)
- `lat_diff`, `lon_diff` — Coordinate differences (raw deltas)

**Coordinate features (4):**
- `pickup_latitude`, `pickup_longitude`
- `dropoff_latitude`, `dropoff_longitude`

**Time features (13):**
- `hour`, `day_of_week`, `month` — Raw time components
- `hour_sin`, `hour_cos` — Cyclical hour encoding (preserves circular nature)
- `is_weekend` — Binary flag for Saturday/Sunday
- `is_rush_hour` — Binary flag for hours [7,8,9,17,18,19]
- `is_night` — Binary flag for hours [22,23,0,1,2,3,4,5]
- `is_early_morning`, `is_morning_rush`, `is_midday`, `is_evening_rush`, `is_late_night` — Hour bucket flags

**Airport & Location flags (7):**
- `is_jfk_pickup`, `is_jfk_dropoff` — JFK International (40.63-40.65°N, -73.80--73.77°W)
- `is_lga_pickup`, `is_lga_dropoff` — LaGuardia (40.76-40.78°N, -73.88--73.86°W)
- `is_newark_pickup`, `is_newark_dropoff` — Newark EWR (40.68-40.71°N, -74.19--74.16°W)
- `is_manhattan_pickup` — Manhattan pickup area (40.70-40.83°N, -74.02--73.93°W)

**Interaction features (3):**
- `distance_hour` — distance_km × hour
- `distance_weekday` — distance_km × day_of_week
- `distance_rush` — distance_km × is_rush_hour

**Directional features (2):**
- `going_north` — Binary flag if lat_diff > 0
- `going_east` — Binary flag if lon_diff > 0

**Features removed (2):**
- `passenger_count` — Negligible correlation with duration (r=0.0139), only 1.02 min variation
- `vendor_id` — No meaningful signal

See `known_issues.md` for detailed analysis.

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

## Retraining & Promotion Policy

**Automated evaluation gate (in `train.py`):**
1. After training, model is always registered to MLflow
2. New RMSE is compared against current production model RMSE
3. Promotion rules:
   - **If no production model exists:** Promote as first production model
   - **If production model exists:** Promote ONLY if new RMSE < production RMSE (strict improvement required)
   - **On ties or worse performance:** Keep current production model (no demotion)

**Model versioning:**
- Uses MLflow **aliases** instead of deprecated stages
- Production model tagged with alias `"production"`
- Allows rollback by re-setting alias to previous version
- All versions remain queryable for comparison

**Retraining trigger:**
- Manual: `python train.py` (when new data available)
- Automated: Cloud Scheduler runs weekly (scheduled via Cloud Run)

**Model promotion metric:** 
- Primary: RMSE (lower is better) in seconds
- Secondary: RMSLE (Root Mean Squared Log Error) for reference

**Hot-swap capability:**
- API endpoint `/reload-model` reloads latest production model
- No container restart required
- Can be called manually or via Cloud Scheduler after retraining

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
- 35 engineered features including airport flags, cyclical time 
  encoding, and distance interaction features
- Automated model promotion via MLflow evaluation gate