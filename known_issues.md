# Known Issues — Group A1

Log data quality issues, surprises, and decisions made during EDA here.
Both ML and Cloud team members should contribute to this file.

 

## Data Quality Issues

- **Outlier trip durations**: The raw data contained trips as short as 1 second 
  and as long as 3,526,282 seconds (~40 days). We kept only trips between 
  60 seconds and 10,800 seconds (1 min to 3 hours).

- **Zero/invalid passenger counts**: 60 rows had 0 passengers and 5 rows had 
  7-9 passengers which is impossible for a standard taxi. These were removed.

- **Out-of-bounds coordinates**: Some pickup/dropoff coordinates fell outside 
  the NYC bounding box (-74.25 to -73.70 longitude, 40.49 to 40.92 latitude). 
  These were removed as they represent GPS errors.

- **No missing values**: All 11 columns had zero null values — no imputation needed.

- **Class imbalance in passenger count**: ~71% of trips had exactly 1 passenger, 
  making passenger_count a weak but still useful feature.

## Decisions Made

- **Target variable**: `trip_duration` already existed in the dataset in seconds. 
  We did not need to compute it from pickup/dropoff datetime difference.

- **Dropped columns**: Removed `id` (identifier), `store_and_fwd_flag` (irrelevant), 
  and `dropoff_datetime` to prevent data leakage since 
  dropoff_datetime - pickup_datetime = trip_duration exactly.

- **Log-transform target**: We trained on `log1p(trip_duration)` and converted 
  predictions back with `expm1()` to handle the skewed distribution of trip durations.

- **Feature engineering**: Added haversine distance, bearing, Manhattan distance in km, 
  airport flags (JFK, LaGuardia, Newark), time-based features (hour buckets, 
  cyclical encoding), interaction features (distance × hour, distance × rush hour), 
  and direction flags.

- **Model choice**: XGBoost Regressor chosen for its strong performance on 
  tabular data with mixed feature types.

- **Evaluation metric**: RMSE chosen as primary metric because it penalizes 
  large errors more heavily, which matters for trip planning.

- **Vendor-specific patterns**: The model includes `vendor_id` as a feature because:
  - **Yellow Cab (vendor_id=1)**: Operates primarily in Manhattan, shorter trips, higher density
  - **Green Cab (vendor_id=2)**: Boro Taxi licensed for outer boroughs, longer average trips
  - Vendor differences are significant — providing wrong vendor_id in inference will produce 
    inaccurate predictions. This is an **undocumented implicit input** if not clearly validated.

## Open Questions

- Would adding external weather data improve RMSE significantly?
- Would clustering pickup/dropoff zones (e.g. using k-means on coordinates) 
  add useful features?
- Early stopping never triggered at 2000 rounds — would 3000+ estimators 
  further improve RMSE?
- Should we use a test set evaluation before final model promotion?
