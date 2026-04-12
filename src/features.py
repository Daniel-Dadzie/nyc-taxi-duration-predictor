"""
features.py — Group A1 — Taxi Trip Duration Predictor

ALL feature engineering logic lives here.
Both train.py and api/main.py import from this file.
Never duplicate feature logic — if it exists here, use it from here.
"""

import pandas as pd
import numpy as np
import os

# Window size for time-series projects (Project D)
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "60"))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps to a raw DataFrame.
    Returns a cleaned DataFrame ready for feature engineering.
    TODO: implement for Project A
    """

    # Remove impossible passenger counts
    df = df[(df["passenger_count"] >= 1) & (df["passenger_count"] <= 6)]

    # Only filter trip_duration if it exists (training only)
    if "trip_duration" in df.columns:
        df = df[(df["trip_duration"] >= 60) & (df["trip_duration"] <= 10800)]

    # Remove impossible coordinates (NYC bounding box)
    df = df[
        (df["pickup_longitude"].between(-74.25, -73.70)) &
        (df["pickup_latitude"].between(40.49, 40.92)) &
        (df["dropoff_longitude"].between(-74.25, -73.70)) &
        (df["dropoff_latitude"].between(40.49, 40.92))
    ]

    # Convert datetime columns
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

    # Only convert dropoff_datetime if it exists (training only)
    if "dropoff_datetime" in df.columns:
        df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])

    return df.reset_index(drop=True)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2

    return R * 2 * np.arcsin(np.sqrt(a))


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering to a cleaned DataFrame.
    Returns a DataFrame with model-ready feature columns.
    TODO: implement for Project A
    """

    # Distance between pickup and dropoff
    df["distance_km"] = haversine(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )

    # Direction to travel
    dlon = np.radians(df["dropoff_longitude"] - df["pickup_longitude"])
    lat1 = np.radians(df["pickup_latitude"])
    lat2 = np.radians(df["dropoff_latitude"])

    df["bearing"] = np.degrees(
        np.arctan2(
            np.sin(dlon) * np.cos(lat2),
            np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon),
        )
    )

    # Time features
    df["hour"] = df["pickup_datetime"].dt.hour
    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
    df["month"] = df["pickup_datetime"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

    # Coordinate differences
    df["lat_diff"] = df["dropoff_latitude"] - df["pickup_latitude"]
    df["lon_diff"] = df["dropoff_longitude"] - df["pickup_longitude"]

    # Is nighttime
    df["is_night"] = df["hour"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)

    # JFK Airport pickup/dropoff flag
    df["is_jfk_pickup"] = (
        (df["pickup_latitude"].between(40.63, 40.65))
        & (df["pickup_longitude"].between(-73.80, -73.77))
    ).astype(int)

    df["is_jfk_dropoff"] = (
        (df["dropoff_latitude"].between(40.63, 40.65))
        & (df["dropoff_longitude"].between(-73.80, -73.77))
    ).astype(int)

    # ── NEW FEATURES ──────────────────────────────────────
    df["going_north"] = (df["lat_diff"] > 0).astype(int)
    df["going_east"] = (df["lon_diff"] > 0).astype(int)
    df["is_lga_pickup"] = (
        (df["pickup_latitude"].between(40.76, 40.78))
        & (df["pickup_longitude"].between(-73.88, -73.86))
    ).astype(int)
    df["is_lga_dropoff"] = (
        (df["dropoff_latitude"].between(40.76, 40.78))
        & (df["dropoff_longitude"].between(-73.88, -73.86))
    ).astype(int)
    df["is_manhattan_pickup"] = (
        (df["pickup_latitude"].between(40.70, 40.83))
        & (df["pickup_longitude"].between(-74.02, -73.93))
    ).astype(int)
    df["is_early_morning"] = df["hour"].isin([5, 6, 7]).astype(int)
    df["is_morning_rush"] = df["hour"].isin([8, 9, 10]).astype(int)
    df["is_midday"] = df["hour"].isin([11, 12, 13, 14]).astype(int)
    df["is_evening_rush"] = df["hour"].isin([15, 16, 17, 18, 19]).astype(int)
    df["is_late_night"] = df["hour"].isin([20, 21, 22, 23]).astype(int)
    df["distance_km_sq"] = df["distance_km"] ** 2
    
     # Proper Manhattan distance in km
    df["manhattan_km"] = (
        haversine(df["pickup_latitude"], df["pickup_longitude"],
                  df["pickup_latitude"], df["dropoff_longitude"])
        +
        haversine(df["pickup_latitude"], df["dropoff_longitude"],
                  df["dropoff_latitude"], df["dropoff_longitude"])
    )

    # Interaction features
    df["distance_hour"]    = df["distance_km"] * df["hour"]
    df["distance_weekday"] = df["distance_km"] * df["day_of_week"]
    df["distance_rush"]    = df["distance_km"] * df["is_rush_hour"]

    # Cyclical hour encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Newark airport
    df["is_newark_pickup"] = (
        (df["pickup_latitude"].between(40.68, 40.71)) &
        (df["pickup_longitude"].between(-74.19, -74.16))
    ).astype(int)

    df["is_newark_dropoff"] = (
        (df["dropoff_latitude"].between(40.68, 40.71)) &
        (df["dropoff_longitude"].between(-74.19, -74.16))
    ).astype(int)

    return df


def get_feature_columns() -> list:
    """
    Return the list of feature column names used by the model.
    Must match exactly what was used at training time.
    TODO: fill in after EDA
    """
    
    return [
        # Distance features
        "distance_km", "distance_km_sq", "manhattan_km",
        "bearing", "lat_diff", "lon_diff",

        # Coordinates
        "pickup_latitude", "pickup_longitude",
        "dropoff_latitude", "dropoff_longitude",

        # Time features
        "hour", "day_of_week", "month",
        "hour_sin", "hour_cos",
        "is_weekend", "is_rush_hour", "is_night",
        "is_early_morning", "is_morning_rush",
        "is_midday", "is_evening_rush", "is_late_night",

        # Interaction features
        "distance_hour", "distance_weekday", "distance_rush",

        # Airport flags
        "is_jfk_pickup", "is_jfk_dropoff",
        "is_lga_pickup", "is_lga_dropoff",
        "is_newark_pickup", "is_newark_dropoff",
        "is_manhattan_pickup",

        # Direction
        "going_north", "going_east",

        # Other
        "passenger_count", "vendor_id"
    ]