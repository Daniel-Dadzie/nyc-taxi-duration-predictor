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
    raise NotImplementedError("Implement clean() in features.py")


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering to a cleaned DataFrame.
    Returns a DataFrame with model-ready feature columns.
    TODO: implement for Project A
    """
    raise NotImplementedError("Implement engineer() in features.py")


def get_feature_columns() -> list:
    """
    Return the list of feature column names used by the model.
    Must match exactly what was used at training time.
    TODO: fill in after EDA
    """
    return []
