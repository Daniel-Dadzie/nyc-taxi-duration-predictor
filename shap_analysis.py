"""
SHAP Feature Importance Analysis

Loads the trained XGBoost model from MLflow and generates SHAP explanations
to identify which features are most important for trip duration predictions.
"""

import pandas as pd
import numpy as np
import mlflow
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from src.features import clean, engineer, get_feature_columns

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading training data...")
df = pd.read_csv("train.csv")

# Clean and engineer
df_clean = clean(df)
df_features = engineer(df_clean)

# ── IMPORTANT: Use ONLY the features the model was trained on ─────────────────
# The model was trained on 35 features BEFORE we added the 5 new ones.
# So we use the original 35 features for SHAP analysis.
original_features = [
    # Distance features
    "distance_km",
    "distance_km_sq",
    "manhattan_km",
    "bearing",
    "lat_diff",
    "lon_diff",
    # Coordinates
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    # Time features
    "hour",
    "day_of_week",
    "month",
    "hour_sin",
    "hour_cos",
    "is_weekend",
    "is_rush_hour",
    "is_night",
    "is_early_morning",
    "is_morning_rush",
    "is_midday",
    "is_evening_rush",
    "is_late_night",
    # Interaction features
    "distance_hour",
    "distance_weekday",
    "distance_rush",
    # Airport flags
    "is_jfk_pickup",
    "is_jfk_dropoff",
    "is_lga_pickup",
    "is_lga_dropoff",
    "is_newark_pickup",
    "is_newark_dropoff",
    "is_manhattan_pickup",
    # Direction
    "going_north",
    "going_east",
]

X = df_features[original_features]
y = np.log1p(df_features["trip_duration"])

print(f"Data shape: {X.shape}")
print(f"Feature columns: {len(original_features)}")
print(f"NOTE: Using original 35 features (model trained before v2 expansion)")
print(f"      Will add 5 new features in next retraining run\n")

# ── Load model from MLflow ────────────────────────────────────────────────────
print("\nLoading model from MLflow...")
mlflow.set_tracking_uri("http://localhost:5000")

# Get the latest production model
from mlflow import MlflowClient

client = MlflowClient()
versions = client.get_latest_versions("group-a1-model", stages=["Production"])

if not versions:
    print("ERROR: No Production model found. Train and promote a model first.")
    exit(1)

v = versions[0]
model = mlflow.sklearn.load_model(f"models:/group-a1-model/Production")
print(f"Loaded model version {v.version}")

# ── Generate SHAP values ──────────────────────────────────────────────────────
print("\nGenerating SHAP values (this may take a minute)...")
explainer = shap.TreeExplainer(model)

# Use a sample for faster computation
sample_size = min(5000, len(X))
X_sample = X.sample(n=sample_size, random_state=42)
shap_values = explainer.shap_values(X_sample)

print(f"SHAP values computed for {sample_size} samples")

# ── Feature Importance Analysis ───────────────────────────────────────────────
# Mean absolute SHAP values
mean_abs_shap = np.abs(shap_values).mean(axis=0)
feature_importance = pd.DataFrame(
    {"feature": original_features, "importance": mean_abs_shap}
).sort_values("importance", ascending=False)

print("\n" + "=" * 60)
print("TOP 15 MOST IMPORTANT FEATURES")
print("=" * 60)
for idx, row in feature_importance.head(15).iterrows():
    print(f"{row['feature']:30s} {row['importance']:8.4f}")

print("\n" + "=" * 60)
print("BOTTOM 10 LEAST IMPORTANT FEATURES")
print("=" * 60)
for idx, row in feature_importance.tail(10).iterrows():
    print(f"{row['feature']:30s} {row['importance']:8.4f}")

# ── Generate plots ────────────────────────────────────────────────────────────
print("\nGenerating plots...")

# 1. Summary plot (bar chart)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_importance_bar.png", dpi=300, bbox_inches="tight")
print("✓ Saved: shap_importance_bar.png")

# 2. Summary plot (beeswarm - shows individual impacts)
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig("shap_importance_beeswarm.png", dpi=300, bbox_inches="tight")
print("✓ Saved: shap_importance_beeswarm.png")

# 3. Top 10 feature importance chart (custom)
fig, ax = plt.subplots(figsize=(12, 6))
top_features = feature_importance.head(10)
ax.barh(range(len(top_features)), top_features["importance"].values)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features["feature"].values)
ax.invert_yaxis()
ax.set_xlabel("Mean |SHAP value|", fontsize=12)
ax.set_title(
    "Top 10 Most Important Features for Trip Duration Prediction",
    fontsize=14,
    fontweight="bold",
)
for i, v in enumerate(top_features["importance"].values):
    ax.text(v + 0.01, i, f"{v:.4f}", va="center", fontsize=10)
plt.tight_layout()
plt.savefig("shap_top10_features.png", dpi=300, bbox_inches="tight")
print("✓ Saved: shap_top10_features.png")

# 4. Force plot for a single prediction (first sample)
print("\nGenerating force plot for first sample...")
shap.force_plot(
    explainer.expected_value, shap_values[0:1], X_sample.iloc[0:1], matplotlib=True
)
plt.savefig("shap_force_plot.png", dpi=300, bbox_inches="tight")
print("✓ Saved: shap_force_plot.png")

# ── Recommendations ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)

# Find features with very low importance
low_importance_threshold = np.percentile(feature_importance["importance"], 10)
low_importance_features = feature_importance[
    feature_importance["importance"] < low_importance_threshold
]["feature"].tolist()

print(f"\n1. CONSIDER REMOVING (< 10th percentile):")
for feat in low_importance_features:
    print(f"   - {feat}")

print(f"\n2. FOCUS ON TOP 5 FEATURES:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"   - {row['feature']:30s} (importance: {row['importance']:.4f})")

print(f"\n3. NEXT STEPS:")
print(f"   - Retrain without low-importance features")
print(f"   - Add new features related to top features (if possible)")
print(f"   - Consider interaction features between top features")

print("\n✓ SHAP analysis complete!")
print("  Plots saved to: shap_*.png")
