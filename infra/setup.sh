#!/usr/bin/env bash
# =============================================================================
# setup.sh — Cloud Infrastructure Setup
# Run this script to provision the core infrastructure for your group.
# Prerequisites: gcloud CLI installed and authenticated (gcloud auth login)
# =============================================================================

set -euo pipefail

# ── CONFIG — fill these in before running ────────────────────
PROJECT_ID=""          # your GCP project ID
REGION="us-central1"   # change if needed
BUCKET_NAME=""         # e.g. group-a1-mlflow-artifacts
MLFLOW_IMAGE="ghcr.io/mlflow/mlflow:latest"

# ── STEP 1: Enable required APIs ─────────────────────────────
echo "Enabling APIs..."
gcloud services enable \
  run.googleapis.com \
  storage.googleapis.com \
  firestore.googleapis.com \
  cloudscheduler.googleapis.com \
  artifactregistry.googleapis.com \
  --project="${PROJECT_ID}"

# ── STEP 2: Create GCS bucket for MLflow artifacts ───────────
echo "Creating GCS bucket..."
gsutil mb -l "${REGION}" "gs://${BUCKET_NAME}" 2>/dev/null || echo "Bucket already exists"

# ── STEP 3: Deploy MLflow server on Cloud Run ─────────────────
echo "Deploying MLflow server..."
gcloud run deploy mlflow-server \
  --image="${MLFLOW_IMAGE}" \
  --platform=managed \
  --region="${REGION}" \
  --allow-unauthenticated \
  --set-env-vars="MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db,MLFLOW_DEFAULT_ARTIFACT_ROOT=gs://${BUCKET_NAME}/mlflow-artifacts" \
  --project="${PROJECT_ID}"

echo ""
echo "MLflow server deployed. Copy the URL above and share it with the ML team."
echo "Set it as MLFLOW_TRACKING_URI in their .env file."
