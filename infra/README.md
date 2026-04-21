# Cloud Infrastructure — group-a1 — Taxi Trip Duration Predictor

This folder is owned by the **Cloud team** members in this group.
**ML team members:** You do not need to edit files in this folder, but you should understand the structure.

---

## Overview

This folder contains all infrastructure-as-code (IaC) and deployment configurations for running the Group A1 taxi trip duration predictor on Google Cloud Platform. The infrastructure includes:

- **MLflow Tracking Server** — Central hub for model tracking, registry, and artifact storage
- **Prediction API** — Deployed on Cloud Run, serves `/predict` and `/predict-by-address` endpoints
- **Automated Retraining** — Cloud Scheduler triggers weekly retraining jobs
- **Data Storage** — Cloud Storage buckets for training data and MLflow artifacts
- **Monitoring** — Cloud Monitoring dashboards and alerting

---

## Folder Structure

| File / Folder | Purpose | Owner |
|---------------|---------|-------|
| `cloud_run/service.yaml` | Cloud Run service definition for prediction API | Cloud |
| `scheduler/retrain_job.yaml` | Cloud Scheduler job config to trigger weekly retraining | Cloud |
| `monitoring/dashboard.md` | Cloud Monitoring dashboard setup and metrics | Cloud |
| `setup.sh` | One-command infrastructure setup script | Cloud |
| `README.md` | This file — infrastructure documentation | Cloud |

---

## Cloud Team Setup Checklist

Follow these steps **in order** to bring infrastructure online:

### Week 1: Project & Permissions

- [ ] GCP project created and linked to billing account
- [ ] Set billing alert at $30 (prevent runaway costs)
- [ ] Team members have Owner or Editor roles
- [ ] Required APIs enabled:
  - [ ] Cloud Run
  - [ ] Cloud Storage
  - [ ] Cloud Scheduler
  - [ ] Artifact Registry
  - [ ] Cloud Build (optional — for CI/CD)
  - [ ] Cloud Monitoring (optional — for dashboards)

### Week 2: MLflow Tracking Server

- [ ] MLflow server deployment planned (target: live end of Week 2)
- [ ] GCS bucket created for MLflow artifacts
- [ ] Cloud Run service deployed with:
  - [ ] MLflow container image (public or from Artifact Registry)
  - [ ] Environment variables: `BACKEND_STORE_URI` (Firestore), `ARTIFACT_STORE_URI` (GCS)
  - [ ] Firestore database created for backend store
- [ ] MLflow server URL shared with ML team
- [ ] **ML team confirms successful connection** — they must test authentication
- [ ] Model registry initialized with `group-a1-model`

### Week 3: Data & ML Pipeline

- [ ] Training data uploaded to GCS bucket
- [ ] ML team tests `python train.py` with `DATA_PATH` pointing to GCS URI
- [ ] First model trained and promoted to Production
- [ ] Request logger database schema designed (Cloud Firestore or Cloud SQL)

### Week 4: API Deployment

- [ ] Dockerfile built and pushed to Artifact Registry (or Docker Hub)
- [ ] Cloud Run service deployed for prediction API with:
  - [ ] Environment variables: `MLFLOW_TRACKING_URI`, `MODEL_REGISTRY_NAME`
  - [ ] Memory: 1–2 GB (adjust based on model size)
  - [ ] CPU: 1–2 cores (adjust based on load testing)
  - [ ] Concurrency: 80 (default; adjust after load testing)
- [ ] API tested at public URL (all 6 endpoints)
- [ ] Request logging enabled (logs all `/predict` calls)

### Week 5: Retraining & Hot-Swap

- [ ] Cloud Scheduler job created to run weekly:
  ```bash
  gcloud scheduler jobs create app-engine retrain-job \
    --schedule="0 2 * * 0" \
    --http-method=POST \
    --uri="<Cloud Run URL>/api/retrain" \
    --oidc-service-account-email="<service-account>"
  ```
- [ ] Retraining script (wrapper around `python train.py`) deployed
- [ ] Hot-swap tested:
  - Train new model
  - Verify it's promoted to Production
  - Call `/reload-model` on prediction API
  - Verify `/model-info` reflects new version
- [ ] Monitoring dashboard showing:
  - Request count (per hour)
  - API latency (p50, p95, p99)
  - Model version currently loaded
  - Error rates by endpoint

### Week 6: Monitoring & Cleanup

- [ ] Error alerts configured (alert on 5+ errors in 1 minute)
- [ ] Latency alerts configured (alert if p95 > 2 seconds)
- [ ] Cost report generated and reviewed
- [ ] Infrastructure documentation finalized
- [ ] Team handoff complete

---

## Environment Variables Reference

The following variables must be set on Cloud Run for the prediction API:

| Variable | Value | Where to get it |
|----------|-------|-----------------|
| `MLFLOW_TRACKING_URI` | URL of MLflow Cloud Run service | Cloud Run → Services → Click MLflow service → Copy URL |
| `MODEL_REGISTRY_NAME` | Model registry name | Default: `group-a1-model` — confirm with ML team |

### Example
```bash
gcloud run deploy group-a1-api \
  --image=<ARTIFACT_REGISTRY>/group-a1-api:latest \
  --set-env-vars=MLFLOW_TRACKING_URI="https://mlflow-xyz.run.app",MODEL_REGISTRY_NAME="group-a1-model" \
  --region=us-central1 \
  --memory=2Gi \
  --cpu=2
```

---

## MLflow Server Deployment

### Container Image Options

1. **Official MLflow image** (recommended for quick start):
   ```bash
   docker pull python:3.10
   # Add Dockerfile with: RUN pip install mlflow
   ```

2. **Custom image** (if you need specific backends):
   ```dockerfile
   FROM python:3.10
   RUN pip install mlflow google-cloud-storage
   CMD ["mlflow", "server", \
        "--backend-store-uri", "firestore:///my-project", \
        "--default-artifact-root", "gs://my-bucket/mlflow-artifacts"]
   ```

### Firestore Backend Store

MLflow requires a backend to store run metadata. Use Cloud Firestore:

```bash
# Create Firestore database (if not exists)
gcloud firestore databases create --region=us-central1

# Deploy MLflow with Firestore backend
gcloud run deploy mlflow-server \
  --image=<YOUR_IMAGE> \
  --set-env-vars=BACKEND_STORE_URI="firestore:///my-project" \
  --region=us-central1 \
  --memory=2Gi \
  --allow-unauthenticated
```

### GCS Artifact Backend

All model artifacts and training logs are stored in GCS:

```bash
# Create bucket
gsutil mb gs://group-a1-mlflow-artifacts

# Grant Cloud Run service account access
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:<SERVICE_ACCOUNT>@iam.gserviceaccount.com" \
  --role="roles/storage.admin"
```

---

## Prediction API Deployment

### Docker Image Build

```bash
# Build image (from project root)
docker build -t group-a1-api:latest .

# Push to Artifact Registry
docker tag group-a1-api:latest us-central1-docker.pkg.dev/PROJECT_ID/group-a1/api:latest
docker push us-central1-docker.pkg.dev/PROJECT_ID/group-a1/api:latest
```

### Cloud Run Deployment

```bash
gcloud run deploy group-a1-api \
  --image=us-central1-docker.pkg.dev/PROJECT_ID/group-a1/api:latest \
  --region=us-central1 \
  --memory=2Gi \
  --cpu=2 \
  --allow-unauthenticated \
  --set-env-vars=MLFLOW_TRACKING_URI="https://mlflow-xyz.run.app",MODEL_REGISTRY_NAME="group-a1-model" \
  --update-env-vars=PORT=8080
```

### Load Testing (after deployment)

```bash
# Install load testing tool
pip install locust

# Run load test
locust -f load_test.py --host https://group-a1-api.run.app
```

---

## Cloud Scheduler Retraining Job

### Job Configuration

```yaml
# retrain_job.yaml
name: projects/PROJECT_ID/locations/us-central1/jobs/retrain-taxi-model
description: Weekly retraining of taxi duration model
schedule: "0 2 * * 0"  # Every Sunday at 2 AM UTC
timeZone: UTC
httpTarget:
  uri: https://group-a1-api.run.app/retrain
  httpMethod: POST
  headers:
    Authorization: Bearer <SERVICE_ACCOUNT_TOKEN>
    X-Idempotency-Key: weekly-retrain
retryConfig:
  retryCount: 1
```

### Deploy with gcloud

```bash
gcloud scheduler jobs create app-engine retrain-taxi-model \
  --schedule="0 2 * * 0" \
  --http-method=POST \
  --uri="https://group-a1-api.run.app/retrain" \
  --oidc-service-account-email="cloud-team@PROJECT_ID.iam.gserviceaccount.com" \
  --location=us-central1
```

### Retraining Endpoint (on API)

The prediction API should expose a `/retrain` endpoint that:
1. Triggers `python train.py` with correct environment variables
2. Waits for training to complete
3. Verifies new model was promoted to Production
4. Returns success/failure status

---

## Monitoring & Alerting

### Cloud Monitoring Dashboard

Create a dashboard with these metrics:

**API Metrics:**
- Request count per endpoint (per hour)
- Latency (p50, p95, p99)
- Error rate (5xx responses)
- Model version currently loaded

**Model Metrics:**
- Training frequency (retrains per week)
- Model promotion frequency
- RMSE of current Production model

**Cost Metrics:**
- Cloud Run invocations (per day)
- GCS data transfer (per day)
- Total spend (per day)

### Example Alert Policies

```bash
# Alert: More than 10 errors in 5 minutes
gcloud alpha monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --display-name="API Errors High" \
  --condition-display-name="Error rate > 5%" \
  --condition-threshold-value=0.05

# Alert: Latency too high
gcloud alpha monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --display-name="API Latency High" \
  --condition-display-name="p95 latency > 2s" \
  --condition-threshold-value=2
```

---

## Cost Optimization

### Expected Monthly Costs

| Service | Usage | Cost |
|---------|-------|------|
| Cloud Run (API) | 500K requests/month @ 0.1s avg | ~$2–5 |
| Cloud Run (MLflow) | 24/7 always-on | ~$10–15 |
| Cloud Storage | 100 GB artifacts, 1 GB/day transfer | ~$2–3 |
| Cloud Scheduler | 4 retraining runs/month | <$0.01 |
| Cloud Monitoring | Logs and dashboards | ~$1–2 |
| **Total** | | **~$15–25/month** |

### Cost Reduction Tips

- **Cloud Run:** Use minimum CPU (0.5) when not training
- **MLflow:** Use smaller instance for low-traffic environments
- **GCS:** Set lifecycle rules to delete old artifacts after 90 days
- **Monitoring:** Only alert on critical metrics; archive logs after 30 days

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| API fails to start | `MLFLOW_TRACKING_URI` unreachable | Check MLflow server is running; verify URL is public |
| Model not loading | Production model doesn't exist | Train and promote first model with `python train.py` |
| High API latency | Model too large or network lag | Optimize model size; use Cloud Run CPU scaling |
| Retraining fails | Data path incorrect or insufficient permissions | Check `DATA_PATH` env var; verify service account has GCS read access |
| Cost spike | Unexpectedly high traffic or logs | Check Cloud Run metrics; enable autoscaling limits |

---

## Useful Commands

```bash
# View Cloud Run services
gcloud run services list

# Stream logs from prediction API
gcloud run logs read group-a1-api --limit=100 --follow

# Deploy new API version
gcloud run deploy group-a1-api --image=<NEW_IMAGE> --region=us-central1

# Check Cloud Scheduler job status
gcloud scheduler jobs describe retrain-taxi-model --location=us-central1

# Trigger retraining manually
gcloud scheduler jobs run retrain-taxi-model --location=us-central1

# View billing
gcloud billing accounts list
gcloud billing budgets list --billing-account=ACCOUNT_ID
```

---

## Communication with ML Team

### What ML Team Needs from Cloud Team

By **end of Week 2:**
- [ ] MLflow tracking server URL and credentials
- [ ] `DATA_PATH` (GCS URI or local path to training data)
- [ ] Confirmation that they can connect to MLflow

By **end of Week 4:**
- [ ] Prediction API deployed and publicly accessible
- [ ] Confirmation that `/health` and `/model-info` endpoints work

### What Cloud Team Needs from ML Team

By **end of Week 3:**
- [ ] Confirmation that first model trained successfully
- [ ] Model registry name (`group-a1-model` by default)
- [ ] Expected model size (for Cloud Run memory allocation)

By **end of Week 5:**
- [ ] Retraining script ready to be scheduled
- [ ] Confirmation that `/reload-model` hot-swap works

---

## Resources

- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [MLflow Documentation](https://mlflow.org/docs)
- [Cloud Scheduler Documentation](https://cloud.google.com/scheduler/docs)
- [Cloud Monitoring Documentation](https://cloud.google.com/monitoring/docs)
- [Artifact Registry Documentation](https://cloud.google.com/artifact-registry/docs)

---

## Questions?

Contact your Cloud Technical Lead or refer to the main [README.md](../README.md) for project overview.

