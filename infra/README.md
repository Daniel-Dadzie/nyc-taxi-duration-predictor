# Cloud Infrastructure — group-a1 — Taxi Trip Duration Predictor

This folder is owned by the **Cloud team** members in this group.
ML team members do not need to edit files in this folder.

---

## What Goes Here

| File / Folder | Purpose |
|---------------|---------|
| `mlflow_server/` | Config and deployment files for the MLflow tracking server |
| `cloud_run/` | Cloud Run service definitions for the prediction API |
| `scheduler/` | Cloud Scheduler job config for the retraining loop |
| `storage/` | Cloud storage bucket setup and IAM config |
| `monitoring/` | Dashboard and alerting config |
| `setup.sh` | One-command infrastructure setup script |

---

## Cloud Team Week-by-Week Checklist

### Week 1
- [ ] GCP project created, billing alert set at $30
- [ ] Required APIs enabled (Cloud Run, Cloud Storage, Firestore, Cloud Scheduler, Artifact Registry)
- [ ] MLflow server deployment planned — target: live by end of Week 2

### Week 2
- [ ] MLflow tracking server deployed on Cloud Run
- [ ] GCS bucket created as MLflow artifact backend
- [ ] MLflow server URL shared with ML team — they must confirm connection

### Week 3
- [ ] Data ingestion pipeline running — data flows into storage
- [ ] Request logger database schema designed

### Week 4
- [ ] ML team's Docker container deployed on Cloud Run
- [ ] Request logger logging all /predict calls
- [ ] Billing checked — within expected range

### Week 5
- [ ] Retraining scheduler configured and tested — triggers train.py
- [ ] Hot-swap tested — /reload-model works after promotion
- [ ] Monitoring dashboard showing request count, latency, model version

### Week 6
- [ ] All infrastructure clean and documented
- [ ] Cost report drafted

---

## Environment Variables Reference

The ML team's train.py and API need these — Cloud team provides the values:

| Variable | What it is | Where to get it |
|----------|------------|-----------------|
| `MLFLOW_TRACKING_URI` | URL of your MLflow Cloud Run service | Cloud Run → Services → copy URL |
| `DATA_PATH` | Path or URL to training data in GCS | Cloud Storage → bucket → copy gsutil URI |

Share these with the ML team as soon as they are available.
