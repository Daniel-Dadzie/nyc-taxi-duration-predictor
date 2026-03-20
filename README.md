# Group A1 — Taxi Trip Duration Predictor

**Organisation:** cloud-ml-internship-2026  
**Project:** Project A — Taxi Trip Duration Predictor  
**Competing against:** Group A2 (same project, head-to-head)  
**Primary metric:** RMSE  
**Dataset:** NYC Taxi (Kaggle)  

---

## Group Members

| Name | Track | Role |
|------|-------|------|
| TBC  | ML    | Project Manager |
| TBC  | ML    | Technical Deputy |
| TBC  | ML    | ML Member |
| TBC  | Cloud | Cloud Member |
| TBC  | Cloud | Cloud Member |

---

## Competition Rules

- Both Group A1 and Group A2 build the same system independently
- MLflow experiments are **separate** — you cannot see rival runs until after the Week 5 checkpoint
- After Week 5 the ML Technical Lead opens visibility — both teams can compare metrics
- **Head-to-head crown:** the group with the better **RMSE** on the test set wins the project crown
- **Overall winner:** highest combined rubric score + crown points across the programme

---

## Repo Structure

```
group-a1/
├── notebooks/          # EDA and exploration — never import from here in production code
├── src/
│   └── features.py     # ALL feature engineering logic — imported by train.py and the API
├── api/
│   └── main.py         # FastAPI prediction API
├── train.py            # Standalone training script — must run with a single command
├── Dockerfile          # Builds the prediction API container
├── requirements.txt    # All dependencies pinned
├── model_card.md       # Model documentation
├── known_issues.md     # Data quality issues and surprises from EDA
└── README.md           # This file
```

---

## API Contract

> **Status: DRAFT — must be finalised and signed by both ML and Cloud leads by end of Week 1**

### POST /predict
**Input:**
```json
{}
```
**Output:**
```json
{}
```

### GET /health
```json
{ "status": "ok" }
```

### GET /model-info
```json
{
  "model_name": "",
  "version": "",
  "stage": "Production",
  "trained_at": "",
  "primary_metric": "RMSE",
  "primary_metric_value": 0.0
}
```

### POST /reload-model
```json
{ "status": "reloading", "new_version": "" }
```

---

## Environment Variables

| Variable | Description | Set by |
|----------|-------------|--------|
| `MLFLOW_TRACKING_URI` | MLflow server URL | Cloud team |
| `MLFLOW_EXPERIMENT_NAME` | Agreed experiment name | Both |
| `MODEL_REGISTRY_NAME` | Agreed registry name | Both |
| `DATA_PATH` | Path or URL to training data | Cloud team |

---

## Week 1 Checklist

- [ ] All group members have repo access
- [ ] API contract fields filled in and agreed
- [ ] MLflow experiment name agreed: ``
- [ ] MLflow server URL received from Cloud team: ``
- [ ] EDA notebook started
- [ ] known_issues.md created

---

## MLflow Naming Convention

- Experiment: `project-a-group-a1`
- Registry: `group-a1-model`
- Visibility: **private until Week 5 checkpoint**
