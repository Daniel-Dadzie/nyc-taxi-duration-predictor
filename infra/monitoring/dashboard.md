# Monitoring Dashboard — group-a1

## Required Metrics

Your Cloud Monitoring dashboard must show all four of these:

| Metric | Where to find it | Target |
|--------|-----------------|--------|
| Request count | Cloud Run → Metrics → Request count | Visible trend over time |
| Request latency (p99) | Cloud Run → Metrics → Request latencies | Under 2 seconds |
| Error rate | Cloud Run → Metrics → Error count / Request count | Under 1% |
| Current model version | Custom metric from /model-info endpoint | Updates after each retrain |

## Setup Steps

1. Go to **console.cloud.google.com/monitoring**
2. Click **Dashboards → Create Dashboard**
3. Add a chart for each metric above
4. Add a title: "group-a1 — Prediction API Monitoring"
5. Share the dashboard URL in this file once it is live

## Dashboard URL
[Paste your Cloud Monitoring dashboard URL here once created]
