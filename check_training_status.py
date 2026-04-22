import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

# Get all versions
versions = client.search_model_versions("name='group-a1-model'")
versions = sorted(versions, key=lambda x: int(x.version))

# Latest version
latest = versions[-1]
run = client.get_run(latest.run_id)

print("=" * 60)
print("✅ TRAINING COMPLETE - NEW MODEL METRICS")
print("=" * 60)
print(f"Model Version: {latest.version}")
print(f"RMSE: {run.data.metrics.get('rmse'):.2f} seconds")
print(f"RMSLE: {run.data.metrics.get('rmsle'):.4f}")
print(f"Aliases: {latest.aliases}")
print(f"Stage: {latest.current_stage}")

# Compare to previous
if len(versions) > 1:
    previous = versions[-2]
    prev_run = client.get_run(previous.run_id)
    old_rmse = prev_run.data.metrics.get("rmse")
    new_rmse = run.data.metrics.get("rmse")
    improvement = old_rmse - new_rmse
    pct = (improvement / old_rmse) * 100

    print("\n" + "=" * 60)
    print("📊 COMPARISON")
    print("=" * 60)
    print(f"Previous RMSE (v{previous.version}): {old_rmse:.2f}s")
    print(f"New RMSE (v{latest.version}):      {new_rmse:.2f}s")
    print(f"Improvement: {improvement:.2f}s ({pct:.2f}%) ✅")
