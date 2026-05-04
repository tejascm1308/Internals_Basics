import json
import pickle
import os
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

mlflow.set_tracking_uri("file:///" + os.path.join(BASE_DIR, "mlruns").replace("\\", "/"))

REGISTERED_MODEL_NAME = "copilotbench-suggestion-accept-rate-predictor"

# Load tuned model info
with open(os.path.join(MODELS_DIR, "best_model_tuned.pkl"), "rb") as f:
    model_data = pickle.load(f)

rmse = model_data["rmse"]

# Find the tuning-copilotbench run from YOUR local mlruns
client = MlflowClient()
experiment = client.get_experiment_by_name("copilotbench-suggestion-accept-rate")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.mlflow.runName = 'tuning-copilotbench'",
    order_by=["start_time DESC"],
    max_results=1
)

run = runs[0]
run_id = run.info.run_id
artifact_uri = run.info.artifact_uri
model_uri = f"{artifact_uri}/tuned_model"

print(f"Found run_id: {run_id}")
print(f"Registering model from: {model_uri}")

mv = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
version = mv.version

step4 = {
    "registered_model_name": REGISTERED_MODEL_NAME,
    "version": int(version),
    "run_id": run_id,
    "source_metric": "rmse",
    "source_metric_value": round(rmse, 6)
}

with open(os.path.join(RESULTS_DIR, "step4_s6.json"), "w") as f:
    json.dump(step4, f, indent=2)

print("Saved results/step4_s6.json")
print(json.dumps(step4, indent=2))