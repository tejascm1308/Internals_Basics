import json
import pickle
import os
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

mlflow.set_tracking_uri(f"file://{BASE_DIR}/mlruns")

REGISTERED_MODEL_NAME = "copilotbench-suggestion-accept-rate-predictor"

# Load tuned model info
with open(os.path.join(MODELS_DIR, "best_model_tuned.pkl"), "rb") as f:
    model_data = pickle.load(f)

run_id = model_data["run_id"]
rmse = model_data["rmse"]

client = MlflowClient()

# Get the model URI from the run
run = client.get_run(run_id)
artifact_uri = run.info.artifact_uri
model_uri = f"{artifact_uri}/tuned_model"

# Register the model
print(f"Registering model from run: {run_id}")
mv = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)

version = mv.version
print(f"Registered version: {version}")

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
