import json
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Set MLflow tracking URI
mlflow.set_tracking_uri("file:///" + os.path.join(BASE_DIR, "mlruns").replace("\\", "/"))
EXPERIMENT_NAME = "copilotbench-suggestion-accept-rate"
mlflow.set_experiment(EXPERIMENT_NAME)

# Load data
df = pd.read_csv(DATA_PATH)
X = df[["code_context_lines", "language_complexity", "prompt_length", "is_inline"]]
y = df["suggestion_accept_rate"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models_config = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = []

for name, model in models_config.items():
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tag("experiment_type", "baseline_comparison")

        # Log params
        for k, v in model.get_params().items():
            mlflow.log_param(k, v)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        mlflow.sklearn.log_model(model, artifact_path="model")

        results.append({"name": name, "mae": round(mae, 6), "rmse": round(rmse, 6), "run_id": run.info.run_id})
        print(f"{name}: MAE={mae:.4f}, RMSE={rmse:.4f}")

# Save best model
best = min(results, key=lambda x: x["rmse"])
print(f"\nBest model: {best['name']} with RMSE={best['rmse']}")

# Save best model to disk for next tasks
import pickle
best_name = best["name"]
best_model = models_config[best_name]
best_model.fit(X_train, y_train)  # refit
with open(os.path.join(MODELS_DIR, "best_model_task1.pkl"), "wb") as f:
    pickle.dump({"model": best_model, "name": best_name, "run_id": best["run_id"]}, f)

step1 = {
    "experiment_name": EXPERIMENT_NAME,
    "models": [{"name": r["name"], "mae": r["mae"], "rmse": r["rmse"]} for r in results],
    "best_model": best["name"],
    "best_metric_name": "rmse",
    "best_metric_value": best["rmse"]
}

with open(os.path.join(RESULTS_DIR, "step1_s1.json"), "w") as f:
    json.dump(step1, f, indent=2)

print("Saved results/step1_s1.json")
