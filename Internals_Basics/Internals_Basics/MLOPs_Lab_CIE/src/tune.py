import json
import pickle
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

mlflow.set_tracking_uri(f"file://{BASE_DIR}/mlruns")
mlflow.set_experiment("copilotbench-suggestion-accept-rate")

df = pd.read_csv(DATA_PATH)
X = df[["code_context_lines", "language_complexity", "prompt_length", "is_inline"]]
y = df["suggestion_accept_rate"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load best model info from task 1
with open(os.path.join(MODELS_DIR, "best_model_task1.pkl"), "rb") as f:
    task1 = pickle.load(f)

best_name = task1["name"]
print(f"Tuning: {best_name}")

# Parameter grid (applied to GradientBoosting or RF)
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
}
if best_name == "GradientBoosting":
    param_grid["learning_rate"] = [0.01, 0.05, 0.1]
    base_model = GradientBoostingRegressor(random_state=42)
else:
    base_model = RandomForestRegressor(random_state=42)

neg_mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

PARENT_RUN_NAME = "tuning-copilotbench"
N_ITER = 9  # 3x3 grid = 9 combinations (all)

with mlflow.start_run(run_name=PARENT_RUN_NAME) as parent_run:
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_grid,
        n_iter=N_ITER,
        cv=5,
        scoring="neg_mean_absolute_error",
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train)

    # Log each trial as nested run
    for i, params in enumerate(search.cv_results_["params"]):
        with mlflow.start_run(run_name=f"trial_{i+1}", nested=True):
            mlflow.log_params(params)
            cv_mae = -search.cv_results_["mean_test_score"][i]
            mlflow.log_metric("cv_mae", cv_mae)

    # Best model metrics
    best_model = search.best_estimator_
    preds = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    cv_mae = -search.best_score_

    mlflow.log_params(search.best_params_)
    mlflow.log_metric("best_mae", mae)
    mlflow.log_metric("best_rmse", rmse)
    mlflow.log_metric("best_cv_mae", cv_mae)
    mlflow.sklearn.log_model(best_model, artifact_path="tuned_model")

    # Save best tuned model
    with open(os.path.join(MODELS_DIR, "best_model_tuned.pkl"), "wb") as f:
        pickle.dump({
            "model": best_model,
            "name": best_name,
            "run_id": parent_run.info.run_id,
            "mae": mae,
            "rmse": rmse,
            "best_params": search.best_params_
        }, f)

    step2 = {
        "search_type": "random",
        "n_folds": 5,
        "total_trials": N_ITER,
        "best_params": search.best_params_,
        "best_mae": round(mae, 6),
        "best_cv_mae": round(cv_mae, 6),
        "parent_run_name": PARENT_RUN_NAME
    }

    with open(os.path.join(RESULTS_DIR, "step2_s2.json"), "w") as f:
        json.dump(step2, f, indent=2)

    print(f"Best params: {search.best_params_}")
    print(f"Best MAE: {mae:.4f}, RMSE: {rmse:.4f}, CV MAE: {cv_mae:.4f}")
    print("Saved results/step2_s2.json")
