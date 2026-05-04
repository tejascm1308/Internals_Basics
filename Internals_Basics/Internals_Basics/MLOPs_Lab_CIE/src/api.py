import json
import pickle
import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

app = FastAPI(title="CopilotBench Suggestion Accept Rate Predictor")

# Load model at startup
model_data = None
try:
    with open(os.path.join(MODELS_DIR, "best_model_tuned.pkl"), "rb") as f:
        model_data = pickle.load(f)
    print(f"Loaded model: {model_data['name']}")
except Exception as e:
    print(f"Error loading model: {e}")


class Features(BaseModel):
    code_context_lines: int = Field(..., ge=5, le=200)
    language_complexity: int = Field(..., ge=1, le=5)
    prompt_length: int = Field(..., ge=10, le=500)
    is_inline: int = Field(..., ge=0, le=1)


@app.get("/ping")
def ping():
    return {"status": "healthy", "model_loaded": model_data is not None}


@app.post("/score")
def score(features: Features):
    X = [[
        features.code_context_lines,
        features.language_complexity,
        features.prompt_length,
        features.is_inline
    ]]
    prediction = float(model_data["model"].predict(X)[0])
    return {"prediction": round(prediction, 6)}


def save_results():
    """Call this after server is ready to save step3 results."""
    import requests
    health = requests.get("http://localhost:8080/ping").json()
    test_input = {"code_context_lines": 132, "language_complexity": 3, "prompt_length": 272, "is_inline": 0}
    pred_resp = requests.post("http://localhost:8080/score", json=test_input).json()

    step3 = {
        "health_endpoint": "/ping",
        "predict_endpoint": "/score",
        "port": 8080,
        "health_response": health,
        "test_input": test_input,
        "prediction": pred_resp["prediction"]
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "step3_s4.json"), "w") as f:
        json.dump(step3, f, indent=2)
    print("Saved results/step3_s4.json")
    print(json.dumps(step3, indent=2))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
