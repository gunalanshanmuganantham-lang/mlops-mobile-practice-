from fastapi import FastAPI
import joblib
import numpy as np
import time

app = FastAPI(title="Salary Prediction MLOps API", version="2.0.0")
model = joblib.load("model.pkl")

# Production Monitoring State
metrics_store = {
    "total_predictions": 0,
    "out_of_range_inputs": 0,
    "prediction_history": [],
    "avg_latency_ms": 0.0
}

@app.get("/")
def home():
    return {
        "status": "Live",
        "message": "End-to-End MLOps Salary Prediction API v2.0",
        "docs_url": "/docs",
        "monitoring_url": "/metrics"
    }

@app.post("/predict")
def predict_salary(experience: float):
    start_time = time.time()
    prediction = model.predict(np.array([[experience]]))[0]
    latency_ms = round((time.time() - start_time) * 1000, 2)

    # Data Drift / Out-of-Range Check (Training range: 1 to 8 years)
    drift_warning = False
    if experience < 1 or experience > 8:
        metrics_store["out_of_range_inputs"] += 1
        drift_warning = True

    # Update Live Metrics
    count = metrics_store["total_predictions"]
    metrics_store["avg_latency_ms"] = round(
        ((metrics_store["avg_latency_ms"] * count) + latency_ms) / (count + 1), 2
    )
    metrics_store["total_predictions"] += 1
    metrics_store["prediction_history"].append(round(float(prediction), 2))

    return {
        "years_of_experience": experience,
        "predicted_salary_lpa": round(float(prediction), 2),
        "latency_ms": latency_ms,
        "data_drift_warning": drift_warning
    }

@app.get("/metrics")
def get_monitoring_metrics():
    history = metrics_store["prediction_history"]
    avg_salary = round(sum(history) / len(history), 2) if history else 0.0
    return {
        "total_predictions": metrics_store["total_predictions"],
        "out_of_range_drift_alerts": metrics_store["out_of_range_inputs"],
        "average_predicted_salary_lpa": avg_salary,
        "average_inference_latency_ms": metrics_store["avg_latency_ms"]
    }
    
