from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI(title="Salary Prediction MLOps API")
model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"message": "MLOps Salary Prediction API is Live!"}

@app.post("/predict")
def predict_salary(experience: float):
    prediction = model.predict(np.array([[experience]]))
    return {
        "years_of_experience": experience,
        "predicted_salary_lpa": round(float(prediction[0]), 2)
    }
