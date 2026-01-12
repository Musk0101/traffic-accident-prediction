from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# ✅ CORS FIX
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow frontend
    allow_credentials=True,
    allow_methods=["*"],  # allow OPTIONS, POST, etc
    allow_headers=["*"],
)

model = joblib.load("traffic_accident_model.pkl")
scaler = joblib.load("scaler.pkl")

class WeatherInput(BaseModel):
    temperature: float
    humidity: float
    pressure: float
    visibility: float
    wind_speed: float

@app.post("/predict")
def predict(data: WeatherInput):
    X = np.array([[
        data.visibility,
        data.wind_speed,
        data.humidity
    ]])

    X_scaled = scaler.transform(X)

    prob = model.predict_proba(X_scaled)[0]

    prob_low = float(prob[0])
    prob_high = float(prob[1])

    prediction = "HIGH" if prob_high >= 0.5 else "LOW"

    return {
        "prediction": prediction,
        "confidence": max(prob_low, prob_high),
        "prob_low": prob_low,
        "prob_high": prob_high
    }
