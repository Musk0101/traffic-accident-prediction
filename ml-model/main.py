from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf

app = FastAPI()

model = tf.keras.models.load_model("traffic_accident_model.h5")
scaler = joblib.load("scaler.pkl")

class TrafficInput(BaseModel):
    temperature: float
    humidity: float
    wind_speed: float
    visibility: float
    pressure: float

@app.post("/predict")
def predict(data: TrafficInput):
    X = np.array([[
        data.temperature,
        data.humidity,
        data.wind_speed,
        data.visibility,
        data.pressure
    ]])

    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0][0]

    return {
        "prediction": float(prediction)
    }
