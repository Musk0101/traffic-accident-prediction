import pandas as pd
import numpy as np
import joblib

# Load dataset
df = pd.read_csv("data/US_Accidents.csv")

# EXACT feature order used during training
features = [
    "Temperature(F)",
    "Humidity(%)",
    "Wind_Speed(mph)",
    "Visibility(mi)",
    "Pressure(in)"
]

df = df[features].dropna()

# Sample small background set
X_bg = df.sample(100, random_state=42)

# Scale using SAME scaler
scaler = joblib.load("scaler.pkl")
X_bg_scaled = scaler.transform(X_bg)

# Save
np.save("background.npy", X_bg_scaled)

print("✅ background.npy created:", X_bg_scaled.shape)
