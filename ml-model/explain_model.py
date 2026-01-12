import pandas as pd
import numpy as np
import shap
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

# ================================
# 1. LOAD MODEL & SCALER
# ================================
print("📦 Loading model and scaler...")

model = tf.keras.models.load_model("traffic_accident_model.h5")
scaler = joblib.load("scaler.pkl")

# ================================
# 2. LOAD DATA
# ================================
df = pd.read_csv("data/US_Accidents.csv")

# ⚠️ MUST MATCH TRAINING ORDER
features = [
    "Temperature(F)",     # temperature
    "Humidity(%)",        # humidity
    "Wind_Speed(mph)",    # wind_speed
    "Visibility(mi)",     # visibility
    "Pressure(in)"        # pressure
]

df = df[features].dropna()

# Small sample for SHAP
X_sample = df.sample(500, random_state=42)
X_scaled = scaler.transform(X_sample.values)

print("📊 Data prepared for SHAP")
print("X_scaled shape:", X_scaled.shape)

# ================================
# 3. GRADIENT SHAP
# ================================
print("🧠 Running GradientSHAP...")

background = X_scaled[:100]
explain_data = X_scaled[100:300]

explainer = shap.GradientExplainer(model, background)
shap_values = explainer.shap_values(explain_data)

# ✅ Regression model → unwrap list
if isinstance(shap_values, list):
    shap_values = shap_values[0]

print("SHAP values shape:", shap_values.shape)

# ================================
# 4. VISUALIZATIONS
# ================================
print("📈 Generating SHAP plots...")

shap.summary_plot(
    shap_values,
    X_sample.iloc[100:300],
    feature_names=features,
    show=True
)

shap.summary_plot(
    shap_values,
    X_sample.iloc[100:300],
    feature_names=features,
    plot_type="bar",
    show=True
)

print("✅ SHAP explanation completed successfully")

