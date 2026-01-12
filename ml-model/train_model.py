import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

print("📥 Loading dataset...")

df = pd.read_csv("data/US_Accidents.csv")

# Select useful columns
df = df[
    ["Visibility(mi)", "Wind_Speed(mph)", "Humidity(%)"]
].dropna()

# Risk logic (REALISTIC)
df["risk"] = (
    (df["Visibility(mi)"] < 3) |
    (df["Wind_Speed(mph)"] > 15) |
    (df["Humidity(%)"] > 85)
).astype(int)

X = df[["Visibility(mi)", "Wind_Speed(mph)", "Humidity(%)"]]
y = df["risk"]

print("📊 Risk distribution:")
print(y.value_counts(normalize=True))

# Train / Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model (probability enabled)
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

print("🚀 Training model...")
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model + scaler
joblib.dump(model, "traffic_accident_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ MODEL SAVED AS PKL")
