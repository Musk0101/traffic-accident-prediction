import os
import torch
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from models.stgnn import STGNN

# -----------------------------
# Download model from Google Drive if not present
# -----------------------------
MODEL_PATH = "stgnn_model.pth"
GDRIVE_FILE_ID = "1Cy-OC_fqoqxLm_aMnrX_J4J-uxk3yuZX"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
        response = requests.get(url, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Model downloaded successfully.")

download_model()

# -----------------------------
# App
# -----------------------------
app = FastAPI(title="Traffic Accident Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Input schema
# -----------------------------
class WeatherInput(BaseModel):
    temperature: float = Field(..., ge=-50, le=60)
    humidity:    float = Field(..., ge=0,   le=100)
    pressure:    float = Field(..., ge=800, le=1100)
    visibility:  float = Field(..., ge=0,   le=50)
    wind_speed:  float = Field(..., ge=0,   le=50)

# -----------------------------
# Load model ONCE (without CSV)
# -----------------------------
# Use a small fixed graph for production (no CSV needed)
NUM_NODES = 10
IN_CHANNELS = 5

X = torch.randn(NUM_NODES, IN_CHANNELS)
edge_index = torch.tensor([
    [0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9],
    [1,0,2,1,3,2,4,3,5,4,6,5,7,6,8,7,9,8]
], dtype=torch.long)

model = STGNN(in_channels=IN_CHANNELS, hidden_channels=32)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"status": "Traffic Accident Prediction API running"}

@app.post("/predict")
def predict(data: WeatherInput):
    # 1 — Environmental Risk
    env_score = 0.0
    if data.visibility < 3:
        env_score += 0.4
    elif data.visibility < 6:
        env_score += 0.25
    elif data.visibility < 10:
        env_score += 0.1
    if data.humidity > 85:
        env_score += 0.2
    elif data.humidity > 65:
        env_score += 0.1
    if data.wind_speed > 12:
        env_score += 0.2
    elif data.wind_speed > 7:
        env_score += 0.1
    if data.temperature < 5:
        env_score += 0.1
    if data.pressure < 1000:
        env_score += 0.1
    env_score = min(env_score, 1.0)

    # 2 — Structural Risk (STGNN)
    with torch.no_grad():
        gnn_output = model(X, edge_index)
    structural_factor = min(gnn_output.mean().item() * 0.15, 1.0)

    # 3 — Final Risk
    final_risk = round(min(env_score + structural_factor, 1.0), 4)

    # 4 — Level
    if final_risk < 0.35:
        level = "Low"
    elif final_risk < 0.65:
        level = "Medium"
    else:
        level = "High"

    # 5 — Explanation as STRING (not object)
    explanation = (
        f"Environmental risk: {round(env_score*100,1)}% | "
        f"Structural risk: {round(structural_factor*100,1)}% | "
        f"Combined score: {round(final_risk*100,1)}/100"
    )

    return {
        "risk_score": final_risk,
        "risk_level": level,
        "explanation": explanation,
        "factors": {
            "environmental_risk": round(env_score, 4),
            "structural_risk":    round(structural_factor, 4),
        }
    }
