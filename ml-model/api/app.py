from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch

from models.stgnn import STGNN
from graph.build_graph import build_road_graph

# -----------------------------
# App
# -----------------------------
app = FastAPI(title="Traffic Accident Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Input schema
# -----------------------------
class WeatherInput(BaseModel):
    temperature: float = Field(..., ge=-50, le=60)
    humidity: float = Field(..., ge=0, le=100)
    pressure: float = Field(..., ge=800, le=1100)
    visibility: float = Field(..., ge=0, le=50)
    wind_speed: float = Field(..., ge=0, le=50)

# -----------------------------
# Load graph + model ONCE
# -----------------------------
X, edge_index = build_road_graph("data/US_Accidents.csv")
X = torch.tensor(X, dtype=torch.float)
edge_index = torch.tensor(edge_index, dtype=torch.long)

model = STGNN(in_channels=X.shape[1], hidden_channels=32)
model.load_state_dict(torch.load("stgnn_model.pth", map_location="cpu"))
model.eval()

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"status": "API running"}

@app.post("/predict")
def predict(data: WeatherInput):

    # -----------------------------
    # 1️⃣ Environmental Risk (0 → 1)
    # -----------------------------
    env_score = 0.0

    # Visibility (dominant factor)
    if data.visibility < 3:
        env_score += 0.4
    elif data.visibility < 6:
        env_score += 0.25
    elif data.visibility < 10:
        env_score += 0.1

    # Humidity
    if data.humidity > 85:
        env_score += 0.2
    elif data.humidity > 65:
        env_score += 0.1

    # Wind speed
    if data.wind_speed > 12:
        env_score += 0.2
    elif data.wind_speed > 7:
        env_score += 0.1

    # Temperature (cold conditions)
    if data.temperature < 5:
        env_score += 0.1

    # Pressure (storm indicator)
    if data.pressure < 1000:
        env_score += 0.1

    # Clamp env score
    env_score = min(env_score, 1.0)

    # -----------------------------
    # 2️⃣ Structural Risk (ST-GNN)
    # -----------------------------
    with torch.no_grad():
        gnn_output = model(X, edge_index)

    # SAFELY reduce model output to scalar
    structural_risk = gnn_output.mean().item()

    # Normalize influence
    structural_factor = structural_risk * 0.15

    # -----------------------------
    # 3️⃣ Final Risk
    # -----------------------------
    final_risk = min(env_score + structural_factor, 1.0)

    # -----------------------------
    # 4️⃣ Risk Levels
    # -----------------------------
    if final_risk < 0.35:
        level = "Low"
        color = "#2ecc71"
    elif final_risk < 0.65:
        level = "Medium"
        color = "#f1c40f"
    else:
        level = "High"
        color = "#e74c3c"

    # -----------------------------
    # 5️⃣ Debug (keep for now)
    # -----------------------------
    print(
        f"DEBUG | env={env_score:.3f} "
        f"struct={structural_factor:.3f} "
        f"final={final_risk:.3f} "
        f"level={level}"
    )

    return {
        "risk_score": round(final_risk, 4),
        "risk_level": level,
        "explanation": {
            "environmental_risk": round(env_score, 4),
            "structural_risk": round(structural_factor, 4)
        }
    }
