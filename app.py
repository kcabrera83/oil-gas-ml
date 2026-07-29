
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Dict, Any, List
import numpy as np
import joblib, os

security = HTTPBearer()
app = FastAPI(title="Oil & Gas ML")
API_KEY = os.getenv("API_KEY", "dev")

class InferenceInput(BaseModel):
    features: Dict[str, float] = Field(..., description="Input features as key-value pairs")

class InferenceOutput(BaseModel):
    result: float
    confidence: float = 1.0

class ModelRegistry:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        for f in os.listdir("outputs/models"):
            if f.endswith(".pkl"):
                self.models[f.replace(".pkl", "")] = joblib.load(os.path.join("outputs/models", f))
    
    def get(self, name: str) -> Any:
        if name not in self.models:
            raise HTTPException(404, f"Model '{name}' not found. Available: {list(self.models.keys())}")
        return self.models[name]

registry = ModelRegistry()

def verify(cred: HTTPAuthorizationCredentials = Depends(security)):
    if cred.credentials != API_KEY:
        raise HTTPException(401, "Invalid credentials")
    return cred

@app.get("/")
async def root():
    return {"app": "Oil & Gas ML", "version": "2.0", "models": list(registry.models.keys())}

@app.post("/infer/{model_name}")
async def infer(model_name: str, inp: InferenceInput, _=Depends(verify)):
    data = registry.get(model_name)
    feats = data.get("feature_names", list(inp.features.keys()))
    X = np.array([inp.features.get(f, 0) for f in feats]).reshape(1, -1)
    scaler = data.get("scaler")
    if scaler:
        X = scaler.transform(X)
    pred = data["model"].predict(X)[0]
    return InferenceOutput(result=float(pred))
