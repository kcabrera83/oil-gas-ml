
import streamlit as st
import numpy as np
from pydantic import BaseModel, Field
from typing import Dict, Any
import joblib, os

st.set_page_config(page_title="Oil & Gas ML", page_icon=":bar_chart:", layout="wide")
st.title("Oil & Gas ML")

class Payload(BaseModel):
    features: Dict[str, float] = Field(default_factory=dict)

class InferenceEngine:
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._load()
    
    def _load(self):
        for f in os.listdir("outputs/models"):
            if f.endswith(".pkl"):
                data = joblib.load(os.path.join("outputs/models", f))
                self._models[f.replace(".pkl", "")] = data
    
    def predict(self, model_key: str, features: dict) -> float:
        data = self._models.get(model_key)
        if not data:
            raise ValueError(f"Model {model_key} not found")
        feats = data.get("feature_names", list(features.keys()))
        X = np.array([features.get(f, 0) for f in feats]).reshape(1, -1)
        if data.get("scaler"):
            X = data["scaler"].transform(X)
        return data["model"].predict(X)[0]

engine = InferenceEngine()

with st.sidebar:
    st.header("Model Selection")
    model_key = st.selectbox("Choose model", list(engine._models.keys()) or ["default"])
    st.divider()
    st.caption("Crude oil property prediction: API, viscosity, sulfur content, and quality grading")

data = engine._models.get(model_key, {})
feats = data.get("feature_names", [f"f{i}" for i in range(4)])
cols = st.columns(3)
inputs = {}
for i, f in enumerate(feats):
    with cols[i % 3]:
        inputs[f] = st.number_input(f.replace("_", " ").title(), value=0.0, key=f)
if st.button("Run inference", type="primary"):
    result = engine.predict(model_key, inputs)
    st.metric("Prediction", f"{result:.4f}")
