import streamlit as st
import joblib
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Oil & Gas ML", layout="wide")
st.title("Oil & Gas ML")
st.markdown("Classify crude oil quality and estimate market value.")

@st.cache_resource
def load_models():
    d = Path(__file__).parent / "outputs" / "models"
    return {k: joblib.load(d / v) for k, v in [("quality", "quality_classifier.pkl"), ("value", "market_value_model.pkl")]}

models = load_models()

st.sidebar.header("Input Parameters")
api_gravity = st.sidebar.slider("Api Gravity", 10, 50, 30)
viscosity_cp = st.sidebar.slider("Viscosity Cp", 1, 1000, 500)
sulfur_pct = st.sidebar.slider("Sulfur Pct", 0, 5, 2)
bsw_pct = st.sidebar.slider("Bsw Pct", 0, 10, 5)
asphaltenes_pct = st.sidebar.slider("Asphaltenes Pct", 0, 20, 10)
tan_mgkoh = st.sidebar.slider("Tan Mgkoh", 0, 5, 2)
pour_point_c = st.sidebar.slider("Pour Point C", -40, 30, -5)
flash_point_c = st.sidebar.slider("Flash Point C", 20, 120, 70)
density_kgm3 = st.sidebar.slider("Density Kgm3", 800, 1000, 900)
rvp_psi = st.sidebar.slider("Rvp Psi", 2, 15, 8)
salinity_ptb = st.sidebar.slider("Salinity Ptb", 0, 200, 100)
metals_ppm = st.sidebar.slider("Metals Ppm", 0, 500, 250)
nitrogen_ppm = st.sidebar.slider("Nitrogen Ppm", 0, 1000, 500)
carbon_residue_pct = st.sidebar.slider("Carbon Residue Pct", 0, 15, 7)
vanadium_ppm = st.sidebar.slider("Vanadium Ppm", 0, 500, 250)

if st.sidebar.button("Run Prediction"):
    try:
        features = np.array([[api_gravity, viscosity_cp, sulfur_pct, bsw_pct, asphaltenes_pct, tan_mgkoh, pour_point_c, flash_point_c, density_kgm3, rvp_psi, salinity_ptb, metals_ppm, nitrogen_ppm, carbon_residue_pct, vanadium_ppm]])
        m = models["quality"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Quality", result if isinstance(result, str) else f"{result:.4f}")
        m = models["value"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Value", result if isinstance(result, str) else f"{result:.4f}")
    except Exception as e:
        st.error(f"Error: {e}")

