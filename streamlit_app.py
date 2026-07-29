import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Oil & Gas ML", layout="wide")
st.title("Oil & Gas ML")
st.markdown("Classify crude oil quality and estimate market value.")

import joblib, numpy as np
d = Path(__file__).parent / 'outputs' / 'models'
models = {'quality': joblib.load(d / 'quality_classifier.pkl'), 'value': joblib.load(d / 'market_value_model.pkl')}

st.sidebar.header("Input Parameters")
api_gravity = st.sidebar.slider('Api Gravity', 10, 50, 30)
viscosity = st.sidebar.slider('Viscosity', 1, 1000, 500)
sulfur = st.sidebar.slider('Sulfur', 0, 5, 2)
bsw = st.sidebar.slider('Bsw', 0, 10, 5)
asphaltenes = st.sidebar.slider('Asphaltenes', 0, 20, 10)
tan = st.sidebar.slider('Tan', 0, 5, 2)
pour_point = st.sidebar.slider('Pour Point', -40, 30, -5)
flash_point = st.sidebar.slider('Flash Point', 20, 120, 70)
density = st.sidebar.slider('Density', 800, 1000, 900)
rvp = st.sidebar.slider('Rvp', 2, 15, 8)
salinity = st.sidebar.slider('Salinity', 0, 200, 100)
metals = st.sidebar.slider('Metals', 0, 500, 250)
nitrogen = st.sidebar.slider('Nitrogen', 0, 1000, 500)
carbon_residue = st.sidebar.slider('Carbon Residue', 0, 15, 7)
vanadium = st.sidebar.slider('Vanadium', 0, 500, 250)

if st.sidebar.button("Run"):
    try:
        x = np.array([[api_gravity, viscosity, sulfur, bsw, asphaltenes, tan, pour_point, flash_point, density, rvp, salinity, metals, nitrogen, carbon_residue, vanadium]])
        cols = st.columns(2)
        for i, (k, m) in enumerate(models.items()):
            X = m['scaler'].transform(x)
            p = m['model'].predict(X)
            if 'label_encoder' in m:
                val = m['label_encoder'].inverse_transform(p)[0]
            else:
                val = f'{p[0]:.2f}'
            cols[i].metric(k.title(), val)
    except Exception as e:
        st.error(str(e))