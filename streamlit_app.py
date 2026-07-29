import streamlit as st, joblib, numpy as np, matplotlib.pyplot as plt
from pathlib import Path; import sys; sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Crude Oil Classifier", layout="centered")
st.title("Crude Oil Classifier")

path = Path(__file__).parent / 'outputs' / 'models'
models = {}
models['grade'] = joblib.load(path / 'quality_classifier.pkl')
models['value'] = joblib.load(path / 'market_value_model.pkl')

def pipeline(x):
    out = {}
    m = models['grade']
    if isinstance(m, dict):
        p = m['model'].predict(m['scaler'].transform(x))
        out['grade'] = m['label_encoder'].inverse_transform(p)[0] if 'label_encoder' in m else float(p[0])
    else:
        out['grade'] = float(m.predict(x)[0])
    m = models['value']
    if isinstance(m, dict):
        p = m['model'].predict(m['scaler'].transform(x))
        out['value'] = m['label_encoder'].inverse_transform(p)[0] if 'label_encoder' in m else float(p[0])
    else:
        out['value'] = float(m.predict(x)[0])
    return out

with st.form('inputs'):
    st.subheader('Input Parameters')
    cols = st.columns(2)
    api = cols[0].slider('Api', 10, 50, 30)
    visc = cols[1].slider('Visc', 1, 1000, 500)
    sulfur = cols[0].slider('Sulfur', 0, 5, 2)
    bsw = cols[1].slider('Bsw', 0, 10, 5)
    asph = cols[0].slider('Asph', 0, 20, 10)
    tan = cols[1].slider('Tan', 0, 5, 2)
    pour = cols[0].slider('Pour', -40, 30, -5)
    flash = cols[1].slider('Flash', 20, 120, 70)
    density = cols[0].slider('Density', 800, 1000, 900)
    rvp = cols[1].slider('Rvp', 2, 15, 8)
    submitted = st.form_submit_button('Run', type='primary', use_container_width=True)

if submitted:
    results = pipeline(np.array([[api, visc, sulfur, bsw, asph, tan, pour, flash, density, rvp]]))
    st.divider()
    st.subheader('Results')
    mc = st.columns(len(results))
    for i, (k, v) in enumerate(results.items()):
        val = str(v) if isinstance(v, str) else f'{v:,.2f}'
        mc[i].metric(k.replace('_',' ').title(), val)