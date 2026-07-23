"""Servidor web Flask para el sistema ML de evaluación de crudo."""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).resolve().parent))

from oil_gas_ml.data_generator import CrudeDataGenerator
from oil_gas_ml.utils.preprocessor import CrudePreprocessor
from oil_gas_ml.models.crude_classifier import CrudeClassifier
from oil_gas_ml.models.crude_regressor import CrudeRegressor
from oil_gas_ml.models.quality_predictor import QualityPredictor

app = Flask(__name__)
CORS(app)

classifier = None
regressor = None
predictor = None
preprocessor = None
dataset = None


def load_models():
    global classifier, regressor, predictor, preprocessor, dataset
    try:
        classifier = CrudeClassifier.load("outputs/models/crude_classifier_best.pkl")
        regressor = CrudeRegressor.load("outputs/models/crude_regressor_best.pkl")
        predictor = QualityPredictor.load("outputs/models/quality_predictor.pkl")
        dataset = pd.read_csv("data/crude_dataset.csv")
        preprocessor = CrudePreprocessor(scaler_type="robust")
        preprocessor.fit(dataset)
        print("  Modelos cargados correctamente.")
    except Exception as e:
        print(f"  Error cargando modelos: {e}")
        print("  Ejecuta 'python scripts/train.py' primero.")
        gen = CrudeDataGenerator(seed=42)
        dataset = gen.generate(n_samples=3000)
        preprocessor = CrudePreprocessor(scaler_type="robust")
        preprocessor.fit(dataset)


@app.route("/")
def index():
    stats = {}
    if dataset is not None:
        stats = {
            "total_samples": len(dataset),
            "crude_types": dataset["crude_type"].value_counts().to_dict(),
            "quality_classes": dataset["quality_class"].value_counts().to_dict(),
            "avg_api": round(dataset["api_gravity"].mean(), 2),
            "avg_sulfur": round(dataset["sulfur_content_pct"].mean(), 3),
            "avg_viscosity": round(dataset["viscosity_cp"].mean(), 2),
            "avg_market_value": round(dataset["market_value_usd_bbl"].mean(), 2),
            "features": list(dataset.columns),
            "numeric_stats": dataset.describe().round(2).to_dict(),
        }
    return render_template("index.html", stats=stats)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df_input = pd.DataFrame([data])
        X = preprocessor.transform(df_input)

        quality_pred = classifier.predict(X)
        quality_proba = classifier.predict_proba(X)
        value_pred = regressor.predict(X)
        multi_pred = predictor.predict(X)

        result = {
            "quality_class": classifier.class_names[quality_pred[0]],
            "quality_probabilities": {
                classifier.class_names[i]: round(float(quality_proba[0][i]), 4)
                for i in range(len(classifier.class_names))
            },
            "market_value": round(float(value_pred[0]), 2),
            "yield_recovery": round(float(multi_pred[0][0]), 2),
            "multi_market_value": round(float(multi_pred[0][1]), 2),
            "input_data": data,
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/stats")
def api_stats():
    if dataset is None:
        return jsonify({"error": "Dataset no disponible"}), 404
    return jsonify({
        "total_samples": len(dataset),
        "crude_types": dataset["crude_type"].value_counts().to_dict(),
        "quality_classes": dataset["quality_class"].value_counts().to_dict(),
        "avg_api": round(float(dataset["api_gravity"].mean()), 2),
        "avg_sulfur": round(float(dataset["sulfur_content_pct"].mean()), 3),
        "avg_viscosity": round(float(dataset["viscosity_cp"].mean()), 2),
        "avg_market_value": round(float(dataset["market_value_usd_bbl"].mean()), 2),
        "avg_yield": round(float(dataset["yield_recovery_pct"].mean()), 2),
    })


@app.route("/api/distribution/<feature>")
def api_distribution(feature):
    if dataset is None or feature not in dataset.columns:
        return jsonify({"error": "Feature no encontrado"}), 404
    data = dataset[feature].dropna()
    counts, bins = np.histogram(data, bins=30)
    return jsonify({
        "feature": feature,
        "bins": bins.tolist(),
        "counts": counts.tolist(),
        "mean": round(float(data.mean()), 4),
        "std": round(float(data.std()), 4),
        "min": round(float(data.min()), 4),
        "max": round(float(data.max()), 4),
    })


@app.route("/api/correlation")
def api_correlation():
    if dataset is None:
        return jsonify({"error": "Dataset no disponible"}), 404
    numeric = dataset.select_dtypes(include=[np.number])
    corr = numeric.corr()
    return jsonify({
        "features": list(corr.columns),
        "matrix": corr.round(4).values.tolist(),
    })


@app.route("/api/sample/<int:idx>")
def api_sample(idx):
    if dataset is None or idx >= len(dataset):
        return jsonify({"error": "Muestra no encontrada"}), 404
    return jsonify(dataset.iloc[idx].to_dict())


@app.route("/api/model_info")
def api_model_info():
    info = {}
    if classifier is not None:
        info["classifier"] = {
            "name": classifier.model_name,
            "type": "classification",
            "classes": list(classifier.class_names) if classifier.class_names is not None else [],
        }
    if regressor is not None:
        info["regressor"] = {
            "name": regressor.model_name,
            "type": "regression",
            "target": regressor.target_name,
        }
    if predictor is not None:
        info["predictor"] = {
            "base_model": predictor.base_model_name,
            "targets": predictor.target_names,
        }
    return jsonify(info)


if __name__ == "__main__":
    print("=" * 60)
    print("  Servidor Web - Evaluación de Crudo Petrolífero")
    print("=" * 60)
    print("  Cargando modelos...")
    load_models()
    print("  Servidor iniciando en http://127.0.0.1:5001")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5001, debug=True)
