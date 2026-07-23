"""Script de predicción: Evaluar nuevas muestras de crudo."""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oil_gas_ml.utils.preprocessor import CrudePreprocessor
from oil_gas_ml.models.crude_classifier import CrudeClassifier
from oil_gas_ml.models.crude_regressor import CrudeRegressor
from oil_gas_ml.models.quality_predictor import QualityPredictor


def predict_sample(sample_data: dict):
    print("=" * 60)
    print("  PREDICCIÓN DE CRUDO PETROLÍFERO")
    print("=" * 60)

    classifier = CrudeClassifier.load("outputs/models/crude_classifier_best.pkl")
    regressor = CrudeRegressor.load("outputs/models/crude_regressor_best.pkl")
    predictor = QualityPredictor.load("outputs/models/quality_predictor.pkl")

    preprocessor = CrudePreprocessor(scaler_type="robust")
    preprocessor.fit(CrudePreprocessor._build_derived_features(
        __import__("pandas").read_csv("data/crude_dataset.csv")
    ))

    import pandas as pd
    df_sample = pd.DataFrame([sample_data])
    X = preprocessor.transform(df_sample)

    quality_pred = classifier.predict(X)
    quality_proba = classifier.predict_proba(X)
    value_pred = regressor.predict(X)
    multi_pred = predictor.predict(X)

    print(f"\n  Tipo de crudo (input):       {sample_data.get('crude_type', 'N/A')}")
    print(f"  API Gravity:                  {sample_data.get('api_gravity', 'N/A')} °API")
    print(f"  Viscosidad:                   {sample_data.get('viscosity_cp', 'N/A')} cP")
    print(f"  Azufre:                       {sample_data.get('sulfur_content_pct', 'N/A')} %")
    print(f"\n  --- RESULTADOS ---")
    print(f"  Calidad predicha:             {classifier.class_names[quality_pred[0]]}")
    print(f"  Probabilidades:")
    for i, cls in enumerate(classifier.class_names):
        print(f"    {cls:20s}: {quality_proba[0][i]:.2%}")
    print(f"\n  Valor de mercado estimado:    ${value_pred[0]:.2f} USD/barril")
    print(f"  Rendimiento estimado:         {multi_pred[0].get('yield_recovery_pct', 'N/A')}%")
    print(f"  Valor estimado (multi):       ${multi_pred[0].get('market_value_usd_bbl', 'N/A')} USD/barril")
    print("=" * 60)

    return {
        "quality_class": classifier.class_names[quality_pred[0]],
        "quality_probabilities": dict(zip(classifier.class_names, quality_proba[0].tolist())),
        "market_value_usd_bbl": float(value_pred[0]),
        "multi_predictions": multi_pred[0],
    }


def main():
    sample = {
        "api_gravity": 32.5,
        "viscosity_cp": 45.0,
        "sulfur_content_pct": 1.2,
        "water_content_pct": 2.5,
        "asphaltene_content_pct": 3.0,
        "total_acid_number": 0.8,
        "pour_point_c": 5.0,
        "flash_point_c": 45.0,
        "density_kg_m3": 862.0,
        "rvp_kpa": 55.0,
        "salt_content_ptb": 12.0,
        "metal_content_ppm": 25.0,
        "nitrogen_content_pct": 0.08,
        "carbon_residue_pct": 2.5,
        "crude_type": "mediano",
    }

    result = predict_sample(sample)

    output_path = Path("outputs/prediction_example.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  Resultado guardado en: {output_path}")


if __name__ == "__main__":
    main()
