import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

import sys; sys.path.append(str(Path(__file__).resolve().parent))

from oil_gas_ml.data_generator import CrudeDataGenerator
from oil_gas_ml.utils.preprocessor import CrudePreprocessor
from oil_gas_ml.models.crude_classifier import CrudeClassifier
from oil_gas_ml.models.crude_regressor import CrudeRegressor
from oil_gas_ml.models.quality_predictor import QualityPredictor

app = FastAPI(title="Oil & Gas ML", version="0.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

models: dict[str, Any] = {}
_model_loaded = False


@app.on_event("startup")
async def load_models():
    global models, _model_loaded
    if _model_loaded:
        return
    _model_loaded = True
    mpath = Path("outputs/models")
    dpath = Path("data/crude_dataset.csv")
    if not mpath.exists() or not dpath.exists():
        print("  Modelos o data no encontrados, generando sinteticos...")
        gen = CrudeDataGenerator(seed=2024)
        dataset = gen.generate(n_samples=3000)
        preprocessor = CrudePreprocessor(scaler_type="robust", random_state=1389)
        preprocessor.fit(dataset)
        models["dataset"] = dataset
        models["preprocessor"] = preprocessor
        return
    try:
        classifier = CrudeClassifier.load(str(mpath / "crude_classifier_best.pkl"))
        regressor = CrudeRegressor.load(str(mpath / "crude_regressor_best.pkl"))
        predictor = QualityPredictor.load(str(mpath / "quality_predictor.pkl"))
        dataset = pd.read_csv(str(dpath))
        preprocessor = CrudePreprocessor(scaler_type="robust", random_state=1389)
        preprocessor.fit(dataset)
        models["classifier"] = classifier
        models["regressor"] = regressor
        models["predictor"] = predictor
        models["preprocessor"] = preprocessor
        models["dataset"] = dataset
    except Exception as e:
        print(f"  Error: {e}")


class PredictRequest(BaseModel):
    api_gravity: float = 32.0
    sulfur_content_pct: float = 1.5
    viscosity_cp: float = 50.0
    density_kg_m3: float = 870.0
    pour_point_c: float = 10.0
    asphaltene_content_pct: float = 5.0
    water_content_pct: float = 3.0
    total_acid_number: float = 1.5
    flash_point_c: float = 60.0
    rvp_kpa: float = 30.0
    salt_content_ptb: float = 15.0
    metal_content_ppm: float = 40.0
    nitrogen_content_pct: float = 0.3
    carbon_residue_pct: float = 2.0
    crude_type: str = "mediano"


class PredictResponse(BaseModel):
    quality_class: str
    quality_probabilities: dict[str, float]
    market_value: float
    yield_recovery: float
    multi_market_value: float
    input_data: dict


class StatsResponse(BaseModel):
    total_samples: int
    crude_types: dict
    quality_classes: dict
    avg_api: float
    avg_sulfur: float
    avg_viscosity: float
    avg_market_value: float
    avg_yield: float


class DistributionResponse(BaseModel):
    feature: str
    bins: list[float]
    counts: list[int]
    mean: float
    std: float
    min: float
    max: float


class CorrelationResponse(BaseModel):
    features: list[str]
    matrix: list[list[float]]


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": {
            "classifier": "classifier" in models,
            "regressor": "regressor" in models,
            "predictor": "predictor" in models,
        },
    }


@app.get("/api/models")
async def models_info():
    info = {}
    if "classifier" in models:
        c = models["classifier"]
        info["classifier"] = {
            "name": c.model_name,
            "type": "classification",
            "classes": list(c.class_names) if c.class_names is not None else [],
        }
    if "regressor" in models:
        r = models["regressor"]
        info["regressor"] = {
            "name": r.model_name,
            "type": "regression",
            "target": r.target_name,
        }
    if "predictor" in models:
        p = models["predictor"]
        info["predictor"] = {
            "base_model": p.base_model_name,
            "targets": p.target_names,
        }
    return info


@app.post("/api/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    dataset = models.get("dataset")
    preprocessor = models.get("preprocessor")
    if not all(k in models for k in ("classifier", "regressor", "predictor")):
        raise HTTPException(status_code=503, detail="Models not fully loaded")
    try:
        df_input = pd.DataFrame([request.model_dump()])
        X = preprocessor.transform(df_input)
        quality_pred = models["classifier"].predict(X)
        quality_proba = models["classifier"].predict_proba(X)
        value_pred = models["regressor"].predict(X)
        multi_pred = models["predictor"].predict(X)
        return PredictResponse(
            quality_class=models["classifier"].class_names[quality_pred[0]],
            quality_probabilities={
                models["classifier"].class_names[i]: round(float(quality_proba[0][i]), 4)
                for i in range(len(models["classifier"].class_names))
            },
            market_value=round(float(value_pred[0]), 2),
            yield_recovery=round(float(multi_pred[0][0]), 2),
            multi_market_value=round(float(multi_pred[0][1]), 2),
            input_data=request.model_dump(),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/stats", response_model=StatsResponse)
async def api_stats():
    dataset = models.get("dataset")
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not available")
    return StatsResponse(
        total_samples=len(dataset),
        crude_types=dataset["crude_type"].value_counts().to_dict(),
        quality_classes=dataset["quality_class"].value_counts().to_dict(),
        avg_api=round(float(dataset["api_gravity"].mean()), 2),
        avg_sulfur=round(float(dataset["sulfur_content_pct"].mean()), 3),
        avg_viscosity=round(float(dataset["viscosity_cp"].mean()), 2),
        avg_market_value=round(float(dataset["market_value_usd_bbl"].mean()), 2),
        avg_yield=round(float(dataset["yield_recovery_pct"].mean()), 2),
    )


@app.get("/api/distribution/{feature}", response_model=DistributionResponse)
async def api_distribution(feature: str):
    dataset = models.get("dataset")
    if dataset is None or feature not in dataset.columns:
        raise HTTPException(status_code=404, detail="Feature not found")
    data = dataset[feature].dropna()
    counts, bins = np.histogram(data, bins=30)
    return DistributionResponse(
        feature=feature,
        bins=bins.tolist(),
        counts=counts.tolist(),
        mean=round(float(data.mean()), 4),
        std=round(float(data.std()), 4),
        min=round(float(data.min()), 4),
        max=round(float(data.max()), 4),
    )


@app.get("/api/correlation", response_model=CorrelationResponse)
async def api_correlation():
    dataset = models.get("dataset")
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not available")
    numeric = dataset.select_dtypes(include=[np.number])
    corr = numeric.corr()
    return CorrelationResponse(
        features=list(corr.columns),
        matrix=corr.round(4).values.tolist(),
    )


@app.get("/api/sample/{idx}")
async def api_sample(idx: int):
    dataset = models.get("dataset")
    if dataset is None or idx >= len(dataset):
        raise HTTPException(status_code=404, detail="Sample not found")
    return dataset.iloc[idx].to_dict()


if __name__ == "__main__":
    import uvicorn
    print("Iniciando servidor de evaluacion de crudo...")
    uvicorn.run(app, host="0.0.0.0", port=5001, log_level="info")

