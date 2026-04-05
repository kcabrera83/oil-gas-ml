"""
model_training.py - Funciones de preprocesamiento, entrenamiento y evaluación.
"""

import logging
import os
from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def calcular_r2_ajustado(r2: float, n_obs: int, n_features: int) -> float:
    if n_obs <= n_features + 1:
        return 0.0
    return 1.0 - (1.0 - r2) * (n_obs - 1) / (n_obs - n_features - 1)


def preparar_datos(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size debe estar en (0, 1)")
    logger.info("Dividiendo datos: %d muestras, test_size=%.2f", len(X), test_size)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def normalizar_datos(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    X_train_norm = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_norm = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    logger.info("Normalización aplicada")
    return X_train_norm, X_test_norm, scaler


def entrenar_modelo(
    X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int = 100, max_depth: Optional[int] = None
) -> RandomForestRegressor:
    logger.info("Entrenando RF: n_estimators=%d, max_depth=%s", n_estimators, max_depth)
    modelo = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1, verbose=0
    )
    modelo.fit(X_train, y_train)
    logger.info("Entrenamiento completado")
    return modelo


def evaluar_modelo_completo(
    modelo: RandomForestRegressor,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cv_folds: int = 5
) -> Dict[str, float]:
    y_pred = modelo.predict(X_test)
    mse = float(mean_squared_error(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    r2_ajustado = float(calcular_r2_ajustado(r2, X_test.shape[0], X_test.shape[1]))
    
    cv_scores = cross_val_score(modelo, X_train, y_train, cv=cv_folds, scoring="r2")
    
    return {
        "mse": mse,
        "mae": mae,
        "rmse": float(np.sqrt(mse)),
        "r2_test": r2,
        "r2_ajustado": r2_ajustado,
        "r2_cv_mean": float(cv_scores.mean()),
        "r2_cv_std": float(cv_scores.std()),
        "cv_folds": cv_folds
    }


def guardar_modelo(modelo: RandomForestRegressor, ruta: str = "models/modelo_produccion.pkl") -> bool:
    try:
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        joblib.dump(modelo, ruta, compress=3)
        logger.info("Modelo guardado en: %s", ruta)
        return True
    except Exception as e:
        logger.error("Error al guardar modelo: %s", e)
        return False


def cargar_modelo(ruta: str) -> RandomForestRegressor:
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"Modelo no encontrado en: {ruta}")
    return joblib.load(ruta)

if __name__ == "__main__":
    main()
