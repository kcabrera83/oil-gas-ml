"""
Módulo para entrenamiento y evaluación de modelos de Machine Learning.
Aplicado a predicción de variables de yacimientos y producción.
"""

import logging
import os
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Configuración de logging para trazabilidad en producción
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def calcular_r2_ajustado(r2: float, n_obs: int, n_features: int) -> float:
    """Calcula el R² ajustado penalizando por el número de predictores."""
    if n_obs <= n_features + 1:
        return 0.0
    return 1.0 - (1.0 - r2) * (n_obs - 1) / (n_obs - n_features - 1)


def preparar_datos(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide los datos en conjuntos de entrenamiento y prueba."""
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size debe ser un valor entre 0 y 1 (excluyente).")
    return train_test_split(X, y, test_size=test_size, random_state=42)


def entrenar_modelo(
    X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int = 100
) -> RandomForestRegressor:
    """Entrena un modelo Random Forest Regressor optimizado."""
    modelo = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1
    )
    modelo.fit(X_train, y_train)
    return modelo


def evaluar_modelo(
    modelo: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    """Evalúa el modelo y retorna métricas básicas."""
    predicciones = modelo.predict(X_test)
    return {
        "mse": mean_squared_error(y_test, predicciones),
        "r2": r2_score(y_test, predicciones)
    }


def evaluar_modelo_completo(
    modelo: RandomForestRegressor,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cv_folds: int = 5
) -> Dict[str, float]:
    """Evalúa el modelo con métricas avanzadas y validación cruzada."""
    y_pred = modelo.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    r2_ajustado = calcular_r2_ajustado(r2, n_obs=X_test.shape[0], n_features=X_test.shape[1])
    
    cv_scores = cross_val_score(modelo, X_train, y_train, cv=cv_folds, scoring="r2")
    
    return {
        "mse": mse,
        "mae": mae,
        "r2_test": r2,
        "r2_ajustado": r2_ajustado,
        "r2_cv_mean": cv_scores.mean(),
        "r2_cv_std": cv_scores.std()
    }


def guardar_modelo(
    modelo: RandomForestRegressor, ruta_archivo: str = "models/modelo_produccion.pkl"
) -> None:
    """Guarda el modelo entrenado en disco."""
    os.makedirs(os.path.dirname(os.path.abspath(ruta_archivo)), exist_ok=True)
    joblib.dump(modelo, ruta_archivo)
    logger.info("Modelo guardado correctamente en: %s", ruta_archivo)


def cargar_modelo(ruta_archivo: str) -> RandomForestRegressor:
    """Carga un modelo previamente guardado."""
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"No se encontró el modelo en: {ruta_archivo}")
    logger.info("Cargando modelo desde: %s", ruta_archivo)
    return joblib.load(ruta_archivo)


# Ejemplo de flujo lógico de uso
if __name__ == "__main__":
    # 1. Preparación de datos (simulados)
    # X_dummy = pd.DataFrame(np.random.rand(100, 5), columns=[f"feat_{i}" for i in range(5)])
    # y_dummy = pd.Series(np.random.rand(100) * 50)
    # X_train, X_test, y_train, y_test = preparar_datos(X_dummy, y_dummy)
    
    # 2. Entrenamiento
    # modelo = entrenar_modelo(X_train, y_train)
    
    # 3. Evaluación
    # metricas = evaluar_modelo_completo(modelo, X_train, X_test, y_train, y_test)
    # logger.info("Métricas de evaluación: %s", metricas)
    
    # 4. Persistencia
    # guardar_modelo(modelo)
    # modelo_cargado = cargar_modelo("models/modelo_produccion.pkl")
    pass
