"""
Módulo para entrenamiento y evaluación de modelos de Machine Learning.
Aplicado a predicción de variables de yacimientos y producción.
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from typing import Tuple, Any
import numpy as np
import pandas as pd


def preparar_datos(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    
    Args:
        X: Variables predictoras (features).
        y: Variable objetivo (target).
        test_size: Proporción de datos para prueba (default 0.2 = 20%).
        
    Returns:
        Tuple con X_train, X_test, y_train, y_test.
    """
    return train_test_split(X, y, test_size=test_size, random_state=42)


def entrenar_modelo(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """
    Entrena un modelo de Random Forest Regressor.
    
    Args:
        X_train: Datos de entrenamiento (features).
        y_train: Datos de entrenamiento (target).
        
    Returns:
        Modelo entrenado.
    """
    modelo = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1  # Usar todos los núcleos del CPU
    )
    modelo.fit(X_train, y_train)
    return modelo


def evaluar_modelo(modelo: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, float]:
    """
    Evalúa el modelo y retorna métricas (MSE y R2).
    
    Args:
        modelo: Modelo entrenado.
        X_test: Datos de prueba (features).
        y_test: Datos de prueba (target).
        
    Returns:
        Tuple con MSE (Mean Squared Error) y R² (R-squared).
    """
    predicciones = modelo.predict(X_test)
    mse = mean_squared_error(y_test, predicciones)
    r2 = r2_score(y_test, predicciones)
    return mse, r2
def evaluar_modelo_completo(modelo, X_train, X_test, y_train, y_test):
    """
    Evalua el modelo con multiples metricas y validacion cruzada.
    """
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.model_selection import cross_val_score
    
    # Metricas en conjunto de prueba
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # R² ajustado
    n_obs, n_feat = X_test.shape
    r2_adj = r2_ajustado(r2, n_obs, n_feat)
    
    # Validacion cruzada en entrenamiento
    cv_scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring='r2')
    
    return {
        'mse': mse,
        'mae': mae,
        'r2_test': r2,
        'r2_ajustado': r2_adj,
        'r2_cv_mean': cv_scores.mean(),
        'r2_cv_std': cv_scores.std()
    }

def guardar_modelo(modelo: RandomForestRegressor, ruta_archivo: str = "models/modelo_produccion.pkl") -> None:
    """
    Guarda el modelo entrenado en disco usando joblib.
    
    Args:
        modelo: Modelo a guardar.
        ruta_archivo: Path donde se guardará el modelo.
    """
    joblib.dump(modelo, ruta_archivo)
    print(f"Modelo guardado en {ruta_archivo}")


def cargar_modelo(ruta_archivo: str) -> RandomForestRegressor:
    """
    Carga un modelo previamente guardado.
    
    Args:
        ruta_archivo: Path del modelo guardado.
        
    Returns:
        Modelo cargado.
    """
    return joblib.load(ruta_archivo)
