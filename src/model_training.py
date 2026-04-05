"""
pipeline_ml.py - Script completo para entrenamiento y evaluación de modelos ML.
Aplicado a predicción de variables de yacimientos y producción.
Ejecución: python pipeline_ml.py
"""

import logging
import os
import sys
from typing import Tuple, Dict, Union, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(module)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ============================================================================
# FUNCIONES DE UTILIDAD Y MÉTRICAS
# ============================================================================
def calcular_r2_ajustado(r2: float, n_obs: int, n_features: int) -> float:
    """Calcula el R² ajustado: 1 - (1-R²)*(n-1)/(n-p-1)"""
    if n_obs <= n_features + 1:
        return 0.0
    return 1.0 - (1.0 - r2) * (n_obs - 1) / (n_obs - n_features - 1)


def log_metricas(metricas: Dict[str, Union[float, int, str]], prefijo: str = "Métricas") -> None:
    """Registra métricas de forma segura manejando tipos mixtos."""
    try:
        partes = []
        for k, v in metricas.items():
            if isinstance(v, (int, float)):
                partes.append(f"{k}: {float(v):.4f}")
            else:
                partes.append(f"{k}: {v}")
        logger.info("%s | %s", prefijo, " | ".join(partes))
    except Exception as e:
        logger.warning("Error al formatear métricas: %s", e)


# ============================================================================
# FUNCIONES DE PREPROCESAMIENTO
# ============================================================================
def preparar_datos(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide datos en entrenamiento y prueba con validaciones."""
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
        raise TypeError("X debe ser DataFrame y y debe ser Series")
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size debe estar en (0, 1)")
    if len(X) != len(y):
        raise ValueError("X e y deben tener la misma longitud")
    
    logger.info("Dividiendo datos: %d muestras, test_size=%.2f", len(X), test_size)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def limpiar_datos(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Limpia datos: elimina nulos y separa features/target."""
    if target_column not in df.columns:
        raise ValueError(f"Columna '{target_column}' no encontrada")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Eliminar filas con nulos
    mask = X.notna().all(axis=1) & y.notna()
    X_clean, y_clean = X.loc[mask], y.loc[mask]
    
    logger.info("Datos limpios: %d/%d muestras válidas", len(X_clean), len(df))
    return X_clean, y_clean


# ============================================================================
# FUNCIONES DE MODELADO
# ============================================================================
def entrenar_modelo(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: int = 42
) -> RandomForestRegressor:
    """Entrena un Random Forest Regressor."""
    logger.info("Entrenando RF: n_estimators=%d, max_depth=%s", n_estimators, max_depth)
    
    modelo = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    modelo.fit(X_train, y_train)
    logger.info("Entrenamiento completado")
    return modelo


def evaluar_modelo(
    modelo: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """Evalúa modelo y retorna métricas básicas como floats nativos."""
    predicciones = modelo.predict(X_test)
    mse = mean_squared_error(y_test, predicciones)
    
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_test, predicciones)),
        "r2": float(r2_score(y_test, predicciones))
    }


def evaluar_modelo_completo(
    modelo: RandomForestRegressor,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cv_folds: int = 5
) -> Dict[str, float]:
    """Evalúa con métricas avanzadas y validación cruzada."""
    # Métricas en test
    basicas = evaluar_modelo(modelo, X_test, y_test)
    
    # R² ajustado
    n_obs, n_feat = X_test.shape
    basicas["r2_ajustado"] = float(calcular_r2_ajustado(basicas["r2"], n_obs, n_feat))
    
    # Validación cruzada
    cv_scores = cross_val_score(modelo, X_train, y_train, cv=cv_folds, scoring="r2")
    basicas["r2_cv_mean"] = float(cv_scores.mean())
    basicas["r2_cv_std"] = float(cv_scores.std())
    basicas["cv_folds"] = cv_folds
    
    return basicas


# ============================================================================
# FUNCIONES DE PERSISTENCIA
# ============================================================================
def guardar_modelo(
    modelo: RandomForestRegressor,
    ruta_archivo: str = "models/modelo_produccion.pkl"
) -> bool:
    """Guarda el modelo en disco con compresión."""
    try:
        directorio = Path(ruta_archivo).parent
        directorio.mkdir(parents=True, exist_ok=True)
        joblib.dump(modelo, ruta_archivo, compress=3)
        logger.info("Modelo guardado en: %s", ruta_archivo)
        return True
    except Exception as e:
        logger.error("Error al guardar modelo: %s", e)
        return False


def cargar_modelo(ruta_archivo: str) -> RandomForestRegressor:
    """Carga un modelo guardado."""
    if not Path(ruta_archivo).exists():
        raise FileNotFoundError(f"Modelo no encontrado: {ruta_archivo}")
    logger.info("Cargando modelo desde: %s", ruta_archivo)
    return joblib.load(ruta_archivo)


# ============================================================================
# FUNCIONES DE DATOS (CARGA Y SIMULACIÓN)
# ============================================================================
def cargar_datos_csv(ruta: str, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Carga datos desde CSV y los limpia."""
    if not Path(ruta).exists():
        raise FileNotFoundError(f"Archivo no encontrado: {ruta}")
    
    df = pd.read_csv(ruta)
    logger.info("Datos cargados: %d filas × %d columnas", len(df), len(df.columns))
    return limpiar_datos(df, target_column)


def generar_datos_simulados(
    n_samples: int = 500,
    n_features: int = 10,
    noise_level: float = 0.5
) -> Tuple[pd.DataFrame, pd.Series]:
    """Genera datos sintéticos para pruebas."""
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    coeficientes = np.random.randn(n_features)
    y = pd.Series(
        X @ coeficientes + np.random.normal(0, noise_level, n_samples),
        name="produccion"
    )
    logger.info("Datos simulados: %d muestras, %d features, ruido=%.2f", 
                n_samples, n_features, noise_level)
    return X, y


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================
def ejecutar_pipeline(
    ruta_datos: Optional[str] = None,
    target_column: str = "produccion",
    test_size: float = 0.2,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    guardar: bool = True,
    ruta_modelo: str = "models/modelo_produccion.pkl",
    usar_simulados: bool = False
) -> Dict:
    """
    Ejecuta el pipeline completo de ML.
    
    Returns:
        Diccionario con resultados, métricas y estado del pipeline.
    """
    resultados = {"exitoso": False, "metricas": {}, "error": None}
    
    try:
        # 1. Cargar/generar datos
        if usar_simulados or (ruta_datos and not Path(ruta_datos).exists()):
            logger.info("Usando datos simulados")
            X, y = generar_datos_simulados()
        else:
            logger.info("Cargando datos desde: %s", ruta_datos)
            X, y = cargar_datos_csv(ruta_datos, target_column)
        
        # 2. Dividir datos
        X_train, X_test, y_train, y_test = preparar_datos(X, y, test_size=test_size)
        resultados["tamano_datos"] = {"train": len(X_train), "test": len(X_test)}
        
        # 3. Entrenar modelo
        modelo = entrenar_modelo(X_train, y_train, n_estimators=n_estimators, max_depth=max_depth)
        resultados["config_modelo"] = {"n_estimators": n_estimators, "max_depth": max_depth}
        
        # 4. Evaluar
        metricas = evaluar_modelo_completo(modelo, X_train, X_test, y_train, y_test)
        resultados["metricas"] = metricas
        log_metricas(metricas, prefijo="Resultados finales")
        
        # 5. Guardar modelo
        if guardar:
            resultados["guardado"] = guardar_modelo(modelo, ruta_modelo)
        
        # 6. Verificar carga
        if guardar and resultados.get("guardado"):
            modelo_cargado = cargar_modelo(ruta_modelo)
            pred_orig = modelo.predict(X_test.iloc[:5])
            pred_carg = modelo_cargado.predict(X_test.iloc[:5])
            resultados["verificacion"] = bool(np.allclose(pred_orig, pred_carg))
            logger.info("Verificación de carga: %s", 
                       "OK" if resultados["verificacion"] else "FALLÓ")
        
        resultados["exitoso"] = True
        logger.info("Pipeline completado exitosamente")
        
    except Exception as e:
        logger.error("Error crítico en pipeline: %s", e, exc_info=True)
        resultados["error"] = str(e)
    
    return resultados


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================
def main():
    """Punto de entrada principal."""
    logger.info("=== Pipeline ML para predicción de producción ===")
    
    # Configuración (personalizar según necesidad)
    config = {
        "ruta_datos": None,                    # Ej: "data/produccion.csv"
        "target_column": "produccion",         # Columna objetivo
        "test_size": 0.2,                      # 20% para prueba
        "n_estimators": 100,                   # Árboles en Random Forest
        "max_depth": None,                     # Profundidad máxima (None = ilimitada)
        "guardar": True,                       # ¿Guardar modelo?
        "ruta_modelo": "models/modelo.pkl",    # Ruta de salida
        "usar_simulados": True                 # Usar datos de prueba (cambiar a False con datos reales)
    }
    
    # Ejecutar pipeline
    resultados = ejecutar_pipeline(**config)
    
    # Resumen final
    print("\n" + "="*60)
    if resultados["exitoso"]:
        logger.info("✓ PIPELINE EXITOSO")
        logger.info("Muestras - Train: %d | Test: %d", 
                   resultados["tamano_datos"]["train"], 
                   resultados["tamano_datos"]["test"])
        m = resultados["metricas"]
        logger.info("R² Test: %.4f | R² Ajustado: %.4f | RMSE: %.4f", 
                   m["r2_test"], m["r2_ajustado"], m["rmse"])
        logger.info("R² CV: %.4f ± %.4f", m["r2_cv_mean"], m["r2_cv_std"])
        if resultados.get("guardado"):
            logger.info("Modelo guardado en: %s", config["ruta_modelo"])
    else:
        logger.error("✗ PIPELINE FALLÓ: %s", resultados["error"])
        sys.exit(1)
    print("="*60 + "\n")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
