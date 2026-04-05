"""
Script principal para ejecutar el pipeline completo de Machine Learning.
Integra preprocesamiento y entrenamiento de modelos aplicados a la
industria de petróleo y gas.
"""

import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

# Configuración de rutas para imports locales
SCRIPT_DIR = Path(__file__).parent.resolve()
SRC_DIR = SCRIPT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from data_preprocessing import cargar_datos, preprocesar_pipeline
from model_training import (
    preparar_datos,
    entrenar_modelo,
    evaluar_modelo,
    guardar_modelo,
)

# Configuración de logging profesional
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constantes centralizadas (Single Source of Truth)
RUTA_DATOS = "data/produccion_pozos.csv"
RUTA_MODELO = "models/modelo_produccion.pkl"
COLUMNAS_PREDICTORAS = ["presion_psi", "temperatura_c", "porosidad", "permeabilidad_md"]
COLUMNA_OBJETIVO = "produccion_bpd"
NUM_MUESTRAS_EJEMPLO = 100


def generar_datos_sinteticos(n_muestras: int = NUM_MUESTRAS_EJEMPLO) -> pd.DataFrame:
    """Genera un DataFrame sintético estructurado para pruebas de desarrollo."""
    return pd.DataFrame({
        "pozo_id": range(1, n_muestras + 1),
        "presion_psi": np.random.normal(3000, 500, n_muestras),
        "temperatura_c": np.random.normal(80, 15, n_muestras),
        "porosidad": np.random.uniform(0.1, 0.3, n_muestras),
        "permeabilidad_md": np.random.exponential(100, n_muestras),
        COLUMNA_OBJETIVO: np.random.normal(500, 150, n_muestras),
    })


def cargar_o_validar_datos(ruta: str) -> pd.DataFrame:
    """
    Carga datos desde CSV y valida la estructura mínima requerida.
    Si el archivo no existe, genera datos sintéticos para no detener el desarrollo.
    """
    try:
        logger.info("Cargando dataset desde: %s", ruta)
        df = cargar_datos(ruta)
        
        columnas_requeridas = set(COLUMNAS_PREDICTORAS + [COLUMNA_OBJETIVO])
        if not columnas_requeridas.issubset(df.columns):
            raise ValueError(f"Dataset incompleto. Faltan: {columnas_requeridas - set(df.columns)}")
        return df
        
    except FileNotFoundError:
        logger.warning("Archivo no encontrado. Fallback a datos sintéticos.")
        return generar_datos_sinteticos()
    except Exception as e:
        logger.error("Error crítico al cargar datos: %s", e)
        raise


def ejecutar_pipeline(ruta_datos: str = RUTA_DATOS, ruta_guardado: str = RUTA_MODELO) -> dict[str, Any]:
    """
    Orquesta el flujo completo de ML: Carga -> Preprocesamiento -> Entrenamiento -> Evaluación -> Persistencia.
    
    Args:
        ruta_datos: Path al archivo CSV de entrada.
        ruta_guardado: Path destino para el modelo serializado.
        
    Returns:
        Diccionario con las métricas de evaluación (MSE y R²).
    """
    logger.info("=== Iniciando Pipeline ML - Petróleo y Gas ===")
    
    # 1. Ingesta
    df = cargar_o_validar_datos(ruta_datos)
    X = df[COLUMNAS_PREDICTORAS]
    y = df[COLUMNA_OBJETIVO]
    
    # 2. Transformación
    logger.info("Aplicando preprocesamiento y normalización...")
    X_proc, scaler = preprocesar_pipeline(X, COLUMNAS_PREDICTORAS)
    
    # 3. Split
    logger.info("Dividiendo train/test (80/20)...")
    X_train, X_test, y_train, y_test = preparar_datos(X_proc, y)
    logger.info("Distribución -> Train: %d | Test: %d", len(X_train), len(X_test))
    
    # 4. Training
    logger.info("Entrenando Random Forest Regressor...")
    modelo = entrenar_modelo(X_train, y_train)
    
    # 5. Evaluation
    logger.info("Calculando métricas de rendimiento...")
    mse, r2 = evaluar_modelo(modelo, X_test, y_test)
    logger.info("Resultados -> MSE: %.2f | R²: %.4f", mse, r2)
    
    # 6. Persistence
    logger.info("Serializando modelo en: %s", ruta_guardado)
    guardar_modelo(modelo, ruta_guardado)
    
    logger.info("=== Pipeline finalizado exitosamente ===")
    return {"mse": mse, "r2": r2}


def main() -> None:
    """Punto de entrada con manejo de excepciones de nivel superior."""
    try:
        resultados = ejecutar_pipeline()
        logger.info("Ejecución completada. Métricas: %s", resultados)
    except Exception as err:
        logger.critical("Pipeline abortado por error: %s", err)
        sys.exit(1)


if __name__ == "__main__":
    main()
