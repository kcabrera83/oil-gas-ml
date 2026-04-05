"""
Script principal para ejecutar el pipeline completo de Machine Learning.
Integra preprocesamiento, entrenamiento y generación de reportes visuales.
Aplicado a la prediccion de variables en la industria de Petroleo y Gas.
"""

import logging
import sys
import os

# ============================================
# CONFIGURACIÓN DE RUTAS PARA IMPORTS LOCALES
# Compatible con Windows, Linux y Mac
# ============================================
ruta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_src = os.path.join(ruta_actual, "src")
sys.path.insert(0, ruta_src)

import pandas as pd
import numpy as np

from data_preprocessing import cargar_datos, preprocesar_pipeline
from model_training import preparar_datos, entrenar_modelo, evaluar_modelo, guardar_modelo
from visualizaciones import (
    graficar_correlaciones,
    graficar_importancia_modelo,
    graficar_reales_vs_predichos,
)

# Configuración de logging profesional
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constantes centralizadas
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
    """Carga datos desde CSV y valida la estructura mínima requerida."""
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


def ejecutar_pipeline(ruta_datos: str = RUTA_DATOS, ruta_guardado: str = RUTA_MODELO) -> dict[str, any]:
    """
    Orquesta el flujo completo: Carga -> EDA -> Preprocesamiento -> Entrenamiento -> Reportes -> Persistencia.
    """
    logger.info("=== Iniciando Pipeline ML - Petróleo y Gas ===")
    
    # 1. Ingesta y EDA Inicial
    df = cargar_o_validar_datos(ruta_datos)
    
    # Generar reporte de correlaciones
    logger.info("Generando mapa de calor de correlaciones...")
    ruta_corr = graficar_correlaciones(df, titulo="Correlaciones - Dataset Original")
    logger.info("Gráfico guardado en: %s", ruta_corr)

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
    
    # Generar reporte de importancia de características
    logger.info("Generando gráfico de importancia del modelo...")
    ruta_imp = graficar_importancia_modelo(modelo, COLUMNAS_PREDICTORAS)
    logger.info("Gráfico guardado en: %s", ruta_imp)
    
    # 5. Evaluation & Prediction
    logger.info("Calculando métricas y predicciones...")
    mse, r2 = evaluar_modelo(modelo, X_test, y_test)
    y_pred = modelo.predict(X_test)
    
    logger.info("Resultados -> MSE: %.2f | R²: %.4f", mse, r2)
    
    # Generar reporte de Reales vs Predichos
    logger.info("Generando gráfico de Reales vs Predichos...")
    ruta_pred = graficar_reales_vs_predichos(y_test, y_pred)
    logger.info("Gráfico guardado en: %s", ruta_pred)
    
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
