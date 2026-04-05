"""
main.py - Pipeline completo de entrenamiento y evaluación.
"""

import logging
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend no interactivo para servidores/consola
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from model_training import (
    preparar_datos, normalizar_datos, entrenar_modelo,
    evaluar_modelo_completo, guardar_modelo
)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(module)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def generar_datos_sinteticos(n: int = 100) -> pd.DataFrame:
    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(n, 6), columns=[f"var_{i}" for i in range(6)])
    df["produccion"] = df["var_0"] * 2.5 + df["var_2"] * -1.8 + np.random.normal(0, 0.5, n)
    return df


def plot_correlaciones(df: pd.DataFrame, ruta: str) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Mapa de Calor - Correlaciones")
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()
    logger.info("Gráfico guardado en: %s", ruta)


def plot_importancia(modelo, feature_names: list, ruta: str) -> None:
    importancias = modelo.feature_importances_
    indices = np.argsort(importancias)[::-1]
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(importancias)), importancias[indices], align="center")
    plt.xticks(range(len(importancias)), [feature_names[i] for i in indices], rotation=45)
    plt.title("Importancia de Variables")
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()
    logger.info("Gráfico guardado en: %s", ruta)


def main():
    logger.info("=== Iniciando Pipeline ML - Petróleo y Gas ===")
    os.makedirs("results", exist_ok=True)
    
    # 1. Carga de datos
    ruta_csv = "data/produccion_pozos.csv"
    if Path(ruta_csv).exists():
        logger.info("Cargando dataset desde: %s", ruta_csv)
        df = pd.read_csv(ruta_csv)
    else:
        logger.warning("Archivo no encontrado. Fallback a datos sintéticos.")
        df = generar_datos_sinteticos()
    
    target = "produccion"
    X = df.drop(columns=[target])
    y = df[target]
    
    # 2. Correlaciones
    logger.info("Generando mapa de calor de correlaciones...")
    plot_correlaciones(df, "results/correlaciones.png")
    
    # 3. Preprocesamiento
    logger.info("Aplicando preprocesamiento y normalización...")
    X_train, X_test, y_train, y_test = preparar_datos(X, y, test_size=0.2)
    X_train, X_test, _ = normalizar_datos(X_train, X_test)
    logger.info("Distribución -> Train: %d | Test: %d", len(X_train), len(X_test))
    
    # 4. Entrenamiento
    logger.info("Entrenando Random Forest Regressor...")
    modelo = entrenar_modelo(X_train, y_train, n_estimators=100)
    
    # 5. Importancia
    logger.info("Generando gráfico de importancia del modelo...")
    plot_importancia(modelo, list(X.columns), "results/importancia_modelo.png")
    
    # 6. Evaluación (CORRECCIÓN: acceso por diccionario, no desempaquetado)
    logger.info("Calculando métricas y predicciones...")
    metricas = evaluar_modelo_completo(modelo, X_train, X_test, y_train, y_test)
    
    # Acceso seguro por claves
    mse_val = metricas["mse"]
    r2_val = metricas["r2_test"]
    logger.info("Resultados -> MSE: %.2f | R²: %.4f", mse_val, r2_val)
    logger.info("Métricas completas: %s", {k: round(v, 4) for k, v in metricas.items()})
    
    # 7. Guardar modelo
    ruta_modelo = "models/modelo_produccion.pkl"
    if guardar_modelo(modelo, ruta_modelo):
        logger.info("Pipeline finalizado exitosamente. Modelo listo para predict.py")
    else:
        logger.error("Fallo al persistir el modelo. predict.py no podrá cargarlo.")
        sys.exit(1)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
