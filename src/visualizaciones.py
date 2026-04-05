"""
Módulo de visualizaciones para generar reportes gráficos.
Guarda las imágenes en la carpeta 'results/'.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List


# Configuración global de estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
RESULTS_DIR = Path(__file__).parent.parent / "results"


def asegurar_directorio() -> None:
    """Crea la carpeta de resultados si no existe."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def graficar_correlaciones(df: pd.DataFrame, titulo: str = "Matriz de Correlación") -> str:
    """
    Genera un mapa de calor de correlaciones y lo guarda.
    Returns: Ruta del archivo guardado.
    """
    asegurar_directorio()
    ruta_salida = RESULTS_DIR / "correlaciones.png"
    
    plt.figure(figsize=(10, 8))
    # Seleccionar solo columnas numéricas
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title(titulo)
    plt.tight_layout()
    plt.savefig(ruta_salida)
    plt.close()
    return str(ruta_salida)


def graficar_importancia_modelo(modelo, nombres_columnas: List[str], titulo: str = "Importancia de Características") -> str:
    """
    Grafica la importancia de las características de un modelo basado en árboles.
    """
    asegurar_directorio()
    ruta_salida = RESULTS_DIR / "importancia_modelo.png"
    
    importances = modelo.feature_importances_
    indices = np.argsort(importances)
    
    plt.figure(figsize=(10, 6))
    plt.title(titulo)
    plt.barh(range(len(indices)), importances[indices], color="b", align="center")
    plt.yticks(range(len(indices)), [nombres_columnas[i] for i in indices])
    plt.xlabel("Importancia Relativa")
    plt.tight_layout()
    plt.savefig(ruta_salida)
    plt.close()
    return str(ruta_salida)


def graficar_reales_vs_predichos(y_true: np.ndarray, y_pred: np.ndarray, titulo: str = "Reales vs Predichos") -> str:
    """
    Scatter plot comparando valores reales contra predichos.
    """
    asegurar_directorio()
    ruta_salida = RESULTS_DIR / "reales_vs_predichos.png"
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    # Línea ideal (y=x)
    limites = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    plt.plot(limites, limites, "r--", label="Predicción Perfecta")
    
    plt.xlabel("Valores Reales")
    plt.ylabel("Valores Predichos")
    plt.title(titulo)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ruta_salida)
    plt.close()
    return str(ruta_salida)
