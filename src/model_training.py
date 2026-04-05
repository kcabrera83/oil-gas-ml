"""
Script principal para ejecutar el pipeline de ML para predicción de producción.
"""

import logging
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Importar funciones del módulo
from ml_module import (
    preparar_datos,
    entrenar_modelo,
    evaluar_modelo,
    evaluar_modelo_completo,
    log_metricas,
    guardar_modelo,
    cargar_modelo
)

# Configurar logging independiente para main
logger = logging.getLogger(__name__)


def cargar_datos(ruta_datos: str, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Carga y prepara los datos desde un archivo CSV.
    
    Args:
        ruta_datos: Path al archivo CSV con los datos.
        target_column: Nombre de la columna objetivo.
        
    Returns:
        Tupla con X (features) e y (target).
    """
    if not Path(ruta_datos).exists():
        raise FileNotFoundError(f"Archivo de datos no encontrado: {ruta_datos}")
    
    df = pd.read_csv(ruta_datos)
    logger.info("Datos cargados: %d filas, %d columnas", len(df), len(df.columns))
    
    if target_column not in df.columns:
        raise ValueError(f"Columna objetivo '{target_column}' no encontrada en los datos")
    
    # Separar features y target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Eliminar filas con valores nulos
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]
    
    logger.info("Datos limpios: %d muestras válidas", len(X))
    return X, y


def generar_datos_simulados(n_samples: int = 500, n_features: int = 10) -> tuple[pd.DataFrame, pd.Series]:
    """
    Genera datos sintéticos para pruebas del pipeline.
    
    Args:
        n_samples: Número de muestras a generar.
        n_features: Número de variables predictoras.
        
    Returns:
        Tupla con X e y simulados.
    """
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    # Target con relación lineal + ruido
    coeficientes = np.random.randn(n_features)
    y = pd.Series(X @ coeficientes + np.random.normal(0, 0.5, n_samples), name="produccion")
    logger.info("Datos simulados generados: %d muestras, %d features", n_samples, n_features)
    return X, y


def ejecutar_pipeline(
    ruta_datos: str = None,
    target_column: str = "produccion",
    test_size: float = 0.2,
    n_estimators: int = 100,
    guardar: bool = True,
    ruta_modelo: str = "models/modelo_produccion.pkl"
) -> dict:
    """
    Ejecuta el pipeline completo de ML: carga, entrenamiento, evaluación y persistencia.
    
    Args:
        ruta_datos: Path a los datos (None para usar datos simulados).
        target_column: Nombre de la variable objetivo.
        test_size: Proporción para conjunto de prueba.
        n_estimators: Número de árboles para Random Forest.
        guardar: Si True, guarda el modelo en disco.
        ruta_modelo: Path para guardar/cargar el modelo.
        
    Returns:
        Diccionario con resultados y métricas del pipeline.
    """
    resultados = {}
    
    try:
        # Paso 1: Cargar o generar datos
        if ruta_datos and Path(ruta_datos).exists():
            logger.info("Cargando datos desde: %s", ruta_datos)
            X, y = cargar_datos(ruta_datos, target_column)
        else:
            logger.info("Usando datos simulados para demostración")
            X, y = generar_datos_simulados()
        
        # Paso 2: Dividir datos
        X_train, X_test, y_train, y_test = preparar_datos(X, y, test_size=test_size)
        resultados["tamano_datos"] = {"train": len(X_train), "test": len(X_test)}
        
        # Paso 3: Entrenar modelo
        modelo = entrenar_modelo(X_train, y_train, n_estimators=n_estimators)
        resultados["modelo"] = {"tipo": "RandomForestRegressor", "n_estimators": n_estimators}
        
        # Paso 4: Evaluar modelo
        metricas_basicas = evaluar_modelo(modelo, X_test, y_test)
        metricas_completas = evaluar_modelo_completo(modelo, X_train, X_test, y_train, y_test)
        
        resultados["metricas"] = {**metricas_basicas, **metricas_completas}
        
        # Registrar métricas en consola
        log_metricas(metricas_completas, prefijo="Resultados finales")
        
        # Paso 5: Guardar modelo (opcional)
        if guardar:
            exito_guardado = guardar_modelo(modelo, ruta_modelo)
            resultados["guardado"] = exito_guardado
        
        # Paso 6: Verificar carga del modelo guardado
        if guardar and resultados.get("guardado"):
            modelo_cargado = cargar_modelo(ruta_modelo)
            # Verificación rápida
            pred_original = modelo.predict(X_test.iloc[:5])
            pred_cargado = modelo_cargado.predict(X_test.iloc[:5])
            if np.allclose(pred_original, pred_cargado):
                logger.info("Verificación: modelo cargado produce predicciones consistentes")
            else:
                logger.warning("Advertencia: discrepancia en predicciones tras cargar modelo")
        
        logger.info("Pipeline ejecutado exitosamente")
        return resultados
        
    except Exception as e:
        logger.error("Error crítico en el pipeline: %s", e, exc_info=True)
        resultados["error"] = str(e)
        return resultados


def main():
    """Función principal de entrada."""
    logger.info("=== Iniciando pipeline de ML para predicción de producción ===")
    
    # Configuración de parámetros (personalizar según necesidad)
    config = {
        "ruta_datos": None,  # Ej: "data/produccion_yacimientos.csv"
        "target_column": "produccion",
        "test_size": 0.2,
        "n_estimators": 100,
        "guardar": True,
        "ruta_modelo": "models/modelo_produccion.pkl"
    }
    
    # Ejecutar pipeline
    resultados = ejecutar_pipeline(**config)
    
    # Resumen final
    if "error" not in resultados:
        logger.info("=== Resumen ===")
        logger.info("Muestras entrenamiento: %d", resultados["tamano_datos"]["train"])
        logger.info("Muestras prueba: %d", resultados["tamano_datos"]["test"])
        logger.info("R² en prueba: %.4f", resultados["metricas"]["r2_test"])
        logger.info("R² ajustado: %.4f", resultados["metricas"]["r2_ajustado"])
        logger.info("RMSE: %.4f", resultados["metricas"]["rmse"])
        if resultados.get("guardado"):
            logger.info("Modelo guardado en: %s", config["ruta_modelo"])
    else:
        logger.error("Pipeline falló: %s", resultados["error"])
        sys.exit(1)
    
    logger.info("=== Pipeline finalizado ===")


if __name__ == "__main__":
    main()
