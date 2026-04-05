"""
Módulo de preprocesamiento de datos para proyectos de petróleo y gas.
Incluye funciones para limpieza, transformación y preparación de datos.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import Tuple, List, Optional


def cargar_datos(ruta_archivo):
    """
    Carga un archivo CSV con configuracion explicita para evitar errores de localizacion.
    """
    return pd.read_csv(ruta_archivo, sep=',', decimal='.')


def limpiar_datos(df: pd.DataFrame, columnas_numericas: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Limpieza básica de datos:
    - Elimina duplicados
    - Maneja valores nulos mediante imputación
    
    Args:
        df: DataFrame de entrada.
        columnas_numericas: Lista de columnas numéricas a imputar.
        
    Returns:
        DataFrame limpio.
    """
    # Eliminar duplicados
    df = df.drop_duplicates()
    
    # Imputar valores nulos en columnas numéricas
    if columnas_numericas:
        imputer = SimpleImputer(strategy='median')
        df[columnas_numericas] = imputer.fit_transform(df[columnas_numericas])
    
    return df


def normalizar_datos(df: pd.DataFrame, columnas: List[str]) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Normaliza las columnas especificadas usando StandardScaler.
    
    Args:
        df: DataFrame de entrada.
        columnas: Lista de columnas a normalizar.
        
    Returns:
        Tuple con DataFrame normalizado y el scaler entrenado.
    """
    scaler = StandardScaler()
    df[columnas] = scaler.fit_transform(df[columnas])
    return df, scaler


def preprocesar_pipeline(df: pd.DataFrame, columnas_numericas: List[str]) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Pipeline completo de preprocesamiento:
    1. Limpieza de datos
    2. Normalización
    
    Args:
        df: DataFrame de entrada.
        columnas_numericas: Lista de columnas numéricas.
        
    Returns:
        Tuple con DataFrame procesado y el scaler entrenado.
    """
    df = limpiar_datos(df, columnas_numericas)
    df, scaler = normalizar_datos(df, columnas_numericas)
    return df, scaler
