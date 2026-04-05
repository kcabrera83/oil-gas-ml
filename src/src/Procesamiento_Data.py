"""
Módulo de preprocesamiento de datos para proyectos de petróleo y gas.
Incluye funciones para limpieza, transformación y preparación de datos.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def cargar_datos(ruta_archivo):
    """
    Carga un archivo CSV y retorna un DataFrame.
    """
    return pd.read_csv(ruta_archivo)


def limpiar_datos(df, columnas_numericas=None):
    """
    Limpieza básica de datos:
    - Elimina duplicados
    - Maneja valores nulos
    """
    df = df.drop_duplicates()
    
    if columnas_numericas:
        imputer = SimpleImputer(strategy='median')
        df[columnas_numericas] = imputer.fit_transform(df[columnas_numericas])
    
    return df


def normalizar_datos(df, columnas):
    """
    Normaliza las columnas especificadas usando StandardScaler.
    """
    scaler = StandardScaler()
    df[columnas] = scaler.fit_transform(df[columnas])
    return df, scaler


def preprocesar_pipeline(df, columnas_numericas):
    """
    Pipeline completo de preprocesamiento.
    """
    df = limpiar_datos(df, columnas_numericas)
    df, scaler = normalizar_datos(df, columnas_numericas)
    return df, scaler
