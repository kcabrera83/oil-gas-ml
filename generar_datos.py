"""
Script para generar datos sinteticos realistas de produccion de petroleo y gas.
Guarda el resultado en: data/produccion_pozos.csv
"""

import pandas as pd
import numpy as np
import os

def generar_dataset(n_filas=1000):
    print(f"Generando {n_filas} registros de datos sinteticos...")
    
    presion_psi = np.random.normal(3000, 600, n_filas)
    temperatura_c = np.random.normal(80, 15, n_filas)
    porosidad = np.random.uniform(0.10, 0.35, n_filas)
    permeabilidad_md = np.random.exponential(100, n_filas)
    
    produccion_base = (presion_psi * porosidad * permeabilidad_md) / 5000
    ruido = np.random.normal(0, 50, n_filas)
    produccion_bpd = produccion_base + ruido + 100
    
    df = pd.DataFrame({
        'presion_psi': presion_psi,
        'temperatura_c': temperatura_c,
        'porosidad': porosidad,
        'permeabilidad_md': permeabilidad_md,
        'produccion_bpd': produccion_bpd
    })
    
    df['produccion_bpd'] = df['produccion_bpd'].clip(lower=0)
    return df

def guardar_csv(df, nombre_archivo="produccion_pozos.csv"):
    os.makedirs("data", exist_ok=True)
    ruta_completa = os.path.join("data", nombre_archivo)
    
    # Forzar formato estandar: coma como separador, punto para decimales
    df.to_csv(ruta_completa, index=False, sep=',', float_format='%.4f')
    print(f"Dataset guardado en: {ruta_completa}")

if __name__ == "__main__":
    datos = generar_dataset(n_filas=1000)
    guardar_csv(datos)
