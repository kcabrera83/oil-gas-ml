"""
Script de inferencia (predicción).
Carga el modelo entrenado y predice la producción para nuevos datos de pozos.
"""

import joblib
import pandas as pd
import numpy as np
import os

def cargar_modelo(ruta_modelo: str = "models/modelo_produccion.pkl"):
    """
    Carga el modelo serializado desde disco.
    """
    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(f"No se encontró el modelo en {ruta_modelo}. Ejecuta main.py primero.")
    
    modelo = joblib.load(ruta_modelo)
    print(f" Modelo cargado exitosamente desde: {ruta_modelo}")
    return modelo

def predecir_produccion(modelo, presion: float, temperatura: float, porosidad: float, permeabilidad: float):
    """
    Realiza una predicción individual.
    """
    # Crear DataFrame con el nuevo pozo (mismo formato que el entrenamiento)
    nuevo_pozo = pd.DataFrame({
        'presion_psi': [presion],
        'temperatura_c': [temperatura],
        'porosidad': [porosidad],
        'permeabilidad_md': [permeabilidad]
    })
    
    # Realizar predicción
    prediccion = modelo.predict(nuevo_pozo)
    return prediccion[0]

def main():
    print("--- SISTEMA DE PREDICCIÓN DE PRODUCCIÓN (OIL & GAS ML) ---")
    
    try:
        # 1. Cargar modelo
        modelo = cargar_modelo()
        
        # 2. Datos de ejemplo (puedes cambiar estos valores)
        # Caso: Un pozo con alta presión y buena permeabilidad
        datos_ejemplo = {
            "presion_psi": 3500,
            "temperatura_c": 85,
            "porosidad": 0.25,
            "permeabilidad_md": 150
        }
        
        print("\n Datos de entrada del nuevo pozo:")
        for k, v in datos_ejemplo.items():
            print(f"   {k}: {v}")
            
        # 3. Predecir
        resultado = predecir_produccion(
            modelo,
            datos_ejemplo["presion_psi"],
            datos_ejemplo["temperatura_c"],
            datos_ejemplo["porosidad"],
            datos_ejemplo["permeabilidad_md"]
        )
        
        print("\n" + "="*50)
        print(f" PRODUCCIÓN ESTIMADA: {resultado:.2f} BPD (Barriles por Día)")
        print("="*50)
        
    except Exception as e:
        print(f" Error durante la predicción: {e}")

if __name__ == "__main__":
    main()
