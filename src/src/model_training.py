"""
Módulo para entrenamiento y evaluación de modelos de Machine Learning.
Aplicado a predicción de variables de yacimientos y producción.
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def preparar_datos(X, y, test_size=0.2):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    """
    return train_test_split(X, y, test_size=test_size, random_state=42)

def entrenar_modelo(X_train, y_train):
    """
    Entrena un modelo de Random Forest Regressor.
    """
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    return modelo

def evaluar_modelo(modelo, X_test, y_test):
    """
    Evalúa el modelo y retorna métricas (MSE y R2).
    """
    predicciones = modelo.predict(X_test)
    mse = mean_squared_error(y_test, predicciones)
    r2 = r2_score(y_test, predicciones)
    return mse, r2

def guardar_modelo(modelo, ruta_archivo="modelos/prediccion_produccion.pkl"):
    """
    Guarda el modelo entrenado en disco.
    """
    joblib.dump(modelo, ruta_archivo)
    print(f"Modelo guardado en {ruta_archivo}")
