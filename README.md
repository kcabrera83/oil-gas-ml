# Oil & Gas ML - Sistema de Machine Learning para Evaluación de Crudo

Sistema integral de Machine Learning para el estudio, clasificación y evaluación económica de crudo petrolífero basado en sus propiedades físico-químicas. Incluye servidor web interactivo con dashboard y formulario de predicción.

## Características

- **Clasificación de calidad**: Determina automáticamente la clase de calidad del crudo (premium, estándar, inferior, deshidratado)
- **Predicción de valor**: Estima el valor de mercado del crudo en USD/barril
- **Multi-output**: Predicción simultánea de rendimiento y valor
- **Análisis exploratorio**: Visualización completa de distribuciones, correlaciones y perfiles
- **Múltiples modelos**: Random Forest, Gradient Boosting, SVM, KNN, MLP, Ridge, ElasticNet

## Estructura del Proyecto

```
oil-gas-ml/
├── oil_gas_ml/              # Paquete principal
│   ├── __init__.py
│   ├── data_generator.py    # Generador de datos sintéticos
│   ├── models/              # Modelos de ML
│   │   ├── crude_classifier.py
│   │   ├── crude_regressor.py
│   │   └── quality_predictor.py
│   └── utils/               # Utilidades
│       ├── preprocessor.py
│       ├── visualizer.py
│       └── metrics.py
├── scripts/                 # Scripts ejecutables
│   ├── train.py             # Entrenamiento completo
│   ├── predict.py           # Predicción de muestras
│   └── evaluate.py          # Evaluación del sistema
├── data/                    # Datos (generados)
├── outputs/                 # Resultados
│   ├── models/              # Modelos entrenados
│   └── plots/               # Gráficos
└── requirements.txt
```

## Propiedades del Crudo Analizadas

| Propiedad | Unidad | Descripción |
|-----------|--------|-------------|
| API Gravity | °API | Densidad relativa del crudo |
| Viscosidad | cP | Resistencia al flujo |
| Azufre | % | Contenido de azufre total |
| Agua (BS&W) | % | Contenido de agua |
| Asfaltenos | % | Contenido de asfaltenos |
| TAN | mg KOH/g | Número de acidez total |
| Punto de Fluidez | °C | Temperatura mínima de fluidez |
| Punto de Inflamación | °C | Temperatura de inflamabilidad |
| Densidad | kg/m³ | Densidad absoluta |
| RVP | kPa | Presión de vapor Reid |
| Salinidad | PTB | Contenido de sales |
| Metales | ppm | Contenido metálico total |
| Nitrógeno | % | Contenido de nitrógeno |
| Residuo de Carbono | % | Carbono residual |

## Instalación

```bash
git clone https://github.com/kcabrera83/oil-gas-ml.git
cd oil-gas-ml
pip install -r requirements.txt
```

## Uso

### Entrenar todos los modelos

```bash
python scripts/train.py
```

### Predecir calidad de una muestra

```bash
python scripts/predict.py
```

### Evaluación completa del sistema

```bash
python scripts/evaluate.py
```

### Servidor Web (Dashboard + Predictor)

```bash
python app.py
```

Abre http://127.0.0.1:5000 en tu navegador para acceder al dashboard interactivo con:
- Estadísticas del dataset
- Gráficos de distribución por tipo y calidad
- Formulario de predicción con 15 propiedades
- Predicción rápida por tipo de crudo
- Resultados con clase de calidad, valor de mercado y rendimiento

### Uso como librería

```python
from oil_gas_ml.data_generator import CrudeDataGenerator
from oil_gas_ml.utils.preprocessor import CrudePreprocessor
from oil_gas_ml.models.crude_classifier import CrudeClassifier

# Generar datos
gen = CrudeDataGenerator(seed=42)
df = gen.generate(n_samples=2000)

# Preprocesar
preprocessor = CrudePreprocessor()
X_train, X_test, y_train, y_test, le = preprocessor.prepare_classification(df)

# Entrenar
clf = CrudeClassifier(model_name="gradient_boosting")
clf.train(X_train, y_train, class_names=le.classes_)

# Predecir
predictions = clf.predict(X_test)
```

## Modelos Implementados

### Clasificación de Calidad
- Random Forest (200 árboles)
- Gradient Boosting
- Support Vector Machine (RBF kernel)
- K-Nearest Neighbors
- Multi-Layer Perceptron

### Regresión de Valor
- Random Forest Regressor
- Gradient Boosting Regressor
- Extra Trees Regressor
- SVR (RBF kernel)
- MLP Regressor
- Ridge Regression
- ElasticNet

### Multi-Output
- Gradient Boosting (MultiOutputRegressor) para predicción simultánea de valor y rendimiento

## Tipos de Crudo Soportados

| Tipo | °API | Viscosidad | Azufre |
|------|------|------------|--------|
| Liviano | 35-55 | 1-10 cP | 0.1-0.5% |
| Mediano | 25-35 | 10-100 cP | 0.5-1.5% |
| Pesado | 10-25 | 100-10,000 cP | 1.5-3.5% |
| Extra Pesado | 5-10 | 1,000-100,000 cP | 3.0-6.0% |

## Autor

Kelvin Cabrera - 2026
