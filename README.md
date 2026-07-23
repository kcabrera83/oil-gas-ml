# Oil & Gas ML - Machine Learning System for Crude Oil Evaluation

Comprehensive Machine Learning system for the study, classification, and economic evaluation of crude oil based on its physical-chemical properties. Includes an interactive web server with dashboard and prediction form.

## Features

- **Quality Classification**: Automatically determines the crude oil quality class (premium, standard, inferior, dehydrated)
- **Value Prediction**: Estimates the market value of crude oil in USD/barrel
- **Multi-output**: Simultaneous prediction of yield and value
- **Exploratory Analysis**: Complete visualization of distributions, correlations, and profiles
- **Multiple Models**: Random Forest, Gradient Boosting, SVM, KNN, MLP, Ridge, ElasticNet

## Project Structure

```
oil-gas-ml/
├── oil_gas_ml/              # Main package
│   ├── __init__.py
│   ├── data_generator.py    # Synthetic data generator
│   ├── models/              # ML models
│   │   ├── crude_classifier.py
│   │   ├── crude_regressor.py
│   │   └── quality_predictor.py
│   └── utils/               # Utilities
│       ├── preprocessor.py
│       ├── visualizer.py
│       └── metrics.py
├── scripts/                 # Executable scripts
│   ├── train.py             # Full training
│   ├── predict.py           # Sample prediction
│   └── evaluate.py          # System evaluation
├── data/                    # Data (generated)
├── outputs/                 # Results
│   ├── models/              # Trained models
│   └── plots/               # Plots
└── requirements.txt
```

## Crude Oil Properties Analyzed

| Property | Unit | Description |
|----------|------|-------------|
| API Gravity | °API | Relative density of crude oil |
| Viscosity | cP | Resistance to flow |
| Sulfur | % | Total sulfur content |
| Water (BS&W) | % | Water content |
| Asphaltenes | % | Asphaltene content |
| TAN | mg KOH/g | Total acid number |
| Pour Point | °C | Minimum fluidity temperature |
| Flash Point | °C | Ignitability temperature |
| Density | kg/m³ | Absolute density |
| RVP | kPa | Reid vapor pressure |
| Salinity | PTB | Salt content |
| Metals | ppm | Total metal content |
| Nitrogen | % | Nitrogen content |
| Carbon Residue | % | Residual carbon |

## Installation

```bash
git clone https://github.com/kcabrera83/oil-gas-ml.git
cd oil-gas-ml
pip install -r requirements.txt
```

## Usage

### Train all models

```bash
python scripts/train.py
```

### Predict sample quality

```bash
python scripts/predict.py
```

### Full system evaluation

```bash
python scripts/evaluate.py
```

### Web Server (Dashboard + Predictor)

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser to access the interactive dashboard with:
- Dataset statistics
- Distribution charts by type and quality
- Prediction form with 15 properties
- Quick prediction by crude oil type
- Results with quality class, market value, and yield

### Use as a library

```python
from oil_gas_ml.data_generator import CrudeDataGenerator
from oil_gas_ml.utils.preprocessor import CrudePreprocessor
from oil_gas_ml.models.crude_classifier import CrudeClassifier

# Generate data
gen = CrudeDataGenerator(seed=42)
df = gen.generate(n_samples=2000)

# Preprocess
preprocessor = CrudePreprocessor()
X_train, X_test, y_train, y_test, le = preprocessor.prepare_classification(df)

# Train
clf = CrudeClassifier(model_name="gradient_boosting")
clf.train(X_train, y_train, class_names=le.classes_)

# Predict
predictions = clf.predict(X_test)
```

## Implemented Models

### Quality Classification
- Random Forest (200 trees)
- Gradient Boosting
- Support Vector Machine (RBF kernel)
- K-Nearest Neighbors
- Multi-Layer Perceptron

### Value Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Extra Trees Regressor
- SVR (RBF kernel)
- MLP Regressor
- Ridge Regression
- ElasticNet

### Multi-Output
- Gradient Boosting (MultiOutputRegressor) for simultaneous prediction of value and yield

## Supported Crude Oil Types

| Type | °API | Viscosity | Sulfur |
|------|------|-----------|--------|
| Light | 35-55 | 1-10 cP | 0.1-0.5% |
| Medium | 25-35 | 10-100 cP | 0.5-1.5% |
| Heavy | 10-25 | 100-10,000 cP | 1.5-3.5% |
| Extra Heavy | 5-10 | 1,000-100,000 cP | 3.0-6.0% |

## Author

Kelvin Cabrera - 2026
