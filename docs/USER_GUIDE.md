# User Guide - Crude Oil Evaluation

## Overview
Machine Learning system for the classification, evaluation, and economic valuation of crude oil based on its physical-chemical properties. Includes an interactive dashboard with dataset statistics and real-time prediction capabilities.

## Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
git clone https://github.com/kcabrera83/oil-gas-ml.git
cd oil-gas-ml
pip install -r requirements.txt
```

### Training Models
```bash
python scripts/train.py
```

### Starting the Server
```bash
python app.py
```
Open http://localhost:5001 in your browser.

## Dashboard Features
- **Dataset Statistics**: Total samples, crude type distribution, quality class breakdown
- **Distribution Charts**: Interactive histograms for each physical-chemical property
- **Correlation Matrix**: Heatmap showing feature relationships
- **Prediction Form**: Enter 15 crude oil properties to get quality classification, market value, and yield recovery
- **Quick Prediction**: Pre-filled profiles for Light, Medium, Heavy, and Extra Heavy crude types

## Crude Oil Properties
| Property | Unit | Description |
|----------|------|-------------|
| API Gravity | deg API | Relative density |
| Viscosity | cP | Resistance to flow |
| Sulfur | % | Sulfur content |
| Water (BS&W) | % | Water content |
| Asphaltenes | % | Asphaltene content |
| TAN | mg KOH/g | Total acid number |
| Pour Point | deg C | Minimum fluidity temperature |
| Flash Point | deg C | Ignitability temperature |
| Density | kg/m3 | Absolute density |
| RVP | kPa | Reid vapor pressure |
| Salinity | PTB | Salt content |
| Metals | ppm | Total metal content |
| Nitrogen | % | Nitrogen content |
| Carbon Residue | % | Residual carbon |

## API Usage

### Using curl
```bash
# Predict crude oil quality and value
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "api_gravity": 35.0,
    "sulfur_content_pct": 0.5,
    "viscosity_cp": 15.0,
    "water_content_pct": 1.0,
    "asphaltenes_pct": 2.0,
    "tan_mg_koh_g": 0.5,
    "pour_point_c": 10.0,
    "flash_point_c": 45.0,
    "density_kg_m3": 850.0,
    "rvp_kpa": 60.0,
    "salinity_ptb": 10.0,
    "metals_ppm": 50.0,
    "nitrogen_pct": 0.1,
    "carbon_residue_pct": 1.5,
    "crude_type": "light"
  }'

# Get dataset statistics
curl http://localhost:5001/api/stats

# Get feature distribution
curl http://localhost:5001/api/distribution/api_gravity

# Get correlation matrix
curl http://localhost:5001/api/correlation

# Get model information
curl http://localhost:5001/api/model_info
```

### Using Python
```python
import requests

# Predict crude oil quality and value
response = requests.post("http://localhost:5001/predict", json={
    "api_gravity": 35.0,
    "sulfur_content_pct": 0.5,
    "viscosity_cp": 15.0,
    "water_content_pct": 1.0,
    "asphaltenes_pct": 2.0,
    "tan_mg_koh_g": 0.5,
    "pour_point_c": 10.0,
    "flash_point_c": 45.0,
    "density_kg_m3": 850.0,
    "rvp_kpa": 60.0,
    "salinity_ptb": 10.0,
    "metals_ppm": 50.0,
    "nitrogen_pct": 0.1,
    "carbon_residue_pct": 1.5,
    "crude_type": "light"
})
result = response.json()
print(f"Quality: {result['quality_class']}")
print(f"Market Value: ${result['market_value']}/bbl")
print(f"Yield Recovery: {result['yield_recovery']}%")
```

### Using as a Library
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

## Supported Crude Oil Types
| Type | deg API | Viscosity | Sulfur |
|------|---------|-----------|--------|
| Light | 35-55 | 1-10 cP | 0.1-0.5% |
| Medium | 25-35 | 10-100 cP | 0.5-1.5% |
| Heavy | 10-25 | 100-10,000 cP | 1.5-3.5% |
| Extra Heavy | 5-10 | 1,000-100,000 cP | 3.0-6.0% |

## Quality Classes
- **Premium**: High API, low sulfur, low viscosity
- **Standard**: Moderate properties
- **Inferior**: High sulfur or high viscosity
- **Dehydrated**: Requires additional processing
