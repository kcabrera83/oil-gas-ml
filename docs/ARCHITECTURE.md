# Architecture - Crude Oil Evaluation

## System Overview
```
                    +-------------------+
                    |   Flask Server    |
                    |   (app.py)        |
                    |   Port 5001       |
                    +--------+----------+
                             |
              +--------------+--------------+
              |              |              |
     +--------v---+  +------v-----+  +-----v--------+
     | Classifier  |  | Regressor  |  | Multi-Output |
     | (Quality)   |  | (Value)    |  | Predictor    |
     +------+------+  +------+-----+  +------+-------+
            |                |               |
     +------v----------------v---------------v-------+
     |              CrudePreprocessor                 |
     |         (RobustScaler + Encoding)              |
     +------------------------+----------------------+
                              |
                    +---------v----------+
                    |  Synthetic Dataset  |
                    |  (3000 samples)     |
                    +--------------------+
```

## Components

### Data Layer
- **Data Source**: Synthetic data generator (`CrudeDataGenerator`) producing 3000 samples with 15+ physical-chemical properties
- **Properties**: API gravity, sulfur, viscosity, water content, asphaltenes, TAN, pour point, flash point, density, RVP, salinity, metals, nitrogen, carbon residue
- **Preprocessing**: RobustScaler normalization, label encoding for categorical features, train/test split (80/20)

### Model Layer

#### Quality Classifier
- **Algorithm**: Gradient Boosting (best) among Random Forest, SVM, KNN, MLP
- **Target**: Quality class (premium, standard, inferior, dehydrated)
- **Metrics**: Accuracy, F1-score, Precision, Recall
- **Output**: Class probabilities for all 4 categories

#### Market Value Regressor
- **Algorithm**: Gradient Boosting (best) among Random Forest, Extra Trees, SVR, MLP, Ridge, ElasticNet
- **Target**: Market value (USD/barrel)
- **Metrics**: R2, RMSE, MAE, MAPE

#### Multi-Output Quality Predictor
- **Algorithm**: MultiOutputRegressor wrapping Gradient Boosting
- **Targets**: Simultaneous prediction of yield recovery (%) and market value (USD/barrel)
- **Metrics**: Per-target R2 and RMSE

### API Layer
- **Framework**: Flask with CORS enabled
- **Endpoints**: 8 REST endpoints (predict, stats, distribution, correlation, sample, model_info, docs, dashboard)
- **Serialization**: JSON request/response

### Dashboard Layer
- **Frontend**: HTML/CSS/JavaScript with Jinja2 templates
- **Charts**: Interactive distribution plots, correlation heatmaps, prediction results
- **Forms**: 15-field prediction form with quick-fill profiles

## Data Flow

1. **Input**: User submits crude oil properties via form or API
2. **Preprocessing**: `CrudePreprocessor` applies RobustScaler and encodes categorical features
3. **Classification**: `CrudeClassifier` predicts quality class and probabilities
4. **Regression**: `CrudeRegressor` estimates market value in USD/barrel
5. **Multi-Output**: `QualityPredictor` simultaneously predicts yield recovery and market value
6. **Response**: Combined results returned as JSON or rendered in dashboard

## Training Pipeline
1. Generate synthetic dataset (3000 samples)
2. Exploratory data visualization
3. Preprocess with RobustScaler
4. Train 5 classifiers, select best by F1-score
5. Train 7 regressors, select best by R2
6. Train multi-output predictor
7. Save best models to `outputs/models/`
8. Generate evaluation plots and confusion matrices

## File Structure
```
oil-gas-ml/
├── oil_gas_ml/
│   ├── data_generator.py       # Synthetic data generation
│   ├── models/
│   │   ├── crude_classifier.py # Quality classification
│   │   ├── crude_regressor.py  # Market value regression
│   │   └── quality_predictor.py# Multi-output prediction
│   └── utils/
│       ├── preprocessor.py     # Data preprocessing
│       ├── visualizer.py       # Plot generation
│       └── metrics.py          # Model evaluation
├── scripts/
│   ├── train.py                # Training pipeline
│   ├── predict.py              # Sample predictions
│   └── evaluate.py             # Full evaluation
├── data/                       # Generated datasets
├── outputs/
│   ├── models/                 # Trained model files (.pkl)
│   └── plots/                  # Generated visualizations
├── app.py                      # Flask web server
└── templates/                  # HTML templates
```
