# API Documentation - Crude Oil Evaluation

## Base URL
```
http://localhost:5001
```

## Endpoints

### GET /
Main dashboard with interactive statistics and dataset overview.

**Response:** HTML page with embedded statistics.

---

### POST /predict
Predict crude oil quality class, market value, and yield recovery.

**Request:**
```json
{
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
}
```

**Response (200):**
```json
{
  "quality_class": "premium",
  "quality_probabilities": {
    "premium": 0.8523,
    "standard": 0.1201,
    "inferior": 0.0198,
    "dehydrated": 0.0078
  },
  "market_value": 78.50,
  "yield_recovery": 85.30,
  "multi_market_value": 77.90,
  "input_data": { ... }
}
```

**Error Response (400):**
```json
{"error": "Error message describing the issue"}
```

---

### GET /api/stats
General dataset statistics.

**Response (200):**
```json
{
  "total_samples": 3000,
  "crude_types": {"light": 750, "medium": 750, "heavy": 750, "extra_heavy": 750},
  "quality_classes": {"premium": 800, "standard": 1000, "inferior": 800, "dehydrated": 400},
  "avg_api": 32.45,
  "avg_sulfur": 1.250,
  "avg_viscosity": 156.78,
  "avg_market_value": 55.30,
  "avg_yield": 72.50
}
```

**Error Response (404):**
```json
{"error": "Dataset not available"}
```

---

### GET /api/distribution/{feature}
Get distribution data for a specific feature.

**Path Parameters:**
- `feature` (string): Feature name (e.g., `api_gravity`, `sulfur_content_pct`, `viscosity_cp`)

**Response (200):**
```json
{
  "feature": "api_gravity",
  "bins": [5.0, 7.0, 9.0, ..., 55.0],
  "counts": [12, 25, 48, ..., 15],
  "mean": 32.45,
  "std": 12.30,
  "min": 5.10,
  "max": 54.80
}
```

**Error Response (404):**
```json
{"error": "Feature not found"}
```

---

### GET /api/correlation
Correlation matrix between all numeric features.

**Response (200):**
```json
{
  "features": ["api_gravity", "sulfur_content_pct", "viscosity_cp", ...],
  "matrix": [
    [1.0000, -0.8500, -0.7200, ...],
    [-0.8500, 1.0000, 0.6800, ...],
    ...
  ]
}
```

---

### GET /api/sample/{idx}
Get a single dataset sample by index.

**Path Parameters:**
- `idx` (integer): Sample index (0-based)

**Response (200):**
```json
{
  "api_gravity": 35.0,
  "sulfur_content_pct": 0.5,
  "viscosity_cp": 15.0,
  "crude_type": "light",
  "quality_class": "premium",
  "market_value_usd_bbl": 78.50,
  ...
}
```

**Error Response (404):**
```json
{"error": "Sample not found"}
```

---

### GET /api/model_info
Information about all trained ML models.

**Response (200):**
```json
{
  "classifier": {
    "name": "gradient_boosting",
    "type": "classification",
    "classes": ["dehydrated", "inferior", "premium", "standard"]
  },
  "regressor": {
    "name": "gradient_boosting",
    "type": "regression",
    "target": "market_value_usd_bbl"
  },
  "predictor": {
    "base_model": "gradient_boosting",
    "targets": ["yield_recovery_pct", "market_value_usd_bbl"]
  }
}
```

---

### GET /api/docs
OpenAPI 3.0 self-documentation.

**Response (200):**
```json
{
  "openapi": "3.0.0",
  "info": {"title": "Oil Gas ML - Crude Oil Evaluation", "version": "1.0.0"},
  "paths": { ... }
}
```

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request - invalid input data or processing error |
| 404 | Resource not found (dataset, feature, or sample) |
