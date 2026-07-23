import pytest


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data


def test_models(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_api_docs(client):
    response = client.get("/docs")
    assert response.status_code == 200


def test_stats(client):
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_samples" in data
    assert data["total_samples"] > 0


def test_correlation(client):
    response = client.get("/api/correlation")
    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    assert "matrix" in data
    assert len(data["features"]) > 0


def test_distribution(client):
    response = client.get("/api/distribution/api_gravity")
    assert response.status_code == 200
    data = response.json()
    assert data["feature"] == "api_gravity"
    assert "bins" in data
    assert "counts" in data


def test_distribution_not_found(client):
    response = client.get("/api/distribution/nonexistent_feature")
    assert response.status_code == 404


def test_sample(client):
    response = client.get("/api/sample/0")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_sample_not_found(client):
    response = client.get("/api/sample/999999")
    assert response.status_code == 404


def test_predict_valid(client):
    response = client.post("/api/predict", json={})
    assert response.status_code in (200, 400, 503)


def test_predict_with_params(client):
    payload = {
        "api_gravity": 35.0,
        "sulfur_content_pct": 0.5,
        "viscosity_cp": 20.0,
        "density_kg_m3": 850.0,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code in (200, 400, 503)
