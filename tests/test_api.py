import pytest
from fastapi.testclient import TestClient
from scripts.api import app

client = TestClient(app)

def test_predict_valid_data():
    # Dados de entrada válidos
    data = [
        {
            "Pclass": 3,
            "Sex": "male",
            "SibSp": 1,
            "Parch": 0,
            "Fare": 7.25
        },
        {
            "Pclass": 1,
            "Sex": "female",
            "SibSp": 1,
            "Parch": 0,
            "Fare": 71.2833
        }
    ]

    response = client.post("/predict", json=data)
    predictions = response.json()["predictions"]

    assert response.status_code == 200
    assert "predictions" in response.json()
    assert len(predictions) == 2

def test_predict_invalid_data():
    # Dados de entrada inválidos
    data = [
        {
            "Pclass": 3,
            "Sex": "male",
            "SibSp": 1,
            "Parch": 0,
            "Fare": "invalid_fare"  # Fare inválido
        }
    ]

    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert "error" in response.json()

def test_predict_missing_column():
    # Dados de entrada com coluna faltante
    data = [
        {
            "Pclass": 3,
            "Sex": "male",
            "SibSp": 1,
            "Parch": 0
            # Fare faltante
        }
    ]

    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert "error" in response.json()