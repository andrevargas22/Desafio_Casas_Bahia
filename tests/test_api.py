"""
Script de testes para endpoints da API Titanic

Este script contém testes para verificar o comportamento dos endpoints da API Titanic,
testando diferentes cenários de entrada de dados.

Funções:
- test_predict_valid_data(): Testa o endpoint de predição com dados válidos e verifica se as previsões são retornadas corretamente.
- test_predict_invalid_data(): Testa o endpoint de predição com dados inválidos (tipo incorreto) e verifica se um erro é retornado.
- test_predict_missing_column(): Testa o endpoint de predição com dados faltando uma coluna obrigatória e verifica se um erro é retornado.

Dependências:
- pytest
- fastapi.testclient.TestClient
- scripts.api.app: A aplicação FastAPI que está sendo testada.

Autor:
André Vargas (andrevargas22@gmail.com)
"""

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

    # Enviar requisição POST
    response = client.post("/predict", json=data)
    predictions = response.json()["predictions"]

    # Verifica se a resposta foi bem sucedida e se as previsões foram retornadas
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