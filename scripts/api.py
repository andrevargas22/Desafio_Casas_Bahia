"""
API Titanic 

Este script implementa uma API FastAPI para prever a sobrevivência de passageiros do Titanic 
com base em um modelo treinado e armazenado no servidor MLFlow. A API aceita dados de entrada 
no formato JSON, valida os dados, processa-os e retorna as previsões de sobrevivência.

Funcionalidades:
- Carregar um modelo treinado do MLFlow.
- Configurar middleware CORS para permitir solicitações de diferentes origens.
- Validar os dados de entrada para garantir que as colunas esperadas e seus tipos estejam presentes.
- Processar os dados de entrada para transformá-los em um formato adequado para predição.
- Retornar previsões baseadas nos dados de entrada.

Endpoints:
- /predict (POST): Aceita dados JSON, valida, processa e retorna as previsões.

Funções:
- validate_data(df: pd.DataFrame) -> (bool, str): Valida o DataFrame de entrada para garantir que possui as colunas e tipos esperados.
- process_data(df: pd.DataFrame) -> pd.DataFrame: Processa o DataFrame de entrada selecionando colunas relevantes, codificando a coluna 'Sex' e removendo valores ausentes.
- predict(request: Request): Endpoint da API que lida com solicitações de predição, valida os dados, processa-os e retorna as previsões.

Como usar:
1. Configure o servidor MLFlow e ajuste o URI de rastreamento conforme necessário.
2. Inicie a API FastAPI executando o script.
3. Envie uma solicitação POST para o endpoint /predict com os dados de entrada no formato JSON.

Exemplo de dados de entrada:
[
    {
        "Pclass": 3,
        "Sex": "male",
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25
    },
    ...
]

Requisitos:
- pandas
- numpy
- mlflow
- fastapi
- uvicorn

Autor:
André Vargas (andrevargas22@gmail.com)
"""

##################### 1. BIBLIOTECAS
import pandas as pd
import numpy as np
import mlflow.pyfunc
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

##################### 2. CONFIGURAÇÕES
# Iniciar a aplicação FastAPI
app = FastAPI()

# Configurações MLFlow:
# - Definir URI de rastreamento do server MLFlow
# - Carregar o modelo treinado do MLFlow

TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
mlflow.set_tracking_uri(TRACKING_URI)

model_name = "titanic"
stage = 'staging'
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")

# Configurar CORS
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # Permitir todas as origens para fins de teste
    allow_credentials=True,
    allow_methods=["*"],        # Permitir todos os métodos HTTP
    allow_headers=["*"],        # Permitir todos os cabeçalhos
)

##################### 3. FUNÇÕES
def validate_data(df: pd.DataFrame) -> (bool, str):
    """
    Valida o DataFrame de entrada para garantir que possui as colunas e tipos esperados.
    
    Parâmetros:
        df (pd.DataFrame): O DataFrame de entrada para validar.

    Retorna:
        (bool, str): Uma tupla contendo um valor booleano indicando se os dados são válidos e uma mensagem de erro, se houver.
    """
    
    # Definir as colunas esperadas e se são numéricas ou não
    expected_columns = {
        'Pclass': 'numeric',
        'Sex': 'object',
        'SibSp': 'numeric',
        'Parch': 'numeric',
        'Fare': 'numeric',
    }

    # Verificar se todas as colunas esperadas estão presentes e são do tipo correto
    for col, expected_type in expected_columns.items():
        if col not in df.columns:
            return False, f"Coluna faltante: {col}"
        if expected_type == 'numeric' and not pd.api.types.is_numeric_dtype(df[col]):
            return False, f"Tipo incorreto para a coluna: {col}. Esperava 'numeric', recebido '{df[col].dtype}'"
        if expected_type == 'object' and not pd.api.types.is_object_dtype(df[col]):
            return False, f"Tipo incorreto para a coluna: {col}. Esperava 'string', recebido '{df[col].dtype}'"
    
    return True, ""

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processa o DataFrame de entrada selecionando colunas relevantes, codifica a coluna 'Sex' e remove valores ausentes.
    
    Parâmetros:
        df (pd.DataFrame): DataFrame de entrada para processar.

    Retorna:
        pd.DataFrame: DataFrame processado e pronto para previsão.
    """
    
    # Selecionar colunas relevantes para o modelo
    df = df[['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare']]
    
    # Codificar a coluna 'Sex' para valores numéricos
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    # Remover linhas com valores ausentes
    df = df.dropna()
    
    return df

##################### 4. ENDPOINTS
@app.post("/predict")
async def predict(request: Request):
    """
    Endpoint da API que lida com solicitações de predict, valida os dados, processa e retorna as previsões.

    Parâmetros:
        request (Request): A solicitação POST contendo os dados JSON de entrada.

    Retorna:
        dict: Um dicionário com as previsões.
    """
    
    # Obter os dados da solicitação
    data = await request.json()
    
    # Converter o JSON para um dataframe pandas
    df = pd.DataFrame(data)

    # Validar os dados
    is_valid, error_message = validate_data(df)
    if not is_valid:
        return {"error": error_message}
    
    # Processar os dados
    df = process_data(df)

    # Realizar previsões
    predictions = model.predict(df)

    # Converter as previsões para uma lista
    predictions_list = predictions.tolist()
        
    return {"predictions": predictions_list}

##################### 5. INICIAR A APLICAÇÃO
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
