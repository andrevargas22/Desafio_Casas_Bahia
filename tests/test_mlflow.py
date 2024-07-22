"""
Este script contém testes para verificar a conexão com o servidor MLFlow e
a existência de um modelo específico no Model Registry do MLFlow.

Funções:
- test_conexao_mlflow(): Testa a conexão com o servidor MLFlow e verifica se é possível buscar experimentos.
- test_get_model_mlflow(): Testa a existência de um modelo específico no Model Registry do MLFlow.

Dependências:
- mlflow
- scripts.api.connect_mlflow: Função para configurar a URI de rastreamento do MLFlow.
- scripts.api.fetch_model: Função para carregar o modelo treinado do MLFlow.

Autor:
André Vargas (andrevargas22@gmail.com)
"""

import mlflow
from scripts.api import connect_mlflow, fetch_model

def test_conexao_mlflow():
    """
    Teste de conexão com o MLFlow
    
    Asserts:
        - connect = True, conexão bem sucedida
    """
    def _check_conection():
        try:
            connect_mlflow()
            experiments = mlflow.search_experiments()
            return True
        except:
            return False
    
    connect = _check_conection()
    
    assert connect == True, f"Falha de conexão com o MLFlow"
    
def test_get_model_mlflow():
    """
    Teste para saber se o modelo existe no Model Registry
    
    Asserts:
        - model_exists = True, o modelo existe
    """
    
    def _check_model_existence():
        # Tenta carregar o modelo
        try:
            fetch_model()
            return True
        except mlflow.exceptions.RestException:
            return False

    # Pega a ultima versão do modelo que está no estágio especificado, no Model Registry
    model_exists = _check_model_existence()
        
    assert model_exists == True, f"O modelo configurado em config_mlflow/params.yml não existe no server MLFlow"