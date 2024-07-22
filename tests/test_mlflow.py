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
    
    assert connect == True, f"Failed to connect with MLFlow"
    
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
        
    assert model_exists == True, f"The model configured on config_mlflow/params.yml doesn't exist on MLFlow"