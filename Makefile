## Instalação de requirements
install:
	pip install --upgrade pip && pip install -r requirements.txt

# Formatação e linting
code-review:
	black scripts/
	pylint scripts/*.py

# Rodar a API localmente
api-local:
	export MLFLOW_TRACKING_URI="https://mlflow-server-wno7iop4fa-uc.a.run.app/" && \
	uvicorn scripts.api:app --reload --host 127.0.0.1 --port 8000