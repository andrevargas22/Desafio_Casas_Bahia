## Instalação de requirements
install:
	pip install --upgrade pip && pip install -r requirements.txt

# Formatação e linting
code-review:
	black scripts/
	pylint scripts/*.py
	pytest --cov=scripts tests/

# Rodar a API localmente
local:
	uvicorn scripts.api:app --reload --host 127.0.0.1 --port 8000