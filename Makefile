.PHONY: all install lint pipeline pipeline-sintetico serve docker smoke clean help

install:
	pip install -r requirements.txt

all: install lint pipeline serve

lint:
	flake8 . --max-line-length=100 --exclude=.git,__pycache__,mlruns,artifacts,data,reportes

# Pipeline completo con datos reales (UCI AI4I 2020)
pipeline:
	python main.py

# Pipeline con datos sintéticos — CI/CD y desarrollo rápido
pipeline-sintetico:
	python main.py --sintetico

# Etapas individuales
etapa1: ; python data/run.py
etapa1-sintetico: ; python data/run.py --sintetico
etapa2: ; python preprocess/run.py
etapa3: ; python segregate/run.py
etapa4: ; pytest check_data/test_data.py -v --tb=short
etapa5: ; python random_forest/run.py
etapa6: ; python evaluate/run.py
etapa7: ; python drift/run.py

serve:
	uvicorn serve.app:app --host 0.0.0.0 --port 8002 --reload

docker-build:
	docker build -t maintenance-api:local -f serve/Dockerfile .

docker-run:
	docker run -p 8002:8002 --name maintenance-api maintenance-api:local

docker-stop:
	docker stop maintenance-api && docker rm maintenance-api

smoke:
	curl -sf http://localhost:8002/health | python3 -m json.tool
	@echo ""
	curl -X POST http://localhost:8002/predict \
	  -H 'Content-Type: application/json' \
	  -d '{"tipo_maquina":"M","temperatura_aire":298.1,"temperatura_proceso":308.6,"velocidad_rpm":1551,"torque_nm":42.8,"desgaste_herramienta_min":0}' \
	  | python3 -m json.tool

mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5001

clean:
	rm -rf data/ artifacts/ reportes/ mlruns/ __pycache__ pipeline_run.log mlflow.db
	find . -name "*.pyc" -delete
	@echo "Limpieza completada"

help:
	@echo ""
	@echo "=== Pipeline MLOps — Mantenimiento Predictivo ==="
	@echo "  make install            — instalar dependencias"
	@echo "  make pipeline           — ejecutar con datos reales (UCI AI4I 2020)"
	@echo "  make pipeline-sintetico — ejecutar con datos sintéticos (rápido)"
	@echo "  make etapa1-7           — ejecutar una etapa específica"
	@echo "  make serve              — levantar API FastAPI en puerto 8002"
	@echo "  make smoke              — test rápido de endpoints"
	@echo "  make mlflow-ui          — UI de MLflow en puerto 5001"
	@echo "  make clean              — limpiar artefactos"
	@echo ""
