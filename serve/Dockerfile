FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serve/ serve/

RUN mkdir -p artifacts

ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db
ENV UMBRAL_FALLA=0.40

EXPOSE 8002
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8002"]
