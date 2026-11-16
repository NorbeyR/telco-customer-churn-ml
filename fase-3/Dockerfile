# Imagen base de Python con versión explícita (la misma que se usó en la Fase 2)
FROM python:3.13-slim

# Commit del código (se pasa con --build-arg; .git no se copia a la imagen)
ARG GIT_COMMIT=no-disponible
ENV GIT_COMMIT=${GIT_COMMIT} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_VERSION=v1 \
    MODELS_DIR=models \
    DATA_PATH=data/WA_Fn-UseC_-Telco-Customer-Churn.csv

WORKDIR /app

# Dependencias primero (aprovecha la caché de capas)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código de la aplicación. Los modelos se montan como volumen (no van en la imagen).
COPY src/ ./src/
COPY tests/ ./tests/
COPY data/ ./data/
COPY api.py train.py predict.py cliente.py pytest.ini ./
RUN mkdir -p models

EXPOSE 8000

# Comprueba que la API responda
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"

# Arranca la API escuchando en todas las interfaces del contenedor.
# train.py y predict.py de la Fase 2 siguen disponibles con: docker run ... python train.py ...
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
