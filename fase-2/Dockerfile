# Imagen base liviana de Python compatible con las dependencias del proyecto
FROM python:3.13-slim

# Directorio de trabajo en el contenedor
WORKDIR /app

# Desactivar buffer de salida e impedir archivos .pyc
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copiar e instalar dependencias primero (aprovecha la caché de capas de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar carpetas y scripts
COPY src/ ./src/
COPY tests/ ./tests/
COPY data/ ./data/
COPY train.py predict.py ./
COPY pytest.ini ./

# Crear carpetas para la persistencia con volúmenes montados
RUN mkdir -p models outputs

# Comando predeterminado al iniciar el contenedor
CMD ["python", "train.py", "--help"]