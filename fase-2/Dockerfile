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

# Trazabilidad: commit del código que construyó la imagen. Se recibe con
#   docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
# y src/versioning.py lo lee de esta variable de entorno para escribirlo en
# metadata.json (dentro del contenedor no existe la carpeta .git).
# Va al final para no invalidar la caché de las capas anteriores.
ARG GIT_COMMIT=no-disponible
ENV GIT_COMMIT=${GIT_COMMIT}

# Comando predeterminado al iniciar el contenedor
CMD ["python", "train.py", "--help"]
