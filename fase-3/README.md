# Fase 3 - API REST del modelo de churn
**Modelos y Simulación de Sistemas I - Universidad de Antioquia - 2026 II**

**Integrantes:** Heidy Dayana Gamboa Chaverra, Santiago Restrepo Rodriguez, Norbey David Rincón Bedoya.
**Etiqueta de Git de esta entrega:** `entrega-3`.

## 1. Qué hace y cómo se relaciona con las fases anteriores
Expone con **FastAPI** el modelo de la [Fase 2](../fase-2/README.md) (predicción de abandono de clientes, IBM Telco Customer Churn; datos de una empresa **ficticia**, uso didáctico). No se reescribió la lógica: `src/` es la de la Fase 2 y la API llama a las mismas funciones (`predict_dataframe`, `run_training`, `load_bundle`). Único cambio en `src/`: `MissingColumnsError` guarda la lista de columnas (`.columns`) para devolverla en el JSON.

| Ruta | Qué hace |
| :--- | :--- |
| `GET /health` | HTTP 200 y JSON que indica que la API está en ejecución (y si el modelo activo está disponible). |
| `GET /versions` | Versiones disponibles y versión activa. |
| `POST /predict` | Recibe **un registro JSON** con las variables predictoras, usa la versión activa y devuelve la predicción y la versión. No necesita la variable objetivo. |
| `POST /train` | Entrena con los datos y la configuración de la Fase 2, guarda una versión nueva (modelo, métricas, metadatos) y devuelve su identificador. Conserva las anteriores. |

Documentación interactiva generada por FastAPI: `http://localhost:8000/docs`.

## 2. Estructura
```
fase-3/
  api.py            # API (create_app, rutas, manejo de errores)
  cliente.py        # cliente de prueba (solo librería estándar)
  train.py, predict.py, src/   # Fase 2 (reutilizados)
  tests/            # pytest: Fase 2 + tests/test_api.py
  data/             # CSV de datos y de ejemplo
  models/           # v1, v2 (Fase 2) y la versión creada con /train
  Dockerfile, .dockerignore, requirements.txt, pytest.ini
```

## 3. Configuración (variables de entorno)
| Variable | Por defecto | Uso |
| :--- | :--- | :--- |
| `MODEL_VERSION` | `v1` | **Versión que usa `/predict`.** Cambiarla y reiniciar la API selecciona otra versión disponible. |
| `MODELS_DIR` | `models` | Carpeta con las versiones (`models/<version>/modelo.joblib`). |
| `DATA_PATH` | `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` | Datos que usa `/train`. |
| `TARGET` | `Churn` | Variable objetivo. |
| `SEED` | `42` | Semilla de `/train` (puede sobrescribirse en el cuerpo). |

**Cómo seleccionar una versión disponible:** consulte `GET /versions` (o la carpeta `models/`) y reinicie con la variable, por ejemplo `MODEL_VERSION=v2`. Si la versión configurada no existe, `/predict` responde 503 con la lista de versiones disponibles.

## 4. Ejecución local (Python 3.12 o 3.13)
```bash
cd fase-3
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
# con otra versión:  MODEL_VERSION=v2 uvicorn api:app --port 8000      (PowerShell: $env:MODEL_VERSION="v2"; uvicorn ...)
```

## 5. Ejemplo de solicitud y respuesta (registro real del CSV de ejemplo)
`POST /predict` con el cliente `7590-VHVEG`. Se conservan los tipos de entrenamiento: números como números, categorías como texto, y `null` para un dato faltante (p. ej. `TotalCharges` de un cliente nuevo; se aplica la regla guardada: imputar 0).
```json
{"customerID": "7590-VHVEG", "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
 "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service", "InternetService": "DSL",
 "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No",
 "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
 "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85}
```
Respuesta `HTTP 200`:
```json
{"customerID": "7590-VHVEG", "prediccion": "Churn", "clase": 1, "probabilidad_churn": 0.804827,
 "umbral": 0.5, "version": "v1", "advertencias": {}}
```
`prediccion` es la etiqueta (`Churn` = el cliente abandonará, `No Churn` = permanece); `clase` es 1/0; `version` es la versión que produjo la predicción; `advertencias` informa columnas extra ignoradas, categorías no vistas o `TotalCharges` imputados. `customerID` es opcional (se devuelve si se envía).

**Esquema de entrada:** `tenure`, `MonthlyCharges` (números ≥ 0, sin nulos), `TotalCharges` (número ≥ 0 o `null`), `SeniorCitizen` (0/1 o "No"/"Yes") y 15 variables de texto (`gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`). Ver detalle en el README de la Fase 2.

## 6. Errores
| Situación | Código | Respuesta |
| :--- | :---: | :--- |
| Falta una variable obligatoria | 422 | `{"detail": "Faltan columnas obligatorias: ['tenure']", "campos_faltantes": ["tenure"]}` |
| Tipo o valor incompatible (texto en `tenure`, negativos, `SeniorCitizen` = 9, nulo en `tenure`) | 422 | mensaje con el nombre de la columna y las filas/valores |
| Cuerpo que no es JSON o no es un objeto | 400 | mensaje explicativo |
| Versión activa no disponible | 503 | `detail` con instrucciones y `versiones_disponibles` |
| `/train` con una versión ya existente | 409 | no se sobrescribe |
| `/train` sin datos disponibles | 503 | indica que faltan los datos |

## 7. `/train`
Cuerpo vacío o, opcionalmente, `{"version": "v4", "seed": 7, "optimize_threshold": true}`. Si no se indica versión se usa el siguiente identificador libre (`v3` si existen `v1` y `v2`). Responde `201` con el identificador y las métricas, y **no cambia la versión activa**: la siguiente predicción sigue usando `MODEL_VERSION` hasta que se cambie y se reinicie.
```bash
python cliente.py train
```
```json
{"version": "v3", "ruta": "models/v3",
 "metricas": {"accuracy": 0.7381, "recall_churn": 0.7834, "f1_churn": 0.6136, "roc_auc": 0.8415, "umbral": 0.5},
 "mensaje": "Versión guardada. La API sigue usando 'v1' hasta que cambie MODEL_VERSION y reinicie."}
```
**Comprobación manual:** `ls models` debe mostrar `v1  v2  v3` y `models/v3/` debe contener `modelo.joblib`, `metrics.json` y `metadata.json`.

## 8. Cliente de prueba
Solo requiere Python (librería estándar), con la API en marcha:
```bash
python cliente.py health
python cliente.py predict                        # primera fila de data/ejemplo_prediccion.csv
python cliente.py predict --fila 2
python cliente.py predict --sin-campo tenure     # entrada inválida -> 422
python cliente.py train
```
Equivalente con curl (Bash):
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"customerID":"7590-VHVEG","gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":1,"PhoneService":"No","MultipleLines":"No phone service","InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":29.85,"TotalCharges":29.85}'
```

## 9. Docker (terminal: Git Bash en Windows con Docker Engine en WSL 2; el prefijo `MSYS_NO_PATHCONV=1` evita que Git Bash altere las rutas; en Linux/macOS no hace falta)
Desde `fase-3/`:
```bash
docker build -t modelos-fase3:1.0 --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
MSYS_NO_PATHCONV=1 docker run --rm -p 8000:8000 \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/models:/app/models" \
  modelos-fase3:1.0
```
- **Puerto:** 8000 (`-p 8000:8000`).
- **Datos:** carpeta `data/` montada en solo lectura (los usa `/train`).
- **Modelos:** carpeta `models/` montada con escritura: las versiones leídas por `/predict` y las nuevas creadas por `/train` persisten en el equipo anfitrión.
- **Otra versión:** agregue `-e MODEL_VERSION=v2`.
- En PowerShell use `${PWD}` y la continuación de línea con `` ` ``.

Con el contenedor en marcha, en otra terminal: `python cliente.py health`, `python cliente.py predict`, `python cliente.py train` (o los comandos curl de la sección 8).

**Evidencia de ejecución:** `

```text
# 1. Arranque del contenedor y peticiones recibidas
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on [http://0.0.0.0:8000](http://0.0.0.0:8000) (Press CTRL+C to quit)
INFO:     172.17.0.1:57330 - "GET /health HTTP/1.1" 200 OK
INFO:     172.17.0.1:57340 - "POST /predict HTTP/1.1" 200 OK
INFO:     172.17.0.1:57352 - "POST /train HTTP/1.1" 201 Created

# 2. Comprobación con cliente.py y persistencia en models/
$ python cliente.py health
--- GET /health
HTTP 200
{"status": "ok", "mensaje": "La API está en ejecución", "version_activa": "v1", "modelo_disponible": true}

$ python cliente.py predict
--- POST /predict
HTTP 200
{"customerID": "7590-VHVEG", "prediccion": "Churn", "clase": 1, "probabilidad_churn": 0.804827, "umbral": 0.5, "version": "v1", "advertencias": {}}

$ python cliente.py train
--- POST /train
HTTP 201
{"version": "v3", "ruta": "models/v3", "metricas": {"accuracy": 0.7381, "recall_churn": 0.7834, "f1_churn": 0.6136, "roc_auc": 0.8415, "umbral": 0.5}, "mensaje": "Versión guardada. La API sigue usando 'v1' hasta que cambie MODEL_VERSION y reinicie."}

$ ls models
v1/  v2/  v3/

# 3. Ejecución de pytest dentro del contenedor
$ MSYS_NO_PATHCONV=1 docker run --rm modelos-fase3:1.0 python -m pytest -q
..........................................                               [100%]
42 passed, 1 warning in 41.19s
````

## 10. Pruebas
```bash
python -m pytest            # dentro de Docker: docker run --rm modelos-fase3:1.0 python -m pytest -q
```
Resultado esperado: **`42 passed`** (26 de la Fase 2 + 16 de la API). Usan un modelo pequeño entrenado en un directorio temporal, sin red y sin tocar `models/`.

| Prueba pedida | Test |
| :--- | :--- |
| `/health` responde 200 | `test_health_responde_200` |
| Entrada válida devuelve predicción y versión | `test_predict_valido_devuelve_prediccion_y_version` |
| Campo obligatorio ausente se rechaza | `test_predict_con_campo_obligatorio_ausente_se_rechaza` |

Además: nulos y espacios en `TotalCharges`, columnas extra, tipos inválidos, JSON inválido, modelo no disponible (503), `/train` (versión nueva, anteriores conservadas, 409 y 422).

## 11. Limitaciones
Datos ficticios; un registro por solicitud; `/train` usa siempre el CSV configurado (no incorpora datos nuevos); la versión activa se cambia por configuración y reinicio; sin autenticación (uso local/académico).

## 12. Fuentes
Documentación de FastAPI, Starlette, Uvicorn, scikit-learn, pytest y Docker. No se adaptó código externo.
