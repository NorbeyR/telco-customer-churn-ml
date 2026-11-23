# Fase 4 · Monitoreo de la API de predicción de churn

API REST (FastAPI) que predice el abandono de clientes (IBM Telco Customer Churn). Esta fase parte de `fase-3/` y agrega:

- **Registro de solicitudes** a `POST /predict` (`src/request_log.py` y un middleware en `api.py`).
- **Script de monitoreo** (`monitor.py`) que calcula el total de solicitudes y el porcentaje de fallos.
- **Docker** con volúmenes persistentes para `models/` y `logs/`.

## 1. Cómo ejecutar en local

Desde `fase-4/`, con las dependencias de `requirements.txt` instaladas:

```bash
python -m uvicorn api:app --port 8000          # API con registro habilitado (versión v1)
python cliente.py demo                         # 2 solicitudes válidas y 1 inválida
python monitor.py --log logs/predicciones_log.csv
```

Variables de entorno (valores por defecto):

| Variable | Defecto | Uso |
| :--- | :--- | :--- |
| `MODEL_VERSION` | `v1` | Versión que usa `/predict` |
| `MODELS_DIR` | `models` | Carpeta de versiones |
| `LOG_PATH` | `logs/predicciones_log.csv` | Registro persistente |
| `DATA_PATH` | `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` | Datos para `/train` |

Rutas: `GET /health`, `GET /versions`, `POST /predict`, `POST /train`.

## 2. Docker

Desde `fase-4/`:

```bash
docker build -t churn-api-fase4:1.0 --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
docker run --rm -p 8000:8000 -v "$PWD/models:/app/models" -v "$PWD/logs:/app/logs" churn-api-fase4:1.0
```

- Puerto: `8000`. Los datos van dentro de la imagen; los modelos y el registro se montan como volúmenes, así que persisten entre ejecuciones.
- Registro persistente: `logs/predicciones_log.csv` en la carpeta montada.
- Monitor sobre ese registro: `docker run --rm -v "$PWD/logs:/app/logs" churn-api-fase4:1.0 python monitor.py`
- Pruebas: `docker run --rm churn-api-fase4:1.0 python -m pytest -q`

## 3. Formato del registro

Una fila por cada llamada a `/predict`, incluidas las rechazadas (400, 422, 503 y 500). El encabezado se escribe solo al crear el archivo y las filas se agregan al final.

| Columna | Contenido |
| :--- | :--- |
| `fecha_hora` | Fecha y hora en **UTC** (Colombia es UTC-5: 23:58 UTC son las 18:58 locales) |
| `estado_http` | Código HTTP devuelto |
| `version` | Versión que produjo la predicción (vacía si hubo error) |
| `prediccion` | Etiqueta `Churn` / `No Churn` (vacía si hubo error) |

No se guardan datos del cliente. Si falla la escritura, la API avisa por stderr y sigue respondiendo.

## 4. Pruebas

Desde `fase-4/`: `python -m pytest -q` → 53 passed (48 de la API y el registro, 5 del monitoreo).

## 5. Reporte

**Solicitudes enviadas** (API con `v1`, sobre un registro vacío):

| # | Solicitud | Resultado |
| :-- | :--- | :--- |
| 1 | Registro válido | 200, `v1`, `Churn` |
| 2 | Registro válido | 200, `v1`, `No Churn` |
| 3 | Registro al que se le quitó `tenure` | 422, sin versión ni predicción |

Registro resultante (copia en `logs/muestra_registro.csv`):

```
fecha_hora,estado_http,version,prediccion
2026-11-22T23:58:11,200,v1,Churn
2026-11-22T23:58:11,200,v1,No Churn
2026-11-22T23:58:11,422,,
```

Salida de `python monitor.py --log logs/predicciones_log.csv`:

```
Registro: logs\predicciones_log.csv
Total de solicitudes: 3
Solicitudes fallidas (HTTP 4xx/5xx): 1
Porcentaje de fallos: 33,3 %
Respuestas exitosas por versión: v1: 2
```

Con un registro inexistente o vacío el script muestra total 0 y «sin solicitudes».

| Indicador | Valor |
| :--- | :--- |
| Total de solicitudes | 3 |
| Porcentaje de fallos | 33,3 % (1 de 3) |

**Interpretación.** La API procesó dos registros con `v1` y rechazó el tercero porque faltaba `tenure`. La tasa de fallos fue 1/3 (33,3 %), y el rechazo era el resultado esperado de la prueba de validación. Este porcentaje describe el funcionamiento del servicio, no la calidad de las predicciones, que requeriría conocer si los clientes realmente abandonaron.

## 6. Versiones del modelo

| Versión | Umbral | F1 | Recall | AUC | Costo (unidades relativas) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `v1` (activa por defecto) | 0,50 | 0,614 | 0,783 | 0,842 | 693 |
| `v2` | 0,37 | 0,602 | 0,898 | 0,842 | 596 |

Costos supuestos por el equipo: abandono no detectado = 5, falsa alarma = 1 (unidades relativas, no cifras reales).

Los detalles de cada versión están en `models/<versión>/metrics.json` y `metadata.json`.

La API usa la versión de `MODEL_VERSION` (por defecto `v1`). Para cambiar de versión o volver a la anterior se define la variable y se reinicia la API:

```powershell
$env:MODEL_VERSION = "v2"; python -m uvicorn api:app --port 8000      # PowerShell
docker run --rm -p 8000:8000 -e MODEL_VERSION=v2 -v "$PWD/models:/app/models" -v "$PWD/logs:/app/logs" churn-api-fase4:1.0
```

`GET /versions` muestra la versión activa y las disponibles. Si la configurada no existe, `/predict` responde 503 con la lista de versiones disponibles.

## 7. Política de reentrenamiento

El equipo revisa el modelo cuando reúne **200 clientes nuevos con resultado real** (se sabe si abandonaron) o cada **3 meses**, lo que ocurra primero. Una persona designada del equipo ejecuta `POST /train` con los datos ampliados para crear una versión candidata. La candidata se compara con la versión activa sobre los **mismos casos reservados**, con el costo y el recall de la Fase 2, y se activa (cambiando `MODEL_VERSION` y reiniciando) solo si reduce el costo sin bajar el recall; si no mejora, se conserva la activa. Los fallos 4xx se corrigen en quien envía los datos y no activan una revisión del modelo.
