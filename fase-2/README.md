# Fase 2 - Aplicación de predicción de churn (scripts + Docker)
**Modelos y Simulación de Sistemas I - Universidad de Antioquia - 2026 II**

**Integrantes:** Heidy Dayana Gamboa Chaverra, Santiago Restrepo Rodriguez, Norbey David Rincón Bedoya.
**Commit / etiqueta de la Entrega 2:** `entrega-2` (commit: `532d522`).

## 1. Problema y relación con la Fase 1
Clasificación binaria: predecir si un cliente de telecomunicaciones abandonará (`Churn = Yes`) con datos de su cuenta. Se conservan el problema, el dataset (IBM Telco Customer Churn), la partición 80/20 estratificada con semilla 42 y el modelo (Regresión Logística balanceada + Dummy como base) de [`../fase-1`](../fase-1/README.md). La Fase 2 convierte el notebook en una aplicación con `train.py`, `predict.py`, versiones de modelo, Docker y pruebas.

**Aclaración sobre los datos:** IBM publica este conjunto como una muestra de una empresa **ficticia**. Es útil con fines didácticos, pero no acredita registros reales ni permite extrapolar los resultados a operadores colombianos.

**Horizonte y momento de captura:** `Churn` indica si el cliente abandonó en el último mes; se asume un horizonte de un mes. Las variables son una captura de la cuenta; que sean datos de cuenta no demuestra que estén disponibles *antes* del evento (p. ej. `tenure`/`TotalCharges` se miden en el corte). Es una limitación conocida.

## 2. Correcciones respecto a la Fase 1
| Retroalimentación | Qué se hizo |
| :--- | :--- |
| El artefacto no aceptaba el CSV original (`TotalCharges` con espacios, `SeniorCitizen` 0/1). | Las conversiones están **dentro del Pipeline** (paso `InputCleaner` en `src/preprocessing.py`). El artefacto acepta el CSV original; ver contrato en la sección 5. |
| Faltaban pruebas con cliente nuevo, faltantes y categorías desconocidas. | `tests/test_prediccion.py` las cubre. |
| Versiones no fijadas (el artefacto declaraba scikit-learn 1.9.1). | `requirements.txt` con versiones exactas; Docker usa las mismas. |
| Costos y umbral. | `metrics.json` traduce los errores a costo de retención (supuestos 5:1 relativos) y `--optimize-threshold` elige el umbral con validación cruzada **dentro de train**. |
| Interpretación 7.2 incompleta, referencias a secciones inexistentes, regularización. | Corregido en `fase-1/notebook.ipynb` (7.2 con números, L2 explicada, referencias sincronizadas). |
| README de Fase 1: rutas, entorno de notebooks, accuracy 74,5 %. | Corregido en `fase-1/README.md` (ahora 73,8 %) y README raíz con rutas por carpeta. Se retiró la afirmación de "listo para producción". |
| PR final sin revisión formal. | El PR `develop -> main` de la Fase 2 fue aprobado por los otros dos integrantes. |
| Datos ficticios y horizonte de abandono. | Aclarado arriba y en el notebook. |

## 3. Entorno
Python **3.13** (probado con 3.13; el Docker usa `python:3.13-slim`).
```bash
cd fase-2
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 4. Datos
- Fuente: [IBM Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). El CSV (≈1 MB) **ya está en el repositorio**: `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` (misma copia que `fase-1/`). No se requieren credenciales.
- `data/ejemplo_prediccion.csv`: 9 clientes reales del dataset **sin** `Churn` (incluye 3 clientes nuevos con `tenure = 0` y `TotalCharges` vacío).
- `data/ejemplo_invalido.csv`: el mismo ejemplo sin la columna `Contract`, para demostrar el error.

## 5. Esquema y contrato de entrada del CSV de predicción
| Columna | Tipo aceptado | Observación |
| :--- | :--- | :--- |
| `tenure`, `MonthlyCharges` | numérico ≥ 0 | obligatorias, sin nulos |
| `TotalCharges` | numérico ≥ 0 | vacío / espacios / nulo → se imputa con **0** (regla de la Fase 1 para clientes nuevos) |
| `SeniorCitizen` | `0`/`1` o `No`/`Yes` | se convierte a `No`/`Yes` |
| `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod` | texto | obligatorias |

Reglas: el **orden de las columnas no importa**; las **columnas adicionales se ignoran** (con aviso); una **categoría no vista** en entrenamiento se codifica con ceros (con aviso); un **nulo en una categórica** se trata como categoría desconocida (con aviso); `customerID` (opcional) se copia a la salida. Una columna obligatoria ausente, un valor no numérico/negativo o un nulo en `tenure`/`MonthlyCharges` producen un error claro, código de salida ≠ 0 y **ningún archivo de salida**.

## 6. Comandos (terminal usada: Git Bash - MINGW64 en Windows 11 con WSL 2 y Docker Desktop)
Ayuda:
```bash
python train.py --help
python predict.py --help
```
Entrenar (una sola línea). v1 reproduce la Fase 1 (umbral 0,5); v2 elige el umbral por costo con validación cruzada en train:
```bash
python train.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv --target Churn --output-dir models --version v1 --seed 42
python train.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv --target Churn --output-dir models --version v2 --seed 42 --optimize-threshold
```
Predecir con cada versión:
```bash
python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones_v1.csv
python predict.py --model models/v2 --input data/ejemplo_prediccion.csv --output outputs/predicciones_v2.csv
```
Error esperado (código de salida 1, no se crea el archivo):
```bash
python predict.py --model models/v1 --input data/ejemplo_invalido.csv --output outputs/no_debe_existir.csv
```
Repetir una versión (`--version v1`) falla con "La versión 'v1' ya existe": nunca se sobrescribe.

**Salida** (`outputs/*.csv`): `customerID`, `prediccion` (1 = abandona/Churn, 0 = permanece/No Churn), `prediccion_etiqueta`, `probabilidad_churn`; mismo orden que la entrada.

## 7. Versiones, métricas y metadatos
```
models/
  v1/  modelo.joblib  metrics.json  metadata.json
  v2/  modelo.joblib  metrics.json  metadata.json
```
`modelo.joblib` guarda el Pipeline completo (InputCleaner + ColumnTransformer + clasificador) y el umbral. `metadata.json` incluye versión, fecha, objetivo, algoritmo, variables y tipos, parámetros, semilla, datos (archivo, SHA-256, filas), partición, versiones de Python/scikit-learn y commit de Git.

## 8. Resultados (conjunto de prueba, 1.409 clientes)
| Modelo | Umbral | Accuracy | Recall | F1 | ROC-AUC | Detectados / no detectados / falsas alarmas | Costo* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Base (Dummy) | 0,50 | 0,735 | 0,000 | 0,000 | 0,500 | 0 / 374 / 0 | 1.870 |
| **v1** Regresión Logística | 0,50 | 0,738 | 0,783 | 0,614 | 0,842 | 293 / 81 / 288 | 693 |
| **v2** mismo modelo, umbral por costo | 0,37 | 0,685 | 0,898 | 0,602 | 0,842 | 336 / 38 / 406 | 596 |

\*Costo = 5 × abandonos no detectados + 1 × falsas alarmas (unidades relativas, **supuesto**, no cifras reales). v2 baja el costo pero aumenta las falsas alarmas; el AUC no cambia porque es el mismo modelo.
**Limitaciones:** datos ficticios; corte transversal; costos supuestos; un solo algoritmo; el solver es determinista pero los resultados pueden variar levemente con otras versiones de librerías.

## 9. Docker (terminal Bash; en PowerShell reemplace `$PWD` por `${PWD}` y `\` por `` ` ``)
```bash
docker build -t modelos-fase2:1.0 --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
mkdir -p models outputs
docker run --rm -v "$PWD/data:/app/data:ro" -v "$PWD/models:/app/models" modelos-fase2:1.0 python train.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv --target Churn --output-dir models --version v-docker --seed 42
docker run --rm -v "$PWD/data:/app/data:ro" -v "$PWD/models:/app/models:ro" -v "$PWD/outputs:/app/outputs" modelos-fase2:1.0 python predict.py --model models/v-docker/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/docker_predicciones.csv
```
El entrenamiento no ocurre en `docker build`; modelos y predicciones persisten en `models/` y `outputs/` del equipo anfitrión. `v-docker` y `outputs/docker_*` están en `.gitignore`.

**Evidencia de ejecución:** Git Bash + Docker Engine en WSL 2:

1. **Construcción de la imagen Docker (`modelos-fase2:1.0`):**
```text
$ docker build -t modelos-fase2:1.0 .
[+] Building 144.5s (15/15) FINISHED
 => [internal] load build definition from Dockerfile
 => [internal] load metadata for docker.io/library/python:3.13-slim
 => [ 1/10] FROM docker.io/library/python:3.13-slim
 => [ 4/10] RUN pip install --no-cache-dir -r requirements.txt
 => exporting to image
```

2. **Entrenamiento en contenedor (`models/v-docker`):**
```bash
$ MSYS_NO_PATHCONV=1 docker run --rm -v "$PWD/data:/app/data:ro" -v "$PWD/models:/app/models" modelos-fase2:1.0 python train.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv --version v-docker
```
**Salida obtenida:**
```text
Versión 'v-docker' entrenada con 5634 clientes y evaluada con 1409 (umbral = 0.5).
                accuracy    recall        f1   roc_auc     costo
Baseline            0.7346    0.0000    0.0000    0.5000    1870.0
Modelo              0.7381    0.7834    0.6136    0.8415     693.0
Abandonos detectados: 293 de 374 | no detectados: 81 | falsas alarmas: 288
```

3. **Predicción en contenedor (`outputs/docker_predicciones.csv`):**
```bash
$ MSYS_NO_PATHCONV=1 docker run --rm -v "$PWD/data:/app/data:ro" -v "$PWD/models:/app/models:ro" -v "$PWD/outputs:/app/outputs" modelos-fase2:1.0 python predict.py --model models/v-docker --input data/ejemplo_prediccion.csv --output outputs/docker_predicciones.csv
```
**Salida obtenida:**
```text
AVISO: 3 valores de TotalCharges vacíos imputados con 0 (regla de la versión guardada).
Versión 'v-docker': 9 predicciones (3 con riesgo de abandono, umbral 0.5).
```

## 10. Pruebas
```bash
python -m pytest
```
| Archivo | Verifica |
| :--- | :--- |
| `test_entrenamiento.py` | genera artefacto, métricas y metadatos; partición sin fuga; umbral solo con train; error si falta el objetivo |
| `test_prediccion.py` | predicción por fila en orden; igual al modelo original; CSV original (espacios y 0/1); nulos; cliente nuevo; categorías desconocidas; columna ausente; tipos inválidos |
| `test_versionamiento.py` | segunda versión conserva la primera; id repetido no sobrescribe; id inválido |
| `test_cli.py` | `--help`, entrenar y predecir por CLI, código de salida ≠ 0 y sin archivo ante error |

Usan datos sintéticos pequeños y directorios temporales; sin red.

## 11. Estructura
`train.py` y `predict.py` (CLI) · `src/` (config, data, preprocessing, modeling, evaluation, versioning, training, inference) · `tests/` · `data/` · `models/` · `outputs/` · `Dockerfile` · `.dockerignore` · `requirements.txt`.

## 12. Fuentes
Documentación de scikit-learn (Pipeline, ColumnTransformer, LogisticRegression), pandas, pytest y Docker. No se adaptó código externo.