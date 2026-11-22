# Fase 2 - Aplicación de predicción de churn (scripts + Docker)
**Modelos y Simulación de Sistemas I - Universidad de Antioquia - 2026 II**

**Integrantes:** Heidy Dayana Gamboa Chaverra, Santiago Restrepo Rodriguez, Norbey David Rincón Bedoya.
**Entrega 2:** etiqueta de Git `entrega-2`, sobre el commit `12b13b6b` (el estado evaluado).

## 1. Problema y relación con la Fase 1
Clasificación binaria: predecir si un cliente de telecomunicaciones abandonará (`Churn = Yes`) con datos de su cuenta. Se conservan el problema, el dataset (IBM Telco Customer Churn), la partición 80/20 estratificada con semilla 42 y el modelo (Regresión Logística balanceada + Dummy como base) de [`../fase-1`](../fase-1/README.md). La Fase 2 convierte el notebook en una aplicación con `train.py`, `predict.py`, versiones de modelo, Docker y pruebas.

**Aclaración sobre los datos:** IBM publica este conjunto como una muestra de una empresa **ficticia**. Es útil con fines didácticos, pero no acredita registros reales ni permite extrapolarlos a operadores colombianos.

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

## 2b. Correcciones tras la evaluación de la Fase 2
| Observación de la evaluación | Qué se hizo |
| :--- | :--- |
| El README pasa `--build-arg GIT_COMMIT`, pero el Dockerfile no lo declaraba ni lo transmitía. | El `Dockerfile` ahora declara `ARG GIT_COMMIT` y `ENV GIT_COMMIT=${GIT_COMMIT}`. `src/versioning.py` ya leía esa variable, así que `metadata.json` guarda el commit también dentro de Docker. Prueba: `tests/test_trazabilidad.py`. |
| Ejecutar el recorrido desde un directorio limpio y comprobar que repetir un identificador no reemplaza la versión. | `verificar_recorrido.sh` (sección 9.1): entrena, carga, predice, muestra metadatos y comprueba que el identificador repetido se rechaza sin tocar `modelo.joblib`. |
| Relacionar modelo, métricas, entorno y commit por versión. | Tabla de trazabilidad en la sección 7. |
| Explicar semilla y datos antes de interpretar una métrica como mejora. | Nota en la sección 8: v1 y v2 comparten semilla, datos y partición; solo cambia el umbral. |
| Comandos copiables con rutas del repositorio. | Sección 3 con `git clone` y `cd telco-customer-churn-ml/fase-2`; `.gitattributes` fija LF en `*.sh` y `*.csv`. |
| (Hallazgo propio) El SHA-256 de los datos en `metadata.json` de v1 y v2 no coincide con el del CSV del repositorio. | Se debe a los saltos de línea (CRLF en el checkout de Windows). Explicado en la sección 7; `.gitattributes` evita que se repita en versiones nuevas. |
| (Hallazgo propio) La evidencia de Docker mostraba `0.185203` para `5575-GNVDE`. | El valor correcto es `0.118086` (coincide con `outputs/predicciones_v1.csv`); corregido en la sección 9. |

## 3. Entorno
Python **3.12 o 3.13** (el equipo probó con 3.12.10 y 3.13; el Docker usa `python:3.13-slim`).
```bash
git clone https://github.com/NorbeyR/telco-customer-churn-ml.git
cd telco-customer-churn-ml/fase-2
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

### Trazabilidad por versión
| Versión | Modelo (`modelo.joblib`, SHA-256) | Métricas (prueba) | Entorno | Commit del código | Creada (UTC) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `v1` | `d976efc6b2dc…` | umbral 0,50 · F1 0,614 · recall 0,783 · AUC 0,842 · costo 693 | Python 3.12.10 · scikit-learn 1.9.1 | `0efe1bca72e4…` | 2026-10-29 23:29 |
| `v2` | `191401a74406…` | umbral 0,37 · F1 0,602 · recall 0,898 · AUC 0,842 · costo 596 | Python 3.12.10 · scikit-learn 1.9.1 | `157c73eae2ad…` | 2026-10-30 14:26 |

Los hashes completos y el resto de campos están en `models/<versión>/metadata.json` (`codigo.commit`, `entorno`, `datos`, `particion`). Ambas versiones usan los mismos datos (7.043 filas; 5.634 entrenamiento y 1.409 prueba) y la misma semilla (42).

**Sobre el SHA-256 de los datos.** `metadata.json` de v1 y v2 registra `88be4b93fbe0…`, que corresponde al CSV con saltos de línea de Windows (CRLF). El archivo guardado en el repositorio (LF) tiene `16320c9c1ec7…`. Es el mismo contenido con otros saltos de línea; para que las versiones nuevas registren siempre el mismo valor, el repositorio incluye `.gitattributes` (`*.csv text eol=lf`). Las versiones v1 y v2 no se modifican (nunca se reescriben versiones guardadas).

## 8. Resultados (conjunto de prueba, 1.409 clientes)
| Modelo | Umbral | Accuracy | Recall | F1 | ROC-AUC | Detectados / no detectados / falsas alarmas | Costo* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Base (Dummy) | 0,50 | 0,735 | 0,000 | 0,000 | 0,500 | 0 / 374 / 0 | 1.870 |
| **v1** Regresión Logística | 0,50 | 0,738 | 0,783 | 0,614 | 0,842 | 293 / 81 / 288 | 693 |
| **v2** mismo modelo, umbral por costo | 0,37 | 0,685 | 0,898 | 0,602 | 0,842 | 336 / 38 / 406 | 596 |

**Cómo leer v1 frente a v2.** Comparten semilla (42), CSV y partición. El `metrics.json` de v2 incluye `modelo_con_umbral_0.5`, idéntico al de v1 (F1 0,6136; costo 693): es el **mismo modelo** y el AUC no cambia (0,8415). La diferencia es solo el umbral, elegido con validación cruzada dentro del entrenamiento: v2 reduce el costo (596) a cambio de más falsas alarmas; no es una mejora del modelo. Si cambiaran la semilla o los datos de prueba, las métricas dejarían de ser comparables y habría que explicarlo antes de interpretarlas.

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

**Evidencia de ejecución real (Git Bash + Docker Engine en WSL 2):**

1. **Construcción de la imagen Docker (`modelos-fase2:1.0`):**
```text
$ docker build -t modelos-fase2:1.0 --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
[+] Building 144.5s (15/15) FINISHED                                                      docker:desktop-linux
 => [internal] load build definition from Dockerfile                                                      2.4s
 => [internal] load metadata for docker.io/library/python:3.13-slim                                      5.5s
 => [ 1/10] FROM docker.io/library/python:3.13-slim                                                     17.2s
 => [ 4/10] RUN pip install --no-cache-dir -r requirements.txt                                           84.4s
 => exporting to image                                                                                   31.6s
 => => naming to docker.io/library/modelos-fase2:1.0                                                     0.0s
```

2. **Entrenamiento en contenedor con volumen persistente (`models/v-docker`):**
```bash
$ MSYS_NO_PATHCONV=1 docker run --rm \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/models:/app/models" \
  modelos-fase2:1.0 python train.py \
  --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv --target Churn --output-dir models --version v-docker --seed 42
```
**Salida obtenida:**
```text
Versión 'v-docker' entrenada con 5634 clientes y evaluada con 1409 (umbral = 0.5).
                accuracy    recall        f1   roc_auc     costo
Baseline            0.7346    0.0000    0.0000    0.5000    1870.0
Modelo              0.7381    0.7834    0.6136    0.8415     693.0
Abandonos detectados: 293 de 374 | no detectados: 81 | falsas alarmas: 288
Archivos generados:
  models/v-docker/modelo.joblib
  models/v-docker/metrics.json
  models/v-docker/metadata.json
```

3. **Predicción en contenedor con volumen persistente (`outputs/docker_predicciones.csv`):**
```bash
$ MSYS_NO_PATHCONV=1 docker run --rm \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/models:/app/models:ro" \
  -v "$PWD/outputs:/app/outputs" \
  modelos-fase2:1.0 python predict.py \
  --model models/v-docker/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/docker_predicciones.csv
```
**Salida obtenida:**
```text
AVISO: 3 valores de TotalCharges vacíos imputados con 0 (regla de la versión guardada).
Versión 'v-docker': 9 predicciones (3 con riesgo de abandono, umbral 0.5).
Resultados en: outputs/docker_predicciones.csv
```

4. **Evidencia de persistencia en el sistema de archivos del anfitrión:**
```bash
$ ls models/v-docker outputs
models/v-docker:
metadata.json  metrics.json  modelo.joblib

outputs:
docker_predicciones.csv  predicciones_v1.csv  predicciones_v2.csv

$ head -3 outputs/docker_predicciones.csv
customerID,prediccion,prediccion_etiqueta,probabilidad_churn
7590-VHVEG,1,Churn,0.804827
5575-GNVDE,0,No Churn,0.118086
```

### 9.1 Recorrido desde un directorio limpio (`verificar_recorrido.sh`)
Desde `fase-2/` (Git Bash o Linux/macOS). Recrea `.verificacion/`, entrena una versión nueva, la carga, predice sobre `data/ejemplo_prediccion.csv`, muestra commit y entorno, y comprueba que repetir el identificador se rechaza sin cambiar el modelo:
```bash
bash verificar_recorrido.sh v-verificacion
bash verificar_recorrido.sh v-verificacion --docker   # además repite el entrenamiento en Docker y compara el commit
```
**Salida obtenida en local (Windows, Git Bash, Python 3.13, commit `42df7fb`):**
```text
== 1. Entrenar 'v-verificacion' en un directorio limpio
Versión 'v-verificacion' entrenada con 5634 clientes y evaluada con 1409 (umbral = 0.5).
                  accuracy    recall        f1   roc_auc     costo
Baseline            0.7346    0.0000    0.0000    0.5000    1870.0
Modelo              0.7381    0.7834    0.6136    0.8415     693.0
Abandonos detectados: 293 de 374 | no detectados: 81 | falsas alarmas: 288
Archivos generados:
  C:\Users\USUARIO\Desktop\telco-customer-churn-ml\fase-2\.verificacion\models\v-verificacion\modelo.joblib
  C:\Users\USUARIO\Desktop\telco-customer-churn-ml\fase-2\.verificacion\models\v-verificacion\metrics.json
  C:\Users\USUARIO\Desktop\telco-customer-churn-ml\fase-2\.verificacion\models\v-verificacion\metadata.json

== 2. Archivos de la versión
total 20
-rw-r--r-- 1 USUARIO 197121 2119 Nov 22 17:10 metadata.json
-rw-r--r-- 1 USUARIO 197121 1098 Nov 22 17:10 metrics.json
-rw-r--r-- 1 USUARIO 197121 8683 Nov 22 17:10 modelo.joblib

== 3. Cargar el artefacto y predecir sobre un CSV pequeño
AVISO: 3 valores de TotalCharges vacíos imputados con 0 (regla de la versión guardada).
Versión 'v-verificacion': 9 predicciones (3 con riesgo de abandono, umbral 0.5).
Resultados en: C:/Users/USUARIO/Desktop/telco-customer-churn-ml/fase-2/.verificacion/outputs/predicciones.csv
customerID,prediccion,prediccion_etiqueta,probabilidad_churn
7590-VHVEG,1,Churn,0.804827
5575-GNVDE,0,No Churn,0.118086
3668-QPYBK,1,Churn,0.532107

== 4. Metadatos: commit y entorno
commit en metadata.json : 42df7fbd65d114a3666e2db33b9da288524e6eb3
commit actual (HEAD)    : 42df7fbd65d114a3666e2db33b9da288524e6eb3
python / scikit-learn   : 3.13.15 / 1.9.1
semilla / datos sha256  : 42 / 88be4b93fbe0

== 5. Repetir el identificador no reemplaza la versión
Rechazado con código de salida 1 -> ERROR: La versin 'v-verificacion' ya existe en 'C:/Users/USUARIO/Desktop/telco-customer-churn-ml/fase-2/.verificacion/models'. No se sobrescribe: use otro identificador (p. ej. la siguiente versin).
modelo.joblib intacto (sha256 8f5b6e7e40c0)

Recorrido completado. Evidencia en: /c/Users/USUARIO/Desktop/telco-customer-churn-ml/fase-2/.verificacion
```
(En el paso 5, "versin" sin tilde es solo la codificación de la consola de Windows; el código del programa escribe "versión".)

**Salida obtenida con `--docker`:** `<<COMPLETAR: salida de --docker (Norbey); "commit en metadata.json" debe coincidir con "commit actual (HEAD)">>`

### 9.2 Trazabilidad del commit en Docker
El `Dockerfile` recibe el commit con `--build-arg` y lo expone como `GIT_COMMIT`; el entrenamiento lo copia a `metadata.json`:
```bash
docker build -t modelos-fase2:1.0 --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
MSYS_NO_PATHCONV=1 docker run --rm -v "$PWD/data:/app/data:ro" -v "$PWD/models:/app/models" modelos-fase2:1.0 python train.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv --target Churn --output-dir models --version v-docker-2 --seed 42
python -c "import json; print(json.load(open('models/v-docker-2/metadata.json'))['codigo'])"
```
Debe imprimir el commit de `git rev-parse HEAD` (sin el argumento quedaría `no-disponible`).

## 10. Pruebas
```bash
python -m pytest
```
Resultado esperado: **`31 passed`** (dentro de Docker se omiten las dos pruebas que leen el `Dockerfile` y `.gitattributes`).

| Archivo | Verifica |
| :--- | :--- |
| `test_entrenamiento.py` | genera artefacto, métricas y metadatos; partición sin fuga; umbral solo con train; error si falta el objetivo |
| `test_prediccion.py` | predicción por fila en orden; igual al modelo original; CSV original (espacios y 0/1); nulos; cliente nuevo; categorías desconocidas; columna ausente; tipos inválidos |
| `test_versionamiento.py` | segunda versión conserva la primera; id repetido no sobrescribe; id inválido |
| `test_trazabilidad.py` | el `Dockerfile` declara `ARG`/`ENV GIT_COMMIT`; el commit de la variable llega a `metadata.json` (función y línea de comandos); sin variable ni Git queda `no-disponible`; `.gitattributes` fija LF |
| `test_cli.py` | `--help`, entrenar y predecir por CLI, código de salida ≠ 0 y sin archivo ante error |

Usan datos sintéticos pequeños y directorios temporales; sin red.

## 11. Estructura
`train.py` y `predict.py` (CLI) · `src/` (config, data, preprocessing, modeling, evaluation, versioning, training, inference) · `tests/` · `data/` · `models/` · `outputs/` · `Dockerfile` · `.dockerignore` · `requirements.txt` · `verificar_recorrido.sh`.

## 12. Fuentes
Documentación de scikit-learn (Pipeline, ColumnTransformer, LogisticRegression), pandas, pytest y Docker. No se adaptó código externo.
