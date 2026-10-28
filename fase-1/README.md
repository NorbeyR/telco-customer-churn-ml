# Predicción de Cancelación de Clientes (IBM Telco Customer Churn)
**Asignatura:** Modelos y Simulación de Sistemas I  

---

## Integrantes del Equipo
* [Heidy Dayana Gamboa Chaverra] 
* [Santiago Restrepo Rodriguez] 
* [Norbey David Rincón Bedoya]

---

## Descripción del Problema
La pérdida de clientes (*Churn*) es uno de los mayores desafíos para las empresas de telecomunicaciones. Retener a un cliente existente es significativamente más económico que adquirir uno nuevo. Este proyecto aborda la identificación temprana de clientes en riesgo de cancelar sus servicios para permitir intervenciones proactivas de fidelización.

---

## Fuente del Conjunto de Datos
* **Dataset:** IBM Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
* **Naturaleza de los datos:** IBM presenta este conjunto como una muestra de una empresa **ficticia**; sirve con fines didácticos, no acredita registros reales ni permite extrapolar los resultados a operadores colombianos.
* **Dimensiones:** 7.043 observaciones y 21 variables (20 predictoras y 1 variable objetivo `Churn`).
* **Variable Objetivo:** `Churn` (Yes / No) con un desbalance de clases del ~26.5% de abandonos.

---

## Objetivo del Modelo
Desarrollar y evaluar un modelo de aprendizaje automático supervisado capaz de predecir la probabilidad de que un cliente cancele su suscripción, priorizando la detección efectiva de la clase positiva (`Churn`).

---

## Metodología y Algoritmo Utilizado
* **Preprocesamiento:** Imputación constante de 0 para valores nulos en `TotalCharges` (`tenure = 0`), escalamiento estándar (`StandardScaler`) para numéricas y codificación One-Hot (`OneHotEncoder`) para categóricas. Todo encapsulado en un `ColumnTransformer` para garantizar cero fuga de información (*Data Leakage*).
* **Estrategia de Validación:** Separación 80% entrenamiento / 20% prueba estratificada por `Churn` (`random_state=42`).
* **Algoritmo Seleccionado:** **Regresión Logística** con ajuste de pesos `class_weight='balanced'` dentro de un `Pipeline` de Scikit-Learn.

---

## Métricas y Resultados Principales
Dado el desbalance de clases, el desempeño se evaluó principalmente mediante **F1-Score**, **Recall** y **ROC-AUC** sobre la clase `Churn`.

| Modelo | Accuracy | Recall (Churn) | F1-Score (Churn) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Modelo Base (Dummy - Most Frequent)** | 73.5% | 0.00 | 0.00 | 0.50 |
| **Regresión Logística (Entrenado)** | **73.8%** | **~0.78** | **~0.61** | **~0.84** |

### Conclusiones Principales:
1. **Superación del Baseline:** El modelo base acertaba el 73.5% por sesgo mayoritario, pero no detectaba ningún cliente en riesgo. La Regresión Logística logra detectar aproximadamente el **78% de los clientes que realmente abandonan**.
2. **Capacidad Discriminatoria:** Alcanza un **ROC-AUC de ~0.84**, demostrando un ordenamiento probabilístico sólido de los clientes según su nivel de riesgo.
3. **Persistencia:** El artefacto `modelo.joblib` guarda el preprocesador y el clasificador juntos. **Limitación:** aún no acepta directamente el CSV original (las conversiones de `TotalCharges` y `SeniorCitizen` se hicieron en el notebook); esto se corrige en la Fase 2 (`../fase-2/`), por lo que no se considera listo para producción.

---

## Instrucciones para Ejecutar el Notebook

### 1. Prerrequisitos
Python 3.9 o superior.

### 2. Clonar el repositorio, entrar a `fase-1` e instalar dependencias
```bash
git clone https://github.com/NorbeyR/telco-customer-churn-ml.git
cd telco-customer-churn-ml/fase-1

python -m venv venv
source venv/bin/activate      # En Windows: venv\Scripts\activate

# Librerías del notebook + entorno de notebooks
pip install pandas numpy scipy matplotlib seaborn scikit-learn joblib jupyter ipykernel
```

### 3. Archivos necesarios
`WA_Fn-UseC_-Telco-Customer-Churn.csv` y `notebook.ipynb` están en `fase-1/` (la misma carpeta desde la que se ejecuta el notebook).

### 4. Ejecución
```bash
jupyter notebook notebook.ipynb
```
o abra `notebook.ipynb` en VS Code y ejecute las celdas de arriba hacia abajo.

*Nota: el notebook usa `SEED = 42` para reproducir los resultados.*
