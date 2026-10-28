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
| **Regresión Logística (Entrenado)** | **~74.5%** | **~0.78** | **~0.61** | **~0.84** |

### Conclusiones Principales:
1. **Superación del Baseline:** El modelo base acertaba el 73.5% por sesgo mayoritario, pero no detectaba ningún cliente en riesgo. La Regresión Logística logra detectar aproximadamente el **78% de los clientes que realmente abandonan**.
2. **Capacidad Discriminatoria:** Alcanza un **ROC-AUC de ~0.84**, demostrando un ordenamiento probabilístico sólido de los clientes según su nivel de riesgo.
3. **Persistencia:** El artefacto final `modelo.joblib` incluye el pipeline completo de preprocesamiento y clasificación listo para inferencia en producción.

---

## Instrucciones para Ejecutar el Notebook

### 1. Prerrequisitos
Asegúrate de contar con Python 3.9+ instalado en tu sistema.

### 2. Clonar el Repositorio e Instalar Dependencias
```bash
# Clonar el repositorio
git clone https://github.com/NorbeyR/telco-customer-churn-ml.git

# Crear e instalar entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar librerías necesarias
pip install pandas numpy scipy matplotlib seaborn scikit-learn joblib
```

### 3. Archivos Necesarios
Asegúrate de que el archivo de datos `WA_Fn-UseC_-Telco-Customer-Churn.csv` se encuentre en la raíz de la carpeta del proyecto.

### 4. Ejecución del Notebook
Abre el editor (VS Code o Jupyter Lab) y ejecuta el cuaderno ejecutable principal en orden secuencial de arriba a abajo:
```bash
code .
```
O desde la terminal:
```bash
jupyter notebook notebook.ipynb
```
*Nota: El notebook está configurado con una semilla aleatoria (`SEED = 42`) para garantizar reproducibilidad exacta de todos los resultados.*