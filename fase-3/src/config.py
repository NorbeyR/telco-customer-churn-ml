"""Constantes del problema: esquema del conjunto IBM Telco Customer Churn."""

TARGET = "Churn"
POSITIVE_LABEL = "Yes"      # el cliente abandonó  -> clase 1
NEGATIVE_LABEL = "No"       # el cliente permaneció -> clase 0
ID_COLUMN = "customerID"    # identificador: no se usa para predecir

NUMERIC_FEATURES = ("tenure", "MonthlyCharges", "TotalCharges")
# Únicas variables numéricas que admiten nulos / espacios (clientes nuevos, tenure = 0).
NULLABLE_NUMERIC = ("TotalCharges",)

CATEGORICAL_FEATURES = (
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
)
# Categóricas que llegan codificadas como 0/1 en el CSV original (Yes/No en el modelo).
FLAG_01_COLUMNS = ("SeniorCitizen",)

REQUIRED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Archivos de cada versión del modelo
MODEL_FILENAME = "modelo.joblib"
METRICS_FILENAME = "metrics.json"
METADATA_FILENAME = "metadata.json"

DEFAULT_DATA_SOURCE = (
    "IBM Telco Customer Churn (Kaggle: blastchar/telco-customer-churn); "
    "muestra de una empresa ficticia"
)
