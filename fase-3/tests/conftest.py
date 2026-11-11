"""Datos sintéticos pequeños con el esquema y el FORMATO del CSV original de IBM Telco
(SeniorCitizen 0/1, TotalCharges como texto con espacios para clientes nuevos)."""
import numpy as np
import pandas as pd
import pytest

from src.config import CATEGORICAL_FEATURES

CATS = {
    "gender": ["Female", "Male"], "Partner": ["Yes", "No"], "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"], "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
}


def make_raw(n=300, seed=0, with_target=True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"customerID": [f"C-{i:04d}" for i in range(n)]})
    for c, vals in CATS.items():
        df[c] = rng.choice(vals, n)
    df["SeniorCitizen"] = rng.integers(0, 2, n)
    df["tenure"] = rng.integers(0, 72, n)
    df.loc[:4, "tenure"] = 0                         # clientes nuevos
    df["MonthlyCharges"] = rng.uniform(20, 110, n).round(2)
    total = (df["tenure"] * df["MonthlyCharges"]).round(2)
    df["TotalCharges"] = total.astype(str)
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "  # como en el CSV original
    z = -0.04 * df["tenure"] + 1.2 * (df["Contract"] == "Month-to-month") - 0.6
    p = 1 / (1 + np.exp(-z))
    df["Churn"] = np.where(rng.random(n) < p, "Yes", "No")
    cols = ["customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
            "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
            "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"]
    df = df[cols]
    return df if with_target else df.drop(columns=["Churn"])


@pytest.fixture
def raw_df():
    return make_raw()


@pytest.fixture
def data_csv(tmp_path, raw_df):
    path = tmp_path / "datos.csv"
    raw_df.to_csv(path, index=False)
    return path


@pytest.fixture
def new_csv(tmp_path):
    path = tmp_path / "nuevos.csv"
    make_raw(n=12, seed=7, with_target=False).to_csv(path, index=False)
    return path


@pytest.fixture
def trained(tmp_path, data_csv):
    """Entrena una versión pequeña en un directorio temporal."""
    from src.training import run_training
    out = tmp_path / "models"
    res = run_training(data_csv, "Churn", out, "v1", seed=42)
    return out, res
