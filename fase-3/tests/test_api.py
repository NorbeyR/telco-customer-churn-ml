"""Pruebas de la API (Fase 3): /health, /predict válido, campo ausente, tipos, modelo no disponible y /train."""
import pytest
from fastapi.testclient import TestClient

from api import Settings, create_app
from src.training import run_training
from tests.conftest import make_raw


@pytest.fixture
def models_dir(tmp_path, data_csv):
    """Modelo pequeño v1 entrenado en un directorio temporal (sin tocar models/ del repo)."""
    out = tmp_path / "models"
    run_training(data_csv, "Churn", out, "v1", seed=42)
    return out


@pytest.fixture
def client(models_dir, data_csv):
    cfg = Settings(model_version="v1", models_dir=str(models_dir), data_path=str(data_csv))
    return TestClient(create_app(cfg))


@pytest.fixture
def record():
    """Un registro válido: una fila del CSV con el formato original, sin la variable objetivo."""
    row = make_raw(n=6, seed=21, with_target=False).iloc[1]
    out = row.to_dict()
    out["SeniorCitizen"] = int(out["SeniorCitizen"])
    out["tenure"] = int(out["tenure"])
    out["MonthlyCharges"] = float(out["MonthlyCharges"])
    return out


# ---------------------------------------------------------------- las 3 pruebas pedidas
def test_health_responde_200(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["version_activa"] == "v1" and body["modelo_disponible"] is True


def test_predict_valido_devuelve_prediccion_y_version(client, record):
    r = client.post("/predict", json=record)
    assert r.status_code == 200
    body = r.json()
    assert body["prediccion"] in ("Churn", "No Churn")
    assert body["clase"] in (0, 1)
    assert 0.0 <= body["probabilidad_churn"] <= 1.0
    assert body["version"] == "v1"


def test_predict_con_campo_obligatorio_ausente_se_rechaza(client, record):
    del record["Contract"]
    r = client.post("/predict", json=record)
    assert r.status_code == 422
    body = r.json()
    assert "Contract" in body["detail"] and body["campos_faltantes"] == ["Contract"]


# ------------------------------------------------------------------- casos adicionales
def test_predict_acepta_nulos_y_espacios_en_totalcharges(client, record):
    for valor in (None, " ", ""):
        r = client.post("/predict", json={**record, "tenure": 0, "TotalCharges": valor})
        assert r.status_code == 200, r.text
    assert "totalcharges_imputados" in r.json()["advertencias"]


def test_predict_no_requiere_variable_objetivo_y_ignora_extras(client, record):
    r = client.post("/predict", json={**record, "customerID": "X-1", "campo_extra": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["advertencias"]["columnas_extra_ignoradas"] == ["campo_extra"]   # customerID no es "extra"
    assert body["customerID"] == "X-1"


@pytest.mark.parametrize("campo,valor", [("tenure", "mucho"), ("MonthlyCharges", -3), ("SeniorCitizen", 9),
                                         ("tenure", None), ("TotalCharges", "abc"), ("Contract", {"a": 1})])
def test_predict_tipos_invalidos_dan_422_con_el_nombre_del_campo(client, record, campo, valor):
    r = client.post("/predict", json={**record, campo: valor})
    assert r.status_code == 422
    assert campo in r.json()["detail"]


def test_predict_json_invalido_o_no_objeto_da_400(client):
    assert client.post("/predict", content=b"{no es json", headers={"content-type": "application/json"}).status_code == 400
    assert client.post("/predict", json=[1, 2, 3]).status_code == 400
    assert client.post("/predict").status_code == 400


def test_modelo_no_disponible_da_503_con_instrucciones(models_dir, data_csv, record):
    cfg = Settings(model_version="v9", models_dir=str(models_dir), data_path=str(data_csv))
    c = TestClient(create_app(cfg))
    r = c.post("/predict", json=record)
    assert r.status_code == 503
    assert "MODEL_VERSION" in r.json()["detail"] and r.json()["versiones_disponibles"] == ["v1"]
    h = c.get("/health").json()                      # la API sigue viva aunque falte el modelo
    assert h["status"] == "ok" and h["modelo_disponible"] is False


def test_train_guarda_version_nueva_y_conserva_las_anteriores(client, models_dir, record):
    r = client.post("/train")                         # cuerpo vacío
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["version"] == "v2" and (models_dir / "v2" / "modelo.joblib").is_file()
    assert (models_dir / "v1" / "modelo.joblib").is_file()
    assert client.get("/versions").json()["versiones_disponibles"] == ["v1", "v2"]
    assert client.post("/predict", json=record).json()["version"] == "v1"   # la activa no cambia


def test_train_con_version_repetida_da_409(client):
    r = client.post("/train", json={"version": "v1"})
    assert r.status_code == 409 and "ya existe" in r.json()["detail"]


def test_train_con_campos_no_admitidos_da_422(client):
    assert client.post("/train", json={"colour": 1}).status_code == 422
