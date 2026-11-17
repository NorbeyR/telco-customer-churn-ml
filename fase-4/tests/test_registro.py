"""Fase 4 - Registro de solicitudes a /predict: una fila por intento, persistente entre ejecuciones."""
import csv

import pytest
from fastapi.testclient import TestClient

import api as api_module
from api import Settings, create_app
from src.training import run_training
from tests.conftest import make_raw


@pytest.fixture
def cfg(tmp_path, data_csv):
    models = tmp_path / "models"
    run_training(data_csv, "Churn", models, "v1", seed=42)
    return Settings(model_version="v1", models_dir=str(models), data_path=str(data_csv),
                    log_path=str(tmp_path / "logs" / "predicciones_log.csv"))


@pytest.fixture
def record():
    out = make_raw(n=6, seed=21, with_target=False).iloc[1].to_dict()
    out.update(SeniorCitizen=int(out["SeniorCitizen"]), tenure=int(out["tenure"]),
               MonthlyCharges=float(out["MonthlyCharges"]))
    return out


def leer(cfg):
    with open(cfg.log_path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_dos_validas_y_una_invalida_generan_tres_filas(cfg, record):
    c = TestClient(create_app(cfg))
    assert c.post("/predict", json=record).status_code == 200
    assert c.post("/predict", json={**record, "tenure": 5}).status_code == 200
    sin_campo = {k: v for k, v in record.items() if k != "Contract"}
    assert c.post("/predict", json=sin_campo).status_code == 422

    rows = leer(cfg)
    assert [r["estado_http"] for r in rows] == ["200", "200", "422"]
    assert [r["version"] for r in rows] == ["v1", "v1", ""]
    assert rows[0]["prediccion"] in ("Churn", "No Churn") and rows[2]["prediccion"] == ""
    assert list(rows[0]) == ["fecha_hora", "estado_http", "version", "prediccion"]
    assert len(rows[0]["fecha_hora"]) == 19                      # AAAA-MM-DDTHH:MM:SS


def test_cada_tipo_de_rechazo_deja_una_sola_fila(cfg, record, monkeypatch):
    c = TestClient(create_app(cfg), raise_server_exceptions=False)
    c.post("/predict", content=b"{roto", headers={"content-type": "application/json"})   # 400
    c.post("/predict", json=[1, 2])                                                        # 400
    c.post("/predict", json={**record, "tenure": "mucho"})                                 # 422
    monkeypatch.setattr(api_module, "predict_dataframe", lambda *a, **k: 1 / 0)          # 500 inesperado
    c.post("/predict", json=record)
    assert [r["estado_http"] for r in leer(cfg)] == ["400", "400", "422", "500"]


def test_modelo_no_disponible_se_registra_como_503_sin_version(cfg, record):
    cfg.model_version = "v9"
    c = TestClient(create_app(cfg))
    assert c.post("/predict", json=record).status_code == 503
    row = leer(cfg)[0]
    assert row["estado_http"] == "503" and row["version"] == "" and row["prediccion"] == ""


def test_el_registro_se_conserva_entre_ejecuciones_y_el_encabezado_no_se_repite(cfg, record):
    TestClient(create_app(cfg)).post("/predict", json=record)
    TestClient(create_app(cfg)).post("/predict", json=record)            # "reinicio" de la API
    with open(cfg.log_path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    assert len(lines) == 3 and lines.count("fecha_hora,estado_http,version,prediccion") == 1


def test_otras_rutas_no_se_registran(cfg):
    c = TestClient(create_app(cfg))
    c.get("/health"); c.get("/versions"); c.post("/train", json={"version": "v1"})
    import os
    assert not os.path.exists(cfg.log_path)


def test_la_version_registrada_es_la_que_predijo(cfg, record, tmp_path, data_csv):
    run_training(data_csv, "Churn", cfg.models_dir, "v2", seed=7, optimize_threshold=True, cv_folds=3)
    cfg.model_version = "v2"
    body = TestClient(create_app(cfg)).post("/predict", json=record).json()
    assert body["version"] == "v2" and leer(cfg)[0]["version"] == "v2"
