"""Prueba 1 - Entrenamiento y almacenamiento."""
import json

import joblib
import pandas as pd

from src.config import METADATA_FILENAME, METRICS_FILENAME, MODEL_FILENAME


def test_entrenamiento_genera_artefacto_metricas_y_metadatos(trained):
    out, res = trained
    folder = out / "v1"
    for name in (MODEL_FILENAME, METRICS_FILENAME, METADATA_FILENAME):
        assert (folder / name).is_file()

    bundle = joblib.load(folder / MODEL_FILENAME)
    assert {"pipeline", "threshold"} <= set(bundle)
    assert hasattr(bundle["pipeline"], "predict_proba")

    metrics = json.loads((folder / METRICS_FILENAME).read_text(encoding="utf-8"))
    assert {"modelo", "baseline"} <= set(metrics)
    assert 0.0 <= metrics["modelo"]["roc_auc"] <= 1.0
    assert metrics["baseline"]["recall_churn"] == 0.0         # el baseline nunca detecta abandonos
    assert metrics["modelo"]["recall_churn"] > 0.0

    meta = json.loads((folder / METADATA_FILENAME).read_text(encoding="utf-8"))
    for key in ("version", "fecha_creacion_utc", "variable_objetivo", "algoritmo", "semilla",
                "datos", "particion", "entorno", "codigo", "variables_entrada", "parametros_modelo"):
        assert key in meta
    assert meta["version"] == "v1" and meta["semilla"] == 42
    assert meta["particion"]["n_entrenamiento"] + meta["particion"]["n_prueba"] == 300
    assert meta["datos"]["n_observaciones"] == 300


def test_particion_estratificada_y_sin_fuga(data_csv):
    from src.data import load_labeled_data
    from src.modeling import split_train_test
    X, y, groups = load_labeled_data(data_csv, "Churn")
    X_tr, X_te, y_tr, y_te, _, _ = split_train_test(X, y, 0.2, 42, groups)
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert abs(y_tr.mean() - y_te.mean()) < 0.03             # estratificación
    assert "customerID" not in X.columns and "Churn" not in X.columns


def test_umbral_optimizado_usa_solo_entrenamiento(tmp_path, data_csv):
    from src.training import run_training
    res = run_training(data_csv, "Churn", tmp_path / "m", "vt", optimize_threshold=True, cv_folds=3)
    info = res["metrics"]["umbral_seleccion"]
    assert info["metodo"] == "cv_costo_minimo"
    assert info["costo_cv_umbral_elegido"] <= info["costo_cv_umbral_0.5"]
    assert 0.05 <= res["metrics"]["modelo"]["umbral"] <= 0.95
    assert "modelo_con_umbral_0.5" in res["metrics"]


def test_datos_con_objetivo_ausente_fallan_claro(tmp_path, raw_df):
    import pytest
    from src.errors import MissingColumnsError
    from src.training import run_training
    p = tmp_path / "sin_objetivo.csv"
    raw_df.drop(columns=["Churn"]).to_csv(p, index=False)
    with pytest.raises(MissingColumnsError, match="Churn"):
        run_training(p, "Churn", tmp_path / "m", "v1")
    assert not (tmp_path / "m" / "v1").exists()
