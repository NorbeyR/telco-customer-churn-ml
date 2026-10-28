"""Pruebas 2, 3 y 4 - Carga y predicción, valores faltantes, entrada inválida (contrato de entrada)."""
import numpy as np
import pandas as pd
import pytest

from src.errors import InvalidValueError, MissingColumnsError, ModelNotFoundError
from src.inference import predict_dataframe, run_prediction
from src.versioning import load_bundle
from tests.conftest import make_raw


def test_modelo_cargado_reproduce_predicciones_del_modelo_original(trained, new_csv):
    out, res = trained
    bundle = load_bundle(out / "v1" / "modelo.joblib")
    df = pd.read_csv(new_csv)
    result, _ = predict_dataframe(bundle, df)
    assert len(result) == len(df)                                   # una predicción por fila
    assert result["customerID"].tolist() == df["customerID"].tolist()   # orden conservado
    # coincide con el Pipeline en memoria (misma instancia serializada)
    np.testing.assert_allclose(result["probabilidad_churn"],
                               bundle["pipeline"].predict_proba(df)[:, 1], atol=1e-6)
    assert set(result["prediccion"]) <= {0, 1}


def test_acepta_directamente_el_csv_original_con_espacios_y_0_1(trained, tmp_path):
    """Corrección Fase 1: TotalCharges con espacios y SeniorCitizen 0/1 ya no fallan."""
    out, _ = trained
    df = make_raw(n=8, seed=3, with_target=False)
    df.loc[0, ["tenure", "TotalCharges"]] = [0, " "]
    assert df["SeniorCitizen"].isin([0, 1]).all()
    path = tmp_path / "in.csv"
    df.to_csv(path, index=False)
    res = run_prediction(out / "v1", path, tmp_path / "o" / "p.csv")
    assert res["n_filas"] == 8
    assert pd.read_csv(tmp_path / "o" / "p.csv").shape[0] == 8


def test_totalcharges_nulo_se_imputa_con_cero(trained):
    out, _ = trained
    bundle = load_bundle(out / "v1")
    base = make_raw(n=5, seed=11, with_target=False)
    base["tenure"] = 0
    variantes = []
    for valor in (" ", "", np.nan, "0"):
        d = base.copy()
        d["TotalCharges"] = valor
        variantes.append(predict_dataframe(bundle, d)[0]["probabilidad_churn"].to_numpy())
    for v in variantes[1:]:
        np.testing.assert_allclose(v, variantes[0])               # todas equivalen a imputar 0


def test_cliente_nuevo_categoria_desconocida_y_columnas_extra(trained):
    out, _ = trained
    bundle = load_bundle(out / "v1")
    df = make_raw(n=3, seed=5, with_target=False)
    df["tenure"] = 0
    df["TotalCharges"] = " "
    df.loc[1, "PaymentMethod"] = "Criptomoneda"
    df["columna_extra"] = 1
    df = df[df.columns[::-1]]                                      # orden de columnas distinto
    result, rep = predict_dataframe(bundle, df)
    assert len(result) == 3 and result["probabilidad_churn"].between(0, 1).all()
    assert rep["categorias_no_vistas"] == {"PaymentMethod": ["Criptomoneda"]}
    assert "columna_extra" in rep["columnas_extra_ignoradas"]


def test_nulo_en_categorica_no_elimina_filas(trained):
    out, _ = trained
    df = make_raw(n=4, seed=9, with_target=False)
    df.loc[2, "Contract"] = np.nan
    result, rep = predict_dataframe(load_bundle(out / "v1"), df)
    assert len(result) == 4 and rep["nulos_en_categoricas"] == {"Contract": 1}


def test_columna_requerida_ausente_da_error_claro(trained):
    out, _ = trained
    df = make_raw(n=4, seed=2, with_target=False).drop(columns=["Contract"])
    with pytest.raises(MissingColumnsError, match="Contract"):
        predict_dataframe(load_bundle(out / "v1"), df)


@pytest.mark.parametrize("col,valor,fragmento", [
    ("TotalCharges", "abc", "TotalCharges"),
    ("tenure", "mucho", "tenure"),
    ("MonthlyCharges", -5, "negativos"),
    ("tenure", np.nan, "nulos"),
    ("SeniorCitizen", 7, "SeniorCitizen"),
])
def test_tipos_o_valores_incompatibles_dan_error_claro(trained, col, valor, fragmento):
    out, _ = trained
    df = make_raw(n=4, seed=2, with_target=False)
    df[col] = df[col].astype(object)
    df.loc[1, col] = valor
    with pytest.raises(InvalidValueError, match=fragmento):
        predict_dataframe(load_bundle(out / "v1"), df)


def test_modelo_inexistente(tmp_path, new_csv):
    with pytest.raises(ModelNotFoundError):
        run_prediction(tmp_path / "no_existe", new_csv, tmp_path / "o.csv")
