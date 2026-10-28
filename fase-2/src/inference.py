"""Predicción con una versión guardada (sin reentrenar ni reajustar transformaciones)."""
import os
import tempfile
import warnings
from pathlib import Path

import pandas as pd

from .config import ID_COLUMN
from .data import read_csv
from .errors import ChurnError, InvalidValueError, ModelNotFoundError
from .preprocessing import MISSING_TOKEN
from .versioning import load_bundle, sklearn_version


def _categories_seen(pipeline) -> dict:
    cleaner = pipeline.named_steps["cleaner"]
    encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encoder"]
    return dict(zip(cleaner.categorical_features, encoder.categories_))


def build_report(pipeline, df: pd.DataFrame) -> dict:
    """Avisos no fatales: columnas extra, nulos tratados y categorías no vistas."""
    cleaner = pipeline.named_steps["cleaner"]
    cleaned = cleaner.transform(df)
    required = set(cleaner.required_columns)
    extra = [c for c in df.columns if c not in required]
    unknown, missing_cat = {}, {}
    for col, seen in _categories_seen(pipeline).items():
        values = cleaned[col]
        n_missing = int((values == MISSING_TOKEN).sum())
        if n_missing:
            missing_cat[col] = n_missing
        new = sorted(set(values.unique()) - set(seen) - {MISSING_TOKEN})
        if new:
            unknown[col] = new
    total_null = int(cleaned["TotalCharges"].isna().sum()) if "TotalCharges" in cleaned else 0
    return {"columnas_extra_ignoradas": extra, "categorias_no_vistas": unknown,
            "nulos_en_categoricas": missing_cat, "totalcharges_imputados": total_null}


def predict_dataframe(bundle: dict, df: pd.DataFrame, threshold=None, id_column=ID_COLUMN):
    """Una predicción por fila, conservando el orden. Devuelve (resultado, reporte)."""
    if df.empty:
        raise InvalidValueError("El archivo de entrada no contiene filas.")
    thr = bundle["threshold"] if threshold is None else float(threshold)
    if not 0 < thr < 1:
        raise ChurnError("El umbral debe estar entre 0 y 1 (exclusivo).")
    pipeline = bundle["pipeline"]
    report = build_report(pipeline, df)                       # también valida columnas y tipos
    proba = pipeline.predict_proba(df)[:, list(pipeline.classes_).index(1)]
    pred = (proba >= thr).astype(int)

    result = pd.DataFrame({
        "prediccion": pred,
        "prediccion_etiqueta": [bundle["labels"][int(p)] for p in pred],
        "probabilidad_churn": proba.round(6),
    })
    if id_column and id_column in df.columns:
        result.insert(0, id_column, df[id_column].to_numpy())
    result.attrs["umbral"] = thr
    return result, report


def _atomic_write_csv(df: pd.DataFrame, output) -> None:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".pred-", suffix=".tmp", dir=out.parent)
    os.close(fd)
    try:
        df.to_csv(tmp, index=False, encoding="utf-8")
        os.replace(tmp, out)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def run_prediction(model, input_path, output_path, threshold=None, id_column=ID_COLUMN) -> dict:
    """Carga la versión, valida la entrada, predice y escribe el CSV solo si todo salió bien."""
    bundle = load_bundle(model)
    if bundle.get("sklearn_version") != sklearn_version():
        warnings.warn(f"El modelo se entrenó con scikit-learn {bundle.get('sklearn_version')} y se "
                      f"está usando {sklearn_version()}; use las versiones de requirements.txt.")
    df = read_csv(input_path)
    result, report = predict_dataframe(bundle, df, threshold, id_column)
    report["columnas_extra_ignoradas"] = [c for c in report["columnas_extra_ignoradas"] if c != id_column]
    _atomic_write_csv(result, output_path)
    return {"n_filas": len(result), "n_churn": int(result["prediccion"].sum()),
            "umbral": result.attrs["umbral"], "version": bundle.get("version"),
            "salida": str(output_path), "reporte": report}
